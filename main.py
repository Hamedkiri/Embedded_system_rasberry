#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI temps réel Raspberry Pi — Détection météo
- PySide6 (Qt) pour l'IHM (responsive, HiDPI, multi-écrans)
- OpenCV pour capture/overlay/vidéo
- ModelRegistry (PyTorch / TFLite / ONNX / PatchGAN)
- Lecture d’enregistrements, transfert, e-mail

Dépendances conseillées :
  pip install PySide6 opencv-python torch torchvision pandas openpyxl onnxruntime tflite-runtime
"""

import os
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

import sys, json, time, glob, queue, datetime, shutil, ssl, smtplib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

import numpy as np
import cv2

# Pandas pour CSV/XLSX ; XLSX nécessite openpyxl
try:
    import pandas as pd
except Exception:
    pd = None

# Torch facultatif si vous n'utilisez pas PyTorch
try:
    import torch
except Exception:
    class _DummyTorch:
        def no_grad(self): return self
        def __enter__(self): return self
        def __exit__(self, *a): pass
        device = "cpu"
        from numpy import ndarray
    torch = _DummyTorch()

from PySide6.QtCore import Qt, QTimer, QObject, Signal, Slot, QThread
from PySide6.QtGui  import QImage, QPixmap, QFont, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QLineEdit,
    QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QMessageBox, QSizePolicy, QDialog, QFormLayout, QDialogButtonBox, QTextEdit
)

# ─────────────────────────────────────────────────────────────
#                  Pré-traitements
# ─────────────────────────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_bgr_imagenet(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return img  # (H,W,3) float32

def preprocess_bgr_tanh(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """Prétraitement [-1,1] utile pour certains GAN."""
    img = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 127.5) - 1.0
    return img  # (H,W,3) float32


# ─────────────────────────────────────────────────────────────
#               Adapters / Registry (chargement modèle)
# ─────────────────────────────────────────────────────────────
class BaseAdapter:
    def __init__(self, tasks: Dict[str, List[str]], device: str = "cpu", input_size: int = 224, preprocess: str = "imagenet"):
        self.tasks = tasks
        self.device = device
        self.input_size = int(input_size)
        self.preprocess = preprocess  # "imagenet" ou "tanh"

    def _prep(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.preprocess == "tanh":
            return preprocess_bgr_tanh(frame_bgr, self.input_size)
        return preprocess_bgr_imagenet(frame_bgr, self.input_size)

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        raise NotImplementedError


class PyTorchMultiTaskAdapter(BaseAdapter):
    """Modèle PyTorch multi-tâches (dict logits par tâche)."""
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224, extra: Optional[Dict[str, Any]]=None, preprocess: str="imagenet"):
        super().__init__(tasks, device, input_size, preprocess)
        self._jit = None
        self._model = None
        # 1) torchscript direct
        try:
            self._jit = torch.jit.load(weights_path, map_location=device)
            self._jit.eval()
        except Exception:
            # 2) fallback vers ton module ResNet50_truncated_module.py
            try:
                from ResNet50_truncated_module import MultiHeadAttentionPerTaskModel, load_best_model
                from torchvision import models
                extra = extra or {}
                base_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                truncate = int(extra.get("truncate_after_layer", 10))
                use_attention = bool(extra.get("use_attention", True))
                attn_token_dim = extra.get("attn_token_dim", None)
                cls_hidden_dims = extra.get("cls_hidden_dims", [])
                cls_num_layers  = int(extra.get("cls_num_layers", 0))
                self._model = MultiHeadAttentionPerTaskModel(
                    base_encoder=base_encoder,
                    truncate_after_layer=truncate,
                    tasks=tasks,
                    device=device,
                    use_attention=use_attention,
                    attn_token_dim=attn_token_dim,
                    cls_hidden_dims=cls_hidden_dims,
                    cls_num_layers=cls_num_layers
                )
                # ⚠️ strict_backbone=False pour tolérer des ckpts hétérogènes
                load_best_model(self._model, weights_path, strict_backbone=False, verbose=False)
                self._model.eval()
            except Exception as e:
                raise RuntimeError(f"Impossible de charger le modèle PyTorch: {e}")

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = self._prep(frame_bgr)
        nchw = np.transpose(nhwc, (2,0,1))[np.newaxis, ...].astype(np.float32)
        inp  = torch.from_numpy(nchw)
        with torch.no_grad():
            if self._jit is not None:
                out = self._jit(inp)
                return out
            else:
                out = self._model(inp)
                return out


class TFLiteMultiHeadAdapter(BaseAdapter):
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224, extra: Optional[Dict[str, Any]]=None, preprocess: str="imagenet"):
        super().__init__(tasks, device, input_size, preprocess)
        extra = extra or {}
        try:
            import tflite_runtime.interpreter as tflite
        except Exception:
            import tensorflow.lite as tflite
        self.interp = tflite.Interpreter(model_path=weights_path, num_threads=int(extra.get("threads", 2)))
        self.interp.allocate_tensors()
        self.input_details  = self.interp.get_input_details()
        self.output_details = self.interp.get_output_details()
        self.output_task_order = extra.get("output_task_order", list(tasks.keys()))

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = self._prep(frame_bgr)
        arr = nhwc[np.newaxis, ...].astype(np.float32)  # [1,H,W,3]
        inp = self.input_details[0]
        # Si le modèle est quantisé uint8 → repasser en [0..255] RGB
        if inp["dtype"] == np.uint8:
            if self.preprocess == "imagenet":
                rgb01 = np.clip((nhwc * _IMAGENET_STD + _IMAGENET_MEAN), 0, 1)
            else:
                rgb01 = np.clip((nhwc + 1.0) / 2.0, 0, 1)
            arr = (rgb01 * 255.0).astype(np.uint8)[np.newaxis, ...]
        self.interp.set_tensor(inp["index"], arr)
        self.interp.invoke()
        outs = [self.interp.get_tensor(od["index"]) for od in self.output_details]

        result = {}
        if len(outs) == len(self.output_task_order):
            for task_name, a in zip(self.output_task_order, outs):
                result[task_name] = torch.from_numpy(a)
        else:
            total = sum(len(v) for v in self.tasks.values())
            flat = outs[0]
            if flat.shape[-1] != total:
                raise RuntimeError("TFLite: sortie incompatible. Configurez 'output_task_order' dans config.")
            off = 0
            for task, cls in self.tasks.items():
                n = len(cls)
                result[task] = torch.from_numpy(flat[:, off:off+n])
                off += n
        return result


class ONNXMultiHeadAdapter(BaseAdapter):
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224, extra: Optional[Dict[str, Any]]=None, preprocess: str="imagenet"):
        super().__init__(tasks, device, input_size, preprocess)
        extra = extra or {}
        import onnxruntime as ort
        providers = extra.get("providers", ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(weights_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.output_task_order = extra.get("output_task_order", list(tasks.keys()))

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = self._prep(frame_bgr)
        nchw = nhwc.transpose(2,0,1)[np.newaxis, ...].astype(np.float32)
        outs = self.sess.run(self.output_names, {self.input_name: nchw})
        result = {}
        if len(outs) == len(self.output_task_order):
            for task_name, a in zip(self.output_task_order, outs):
                result[task_name] = torch.from_numpy(a)
        else:
            total = sum(len(v) for v in self.tasks.values())
            flat = outs[0]
            if flat.shape[-1] != total:
                raise RuntimeError("ONNX: sortie incompatible. Configurez 'output_task_order' dans config.")
            off = 0
            for task, cls in self.tasks.items():
                n = len(cls)
                result[task] = torch.from_numpy(flat[:, off:off+n])
                off += n
        return result


class PatchGANAdapter(BaseAdapter):
    """
    Adapter pour MultiTaskPatchGAN via loader_patchgan_from_manifest.py.
    On s'appuie sur un manifest.json voisin pour récupérer tasks, extra, input_size, preprocess.
    """
    def __init__(self, model_path: str, device="cpu"):
        p = Path(model_path)
        manifest = p.parent / "manifest.json"
        if not manifest.exists():
            raise RuntimeError(f"manifest.json introuvable à côté de {model_path}")

        try:
            from loader_patchgan_from_manifest import create_patchgan_from_manifest
        except Exception as e:
            raise RuntimeError(f"Impossible d'importer loader_patchgan_from_manifest.py : {e}")

        model, tasks = create_patchgan_from_manifest(str(manifest), device=device)
        man = json.loads(manifest.read_text(encoding="utf-8"))
        extra = man.get("extra", {})
        input_size = int(man.get("input_size", extra.get("input_size", 224)))
        preprocess = str(extra.get("preprocess", "tanh")).lower()

        super().__init__(tasks=tasks, device=device, input_size=input_size, preprocess=preprocess)
        self.model = model.eval()

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = self._prep(frame_bgr)
        nchw = np.transpose(nhwc, (2,0,1))[np.newaxis, ...].astype(np.float32)
        x = torch.from_numpy(nchw)
        with torch.no_grad():
            out = self.model(x)  # dict{task: logits}
        return out


class ModelRegistry:
    _MAP = {
        "pytorch":  PyTorchMultiTaskAdapter,
        "tflite":   TFLiteMultiHeadAdapter,
        "onnx":     ONNXMultiHeadAdapter,
        "patchgan": PatchGANAdapter,   # clé interne
    }

    @staticmethod
    def _labels_from_tasks(tasks_dict: Optional[Dict[str, int]]) -> Dict[str, List[str]]:
        if not isinstance(tasks_dict, dict):
            return {"Weather": ["No", "Yes"]}
        out: Dict[str, List[str]] = {}
        for task, n in tasks_dict.items():
            try: n = int(n)
            except Exception: n = 2
            out[task] = [f"{task}_{i}" for i in range(n)]
        return out

    @staticmethod
    def _looks_like_patchgan(man: Dict[str, Any], path_hint: str) -> bool:
        txt = (json.dumps(man, ensure_ascii=False) + " " + path_hint).lower()
        if "patchgan" in txt: return True
        # indices fréquents PatchGAN
        for k in ("patch_size", "attn_tau", "attn_use_se", "ndf", "discriminator"):
            if k in man or k in man.get("extra", {}):
                return True
        return False

    @staticmethod
    def _read_config(model_path: str) -> Dict[str, Any]:
        """
        Retourne un cfg standardisé:
          { type, input_size, tasks, extra, adapter? }
        - Regarde d'abord config.json voisin
        - Puis manifest.json voisin (et peut forcer adapter='patchgan')
        """
        p = Path(model_path); base = p.parent

        # 1) config.json voisin
        cfg_path = base / "config.json"
        if cfg_path.exists():
            try:
                return json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # 2) manifest.json voisin
        mf = base / "manifest.json"
        if mf.exists():
            try:
                man = json.loads(mf.read_text(encoding="utf-8"))
                cfg: Dict[str, Any] = {
                    "type": man.get("type") or "pytorch",
                    "input_size": int(man.get("input_size", 224)),
                    "tasks": man.get("labels") or man.get("tasks") or {},
                    "extra": man.get("extra", {}),
                }
                if man.get("adapter", "").lower() == "patchgan" or ModelRegistry._looks_like_patchgan(man, str(p)):
                    cfg["adapter"] = "patchgan"
                # Si tasks = dict{task:count} → fabriquer labels
                if cfg["tasks"] and isinstance(list(cfg["tasks"].values())[0], int):
                    cfg["tasks"] = ModelRegistry._labels_from_tasks(cfg["tasks"])
                return cfg
            except Exception:
                pass

        # 3) défaut
        return {}

    @staticmethod
    def _infer_type_from_ext(model_path: str) -> Optional[str]:
        ext = Path(model_path).suffix.lower()
        return {".pt":"pytorch",".pth":"pytorch",".tflite":"tflite",".onnx":"onnx"}.get(ext)

    @classmethod
    def create(cls, model_path: str, tasks: Optional[Dict[str, List[str]]], device="cpu", extra: Optional[Dict[str, Any]]=None):
        cfg = cls._read_config(model_path)
        adapter = cfg.get("adapter")
        mtype = cfg.get("type") or cls._infer_type_from_ext(model_path)
        input_size = int(cfg.get("input_size", 224))
        merged_extra = dict(cfg.get("extra", {}))
        if extra: merged_extra.update(extra)
        preprocess = merged_extra.get("preprocess", "imagenet")

        # PATCHGAN : si détecté → utiliser l'adapter dédié
        if adapter == "patchgan":
            return PatchGANAdapter(model_path, device=device)

        # Sinon, adapter standard
        if mtype not in cls._MAP:
            raise ValueError(f"Type inconnu pour {model_path}. Ajoutez 'type' dans config.json (pytorch|tflite|onnx).")

        if not tasks:
            tasks = cfg.get("tasks") or {"Weather": ["No", "Yes"]}

        return cls._MAP[mtype](
            weights_path=model_path,
            tasks=tasks,
            device=device,
            input_size=input_size,
            extra=merged_extra,
            preprocess=preprocess,
        )


# ─────────────────────────────────────────────────────────────
#                 Thread léger d’inférence (non bloquant)
# ─────────────────────────────────────────────────────────────
class InferWorker(QObject):
    resultReady = Signal(np.ndarray, dict, float)  # frame_bgr, outputs, elapsed_sec

    def __init__(self, infer_adapter: BaseAdapter, prob_threshold: float):
        super().__init__()
        self.infer = infer_adapter
        self.prob_threshold = float(prob_threshold)
        self.queue = queue.Queue(maxsize=1)
        self._running = True

    @Slot()
    def run(self):
        while self._running:
            try:
                frame = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.time()
            outputs = self.infer.predict_bgr(frame)
            elapsed = time.time() - t0
            self.resultReady.emit(frame, outputs, elapsed)

    def submit(self, frame: np.ndarray):
        try:
            if self.queue.full():
                _ = self.queue.get_nowait()
            self.queue.put_nowait(frame)
        except queue.Full:
            pass

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────
#                    Dialogue e-mail (SMTP)
# ─────────────────────────────────────────────────────────────
class EmailDialog(QDialog):
    def __init__(self, parent=None, default_attachments: Optional[List[Path]]=None):
        super().__init__(parent)
        self.setWindowTitle("Envoyer par e-mail")
        self.resize(520, 420)
        self.attachments: List[Path] = list(default_attachments or [])

        self.serverEdit = QLineEdit("smtp.gmail.com")
        self.portEdit   = QLineEdit("465")  # SSL
        self.useTlsChk  = QCheckBox("Utiliser STARTTLS (au lieu de SSL)")
        self.userEdit   = QLineEdit()
        self.passEdit   = QLineEdit(); self.passEdit.setEchoMode(QLineEdit.Password)
        self.fromEdit   = QLineEdit()
        self.toEdit     = QLineEdit()
        self.subjEdit   = QLineEdit("Fichiers de détection")
        self.bodyEdit   = QTextEdit("Veuillez trouver ci-joint les fichiers de détection.")

        self.attLabel   = QLabel(self._att_text())
        addBtn = QPushButton("Ajouter pièces jointes…"); addBtn.clicked.connect(self._add_attachments)
        clearBtn = QPushButton("Vider la liste"); clearBtn.clicked.connect(self._clear_attachments)

        form = QFormLayout()
        form.addRow("Serveur SMTP :", self.serverEdit)
        form.addRow("Port :", self.portEdit)
        form.addRow("", self.useTlsChk)
        form.addRow("Utilisateur :", self.userEdit)
        form.addRow("Mot de passe :", self.passEdit)
        form.addRow("From :", self.fromEdit)
        form.addRow("To (séparés par ,) :", self.toEdit)
        form.addRow("Sujet :", self.subjEdit)
        form.addRow("Message :", self.bodyEdit)

        attRow = QHBoxLayout(); attRow.addWidget(addBtn); attRow.addWidget(clearBtn); attRow.addStretch()
        v = QVBoxLayout(self)
        v.addLayout(form)
        v.addWidget(QLabel("Pièces jointes :"))
        v.addWidget(self.attLabel)
        v.addLayout(attRow)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)

    def _att_text(self) -> str:
        return "(aucune)" if not self.attachments else "\n".join(str(p) for p in self.attachments)

    def _add_attachments(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choisir des fichiers", "", "Tous (*.*)")
        if files:
            for f in files:
                p = Path(f)
                if p.exists():
                    self.attachments.append(p)
        self.attLabel.setText(self._att_text())

    def _clear_attachments(self):
        self.attachments.clear()
        self.attLabel.setText(self._att_text())

    def send_email(self) -> Tuple[bool, str]:
        try:
            server = self.serverEdit.text().strip()
            port   = int(self.portEdit.text().strip() or "465")
            use_tls = self.useTlsChk.isChecked()
            user  = self.userEdit.text().strip()
            pwd   = self.passEdit.text()
            sender = self.fromEdit.text().strip() or user
            to_list = [t.strip() for t in self.toEdit.text().split(",") if t.strip()]
            subject = self.subjEdit.text().strip()
            body    = self.bodyEdit.toPlainText()

            if not (server and port and sender and to_list):
                return False, "Champs requis manquants (serveur/port/from/to)."

            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.mime.base import MIMEBase
            from email import encoders

            msg = MIMEMultipart()
            msg["From"] = sender
            msg["To"] = ", ".join(to_list)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))

            for att in self.attachments:
                part = MIMEBase("application", "octet-stream")
                with open(att, "rb") as f:
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f'attachment; filename="{att.name}"')
                msg.attach(part)

            if use_tls:
                s = smtplib.SMTP(server, port, timeout=30); s.ehlo()
                s.starttls(context=ssl.create_default_context())
            else:
                s = smtplib.SMTP_SSL(server, port, timeout=30); s.ehlo()

            if user and pwd:
                s.login(user, pwd)
            s.sendmail(sender, to_list, msg.as_string())
            s.quit()
            return True, "E-mail envoyé."
        except Exception as e:
            return False, f"Échec envoi : {e}"


# ─────────────────────────────────────────────────────────────
#                    Fenêtre principale (Qt)
# ─────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S.T.I Innovation — Real-time weather detection")
        self.setMinimumSize(1100, 640)

        # État
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_capture)
        self.target_fps = 20

        self.infer_thread = None
        self.infer_worker = None

        self.current_adapter: Optional[BaseAdapter] = None
        self.tasks: Dict[str, List[str]] = {}
        self.device = "cpu"

        # Session / enregistrement
        self.recording_video = False
        self.video_writer = None
        self.output_dir = Path("runs"); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_rows: List[Dict[str, Any]] = []
        self.meta = {}
        self.session_started_at: Optional[datetime.datetime] = None

        # Lecture d’enregistrements
        self.playback_mode = False
        self.play_source_path: Optional[Path] = None
        self.play_fps = 25.0

        # Mesure vitesse (EMA & FPS)
        self.lat_ema_s = None
        self.fps_hist = deque(maxlen=30)

        self._build_ui()
        self._post_show_setup()
        self.reload_models()

    # ---------- UI ----------
    def _build_ui(self):
        self.logo = QLabel(); self.logo.setObjectName("logo")
        self.logo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.logoPath = str(Path(__file__).parent / "images" / "logo_cerema.png")

        self.appTitle = QLabel("S.T.I-WeatherMeasure")
        self.appTitle.setStyleSheet("font-size:20px; font-weight:700;")

        topbar = QHBoxLayout()
        topbar.addWidget(self.logo); topbar.addWidget(self.appTitle); topbar.addStretch()

        self.videoLabel = QLabel("Prévisualisation vidéo")
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoLabel.setStyleSheet("background:#000; color:#aaa;")

        controls = self._build_controls_panel()

        central = QWidget(); self.setCentralWidget(central)
        grid = QGridLayout(central); grid.setContentsMargins(10, 10, 10, 10); grid.setSpacing(10)
        grid.addLayout(topbar, 0, 0, 1, 2)
        grid.addWidget(self.videoLabel, 1, 0, 1, 1)
        grid.addWidget(controls, 1, 1, 1, 1)
        grid.setColumnStretch(0, 3); grid.setColumnStretch(1, 1)

        actOpenOut = QAction("Ouvrir dossier sorties", self)
        actOpenOut.triggered.connect(lambda: os.system(f'xdg-open "{self.output_dir}" 2>/dev/null || open "{self.output_dir}"'))
        self.menuBar().addAction(actOpenOut)

        self._update_logo_pixmap(scale=1.0)

    def _build_controls_panel(self) -> QWidget:
        box = QGroupBox("Contrôles"); lay = QVBoxLayout(box)

        self.modelCombo = QComboBox()
        self.reloadBtn = QPushButton("Recharger modèles"); self.reloadBtn.clicked.connect(self.reload_models)
        self.deviceLabel = QLabel("Device: CPU")

        self.classesEdit = QLineEdit(); self.classesEdit.setPlaceholderText("classes JSON (facultatif)")
        chooseClasses = QPushButton("Parcourir classes…"); chooseClasses.clicked.connect(self._choose_classes)

        self.cameraCombo = QComboBox(); self.cameraCombo.addItems([f"Caméra {i}" for i in range(4)])
        self.fpsSpin = QSpinBox(); self.fpsSpin.setRange(5, 60); self.fpsSpin.setValue(20)
        self.threshSpin = QDoubleSpinBox(); self.threshSpin.setRange(0.0, 1.0); self.threshSpin.setSingleStep(0.05); self.threshSpin.setValue(0.5)

        self.formatCombo = QComboBox(); self.formatCombo.addItems(["json", "csv", "xlsx", "txt"])
        self.sessionEdit = QLineEdit(); self.sessionEdit.setPlaceholderText("Nom de session (défaut: <modèle>_<timestamp>)")
        self.dirEdit = QLineEdit(str(self.output_dir))
        chooseDir = QPushButton("Dossier sortie…"); chooseDir.clicked.connect(self._choose_dir)

        self.speedCheck = QCheckBox("Afficher vitesse (latence/FPS)"); self.speedCheck.setChecked(False)

        self.startBtn = QPushButton("▶ Start (caméra)")
        self.stopBtn  = QPushButton("■ Stop"); self.stopBtn.setEnabled(False)
        self.recBtn   = QPushButton("● Enregistrer vidéo"); self.recBtn.setCheckable(True); self.recBtn.setChecked(False)

        self.startBtn.clicked.connect(self.start_session)
        self.stopBtn.clicked.connect(self.stop_session)
        self.recBtn.toggled.connect(self.toggle_video_record)

        self.playBtn = QPushButton("Lire un enregistrement…"); self.playBtn.clicked.connect(self.open_recording_file)
        self.transferBtn = QPushButton("Transférer fichiers…"); self.transferBtn.clicked.connect(self.transfer_files)
        self.emailBtn = QPushButton("Envoyer par e-mail…"); self.emailBtn.clicked.connect(self.send_email_dialog)

        lay.addWidget(QLabel("Modèle :")); lay.addWidget(self.modelCombo); lay.addWidget(self.reloadBtn); lay.addWidget(self.deviceLabel); lay.addSpacing(10)

        lay.addWidget(QLabel("Fichier classes (JSON) :"))
        r1 = QHBoxLayout(); r1.addWidget(self.classesEdit, 1); r1.addWidget(chooseClasses); lay.addLayout(r1)

        lay.addSpacing(10)
        lay.addWidget(QLabel("Caméra & runtime :"))
        r2 = QGridLayout()
        r2.addWidget(QLabel("Caméra :"), 0, 0); r2.addWidget(self.cameraCombo, 0, 1)
        r2.addWidget(QLabel("FPS cible :"), 1, 0); r2.addWidget(self.fpsSpin, 1, 1)
        r2.addWidget(QLabel("Seuil proba :"), 2, 0); r2.addWidget(self.threshSpin, 2, 1)
        lay.addLayout(r2)
        lay.addWidget(self.speedCheck)

        lay.addSpacing(10)
        lay.addWidget(QLabel("Export résumé :"))
        r3 = QGridLayout()
        r3.addWidget(QLabel("Format :"), 0, 0); r3.addWidget(self.formatCombo, 0, 1)
        r3.addWidget(QLabel("Nom session :"), 1, 0); r3.addWidget(self.sessionEdit, 1, 1)
        r3.addWidget(QLabel("Dossier :"), 2, 0); r3.addWidget(self.dirEdit, 2, 1)
        lay.addLayout(r3)

        lay.addSpacing(10)
        lay.addWidget(self.startBtn); lay.addWidget(self.stopBtn); lay.addWidget(self.recBtn)

        lay.addSpacing(10)
        lay.addWidget(QLabel("Enregistrements :"))
        lay.addWidget(self.playBtn); lay.addWidget(self.transferBtn); lay.addWidget(self.emailBtn)

        lay.addStretch(1)
        return box

    # ---------- Affichage ----------
    def showEvent(self, e):
        super().showEvent(e); QTimer.singleShot(0, self._post_show_setup)

    def _post_show_setup(self):
        w = self.windowHandle()
        if not w:
            screen = QApplication.primaryScreen()
            if screen: self._on_screen_changed(screen)
            return
        try: w.screenChanged.disconnect(self._on_screen_changed)
        except Exception: pass
        w.screenChanged.connect(self._on_screen_changed)
        self._on_screen_changed(w.screen())

    def _on_screen_changed(self, screen):
        dpi = screen.logicalDotsPerInch() or 96.0
        scale = dpi / 96.0
        f = QFont(); f.setPointSizeF(11.0 * scale); self.setFont(f)
        self._update_logo_pixmap(scale)

    def _update_logo_pixmap(self, scale: float):
        base_h = 36; target_h = max(24, int(base_h * max(0.75, scale)))
        if os.path.exists(self.logoPath):
            pm = QPixmap(self.logoPath)
            if not pm.isNull():
                pm = pm.scaledToHeight(target_h, Qt.SmoothTransformation)
                self.logo.setPixmap(pm); self.logo.setMinimumSize(pm.size()); self.logo.setToolTip(self.logoPath); return
        self.logo.setText("LOGO"); self.logo.setStyleSheet("font-weight:600; padding:6px;")

    # ---------- Modèles ----------
    def reload_models(self):
        """Scan récursif de models/** pour *.pt|*.pth|*.tflite|*.onnx (on cache manifest.json)."""
        self.modelCombo.clear()
        models_dir = Path("models").resolve(); models_dir.mkdir(exist_ok=True)
        patterns = ("**/*.pt", "**/*.pth", "**/*.tflite", "**/*.onnx")  # pas de manifest.json ici
        files: List[str] = []
        for pat in patterns:
            files += glob.glob(str(models_dir / pat), recursive=True)
        files = sorted(files)
        if not files:
            self.modelCombo.addItem("(aucun modèle trouvé…)")
            self.modelCombo.setEnabled(False)
        else:
            self.modelCombo.setEnabled(True)
            for f in files:
                fpath = Path(f).resolve()
                try: rel = str(fpath.relative_to(models_dir))
                except Exception: rel = str(fpath)
                self.modelCombo.addItem(rel, str(fpath))

    def _choose_classes(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir classes.json", "", "JSON (*.json)")
        if path: self.classesEdit.setText(path)

    def _choose_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choisir dossier de sortie", str(self.output_dir))
        if path: self.dirEdit.setText(path)

    def _load_tasks(self, model_path: str) -> Dict[str, List[str]]:
        classes_path = self.classesEdit.text().strip()
        if classes_path and Path(classes_path).exists():
            data = json.loads(Path(classes_path).read_text(encoding="utf-8"))
            return {k: list(v) for k, v in data.items()}
        cfg = ModelRegistry._read_config(model_path)
        if "tasks" in cfg and isinstance(cfg["tasks"], dict):
            return {k: list(v) for k, v in cfg["tasks"].items()}
        return {"Weather": ["No", "Yes"]}

    def _build_default_session_name(self, model_path: str) -> str:
        base = Path(model_path).stem; ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{ts}"

    # ---------- Start / Stop ----------
    def start_session(self):
        if self.playback_mode: self._stop_playback()

        model_path = self.modelCombo.currentData() or self.modelCombo.currentText()
        model_path = str(model_path).strip()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "Modèle", "Aucun modèle valide sélectionné."); return

        # Charger tâches (si adapter normal). PatchGAN les obtiendra via son manifest.
        self.tasks = self._load_tasks(model_path)

        self.device = "cuda" if hasattr(torch, "cuda") and getattr(torch.cuda, "is_available", lambda: False)() else "cpu"
        try:
            self.current_adapter = ModelRegistry.create(
                model_path=model_path,
                tasks=self.tasks,
                device=self.device,
                extra=None
            )
            # Si c'est PatchGANAdapter, actualiser tasks
            if isinstance(self.current_adapter, PatchGANAdapter):
                self.tasks = self.current_adapter.tasks
        except Exception as e:
            QMessageBox.critical(self, "Chargement modèle", str(e)); return

        self.deviceLabel.setText(f"Device: {'GPU' if self.device=='cuda' else 'CPU'}")

        cam_index = self.cameraCombo.currentIndex()
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Caméra", f"Impossible d'ouvrir la caméra {cam_index}.")
            self.cap = None; return

        if self.infer_thread: self._stop_infer_thread()
        self.infer_thread = QThread(self)
        self.infer_worker = InferWorker(self.current_adapter, prob_threshold=self.threshSpin.value())
        self.infer_worker.moveToThread(self.infer_thread)
        self.infer_thread.started.connect(self.infer_worker.run)
        self.infer_worker.resultReady.connect(self._on_infer_result)
        self.infer_thread.start()

        outdir = Path(self.dirEdit.text().strip() or self.output_dir); outdir.mkdir(parents=True, exist_ok=True)
        self.output_dir = outdir
        if not self.sessionEdit.text().strip(): self.sessionEdit.setText(self._build_default_session_name(model_path))
        self.summary_rows.clear()
        self.session_started_at = datetime.datetime.now()
        self.meta = {"model_path": model_path, "tasks": list(self.tasks.keys()), "start_time": self.session_started_at.isoformat(timespec="seconds")}
        self.lat_ema_s = None; self.fps_hist.clear()

        self.target_fps = self.fpsSpin.value()
        self.timer.start(int(1000 / max(1, self.target_fps)))
        self.startBtn.setEnabled(False); self.stopBtn.setEnabled(True)

    def stop_session(self):
        self.timer.stop()
        if self.cap: self.cap.release(); self.cap = None
        if self.video_writer: self.video_writer.release(); self.video_writer = None; self.recording_video = False; self.recBtn.setChecked(False)
        self._stop_infer_thread()
        if self.session_started_at is not None:
            self._write_summary_file(); self.session_started_at = None
        self.startBtn.setEnabled(True); self.stopBtn.setEnabled(False)

    def _stop_infer_thread(self):
        if self.infer_worker: self.infer_worker.stop()
        if self.infer_thread:
            self.infer_thread.quit(); self.infer_thread.wait()
        self.infer_worker = None; self.infer_thread = None

    # ---------- Capture & inférence ----------
    def _on_timer_capture(self):
        if not self.cap: return

        if self.playback_mode:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._stop_playback(); return
            disp = draw_bottom_right(frame.copy(), "Lecture enregistrement", alpha=0.6)
            self._set_pixmap(self.videoLabel, disp)
            return

        ok, frame = self.cap.read()
        if not ok or frame is None: return
        if self.infer_worker:
            self.infer_worker.prob_threshold = self.threshSpin.value()
            self.infer_worker.submit(frame)

    @Slot(np.ndarray, dict, float)
    def _on_infer_result(self, frame_bgr: np.ndarray, outputs: dict, elapsed: float):
        # Post-traitement : softmax + libellés + overlay
        lines = []
        per_task = {}
        for task, logits in outputs.items():
            logits_np = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
            vec = logits_np[0] if logits_np.ndim > 1 else logits_np
            probs = softmax_np(vec)
            idx = int(np.argmax(probs))
            score = float(probs[idx])
            labels = self.tasks.get(task, [f"{task}_{i}" for i in range(len(probs))])
            label = labels[idx] if score >= self.threshSpin.value() and idx < len(labels) else "Unknown"
            lines.append(f"{task}: {label} ({score:.2f})")
            per_task[task] = {"label": label, "score": score}

        # Vitesse (EMA + FPS)
        self.lat_ema_s = elapsed if self.lat_ema_s is None else (0.1*elapsed + 0.9*self.lat_ema_s)
        inst_fps = 1.0 / max(elapsed, 1e-6); self.fps_hist.append(inst_fps)
        avg_fps = sum(self.fps_hist)/len(self.fps_hist)

        bottom_text = f"{self.lat_ema_s*1000:.1f} ms  |  {avg_fps:.1f} FPS" if self.speedCheck.isChecked() else None
        disp = draw_overlay(frame_bgr, lines, bottom_right_text=bottom_text)

        # Enregistrement vidéo
        if self.recording_video:
            h, w = disp.shape[:2]
            if self.video_writer is None:
                save_name = self.sessionEdit.text().strip() or self._build_default_session_name(self.meta.get("model_path","session"))
                out_path = self.output_dir / f"{save_name}.avi"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.video_writer = cv2.VideoWriter(str(out_path), fourcc, float(self.target_fps), (w, h))
            if self.video_writer: self.video_writer.write(disp)

        # Résumé
        now = datetime.datetime.now(); ts_ms = int(now.timestamp() * 1000)
        row = {"timestamp_ms": ts_ms, "iso_time": now.isoformat(timespec="milliseconds"),
               "latency_s": round(elapsed, 4), "camera_index": self.cameraCombo.currentIndex(),
               "model": Path(self.meta.get("model_path","")).name}
        for t, info in per_task.items():
            row[f"{t}_label"] = info["label"]; row[f"{t}_score"] = round(float(info["score"]), 4)
        self.summary_rows.append(row)

        self._set_pixmap(self.videoLabel, disp)

    # ---------- Vidéo on/off ----------
    def toggle_video_record(self, checked: bool):
        if self.playback_mode and checked:
            QMessageBox.information(self, "Lecture", "Enregistrement indisponible pendant la lecture d’un fichier.")
            self.recBtn.setChecked(False); return
        self.recording_video = checked
        if not checked and self.video_writer is not None:
            self.video_writer.release(); self.video_writer = None

    # ---------- Lecture d’enregistrements ----------
    def open_recording_file(self):
        if self.cap or self.infer_thread: self.stop_session()
        vid, _ = QFileDialog.getOpenFileName(self, "Ouvrir un enregistrement vidéo", str(self.output_dir),
                                             "Vidéos (*.avi *.mp4 *.mkv *.mov)")
        if not vid: return
        self.play_source_path = Path(vid)
        self.cap = cv2.VideoCapture(str(self.play_source_path))
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Lecture", f"Impossible d’ouvrir {vid}")
            self.cap = None; self.play_source_path = None; return
        fps = self.cap.get(cv2.CAP_PROP_FPS); self.play_fps = fps if fps and fps > 1e-3 else 25.0
        self.playback_mode = True
        self.timer.start(int(1000 / self.play_fps))
        self.startBtn.setEnabled(False); self.stopBtn.setEnabled(True); self.recBtn.setEnabled(False)

    def _stop_playback(self):
        self.playback_mode = False
        if self.cap: self.cap.release(); self.cap = None
        self.timer.stop(); self.play_source_path = None
        self.recBtn.setEnabled(True); self.startBtn.setEnabled(True); self.stopBtn.setEnabled(False)

    # ---------- Transfert ----------
    def transfer_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choisir des fichiers à transférer",
                                                str(self.output_dir), "Tous (*.*)")
        if not files: return
        dest = QFileDialog.getExistingDirectory(self, "Choisir le dossier de destination")
        if not dest: return
        ok_count = 0
        for f in files:
            try: shutil.copy2(f, dest); ok_count += 1
            except Exception as e: print(f"Transfert échoué pour {f}: {e}")
        QMessageBox.information(self, "Transfert", f"{ok_count} fichier(s) copié(s) vers\n{dest}")

    # ---------- E-mail ----------
    def _collect_latest_session_files(self) -> List[Path]:
        base = self.sessionEdit.text().strip()
        if not base: return []
        outs: List[Path] = []
        meta = self.output_dir / f"{base}.meta.json"
        if meta.exists(): outs.append(meta)
        ext = self.formatCombo.currentText()
        path = self.output_dir / f"{base}.{ 'xlsx' if ext=='xlsx' else ext}"
        if not path.exists() and ext == "xlsx":
            alt = self.output_dir / f"{base}.csv"
            if alt.exists(): path = alt
        if path.exists(): outs.append(path)
        vid = self.output_dir / f"{base}.avi"
        if vid.exists(): outs.append(vid)
        return outs

    def send_email_dialog(self):
        defaults = self._collect_latest_session_files()
        dlg = EmailDialog(self, default_attachments=defaults)
        if dlg.exec() == QDialog.Accepted:
            ok, msg = dlg.send_email()
            if ok: QMessageBox.information(self, "E-mail", msg)
            else: QMessageBox.critical(self, "E-mail", msg)

    # ---------- Helpers ----------
    def _set_pixmap(self, label: QLabel, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)

    def closeEvent(self, e):
        try:
            if self.playback_mode: self._stop_playback()
            else: self.stop_session()
        except Exception: pass
        super().closeEvent(e)

    def _write_summary_file(self):
        end_time = datetime.datetime.now()
        self.meta["end_time"] = end_time.isoformat(timespec="seconds")
        start_dt = datetime.datetime.fromisoformat(self.meta["start_time"])
        self.meta["duration_s"] = (end_time - start_dt).total_seconds()

        name = self.sessionEdit.text().strip() or self._build_default_session_name(self.meta.get("model_path","session"))
        fmt  = self.formatCombo.currentText()
        base = self.output_dir / f"{name}"

        meta_path = base.with_suffix(".meta.json")
        meta_obj = { "meta": self.meta, "summary_count": len(self.summary_rows) }
        meta_path.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")

        if fmt == "json":
            out = {"meta": self.meta, "frames": self.summary_rows}
            (base.with_suffix(".json")).write_text(json.dumps(out, indent=2), encoding="utf-8")
        elif fmt == "csv":
            if pd is not None:
                df = pd.DataFrame(self.summary_rows); df.to_csv(base.with_suffix(".csv"), index=False)
            else:
                write_csv_fallback(base.with_suffix(".csv"), self.summary_rows)
        elif fmt == "xlsx":
            if pd is not None:
                df = pd.DataFrame(self.summary_rows)
                try: df.to_excel(base.with_suffix(".xlsx"), index=False)
                except Exception: df.to_csv(base.with_suffix(".csv"), index=False)
            else:
                write_csv_fallback(base.with_suffix(".csv"), self.summary_rows)
        elif fmt == "txt":
            with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write("# META\n")
                for k, v in self.meta.items(): f.write(f"{k}: {v}\n")
                f.write("\n# FRAMES\n")
                if self.summary_rows:
                    keys = list(self.summary_rows[0].keys())
                    f.write("\t".join(keys) + "\n")
                    for r in self.summary_rows:
                        f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")


# ─────────────────────────────────────────────────────────────
#                      Utils overlay & I/O
# ─────────────────────────────────────────────────────────────
def softmax_np(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / (s if s > 0 else 1.0)

def draw_overlay(frame_bgr: np.ndarray, lines: List[str], bottom_right_text: Optional[str]=None) -> np.ndarray:
    disp = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.7, 2
    pad_x, pad_y = 12, 10
    y0, y_step = 30, 28

    longest = max(lines + ["Prévision"], key=len)
    (tw, th), _ = cv2.getTextSize(longest, font, scale, thick)
    box_left, box_top = 0, 0
    box_right = int(tw + 2 * pad_x)
    box_bottom = int(y0 + (len(lines)-1) * y_step + pad_y)

    overlay = disp.copy()
    cv2.rectangle(overlay, (box_left, box_top), (box_right, box_bottom), (255,255,255), -1)
    cv2.addWeighted(overlay, 0.35, disp, 0.65, 0, disp)

    for i, line in enumerate(lines):
        y = y0 + i * y_step
        cv2.putText(disp, line, (pad_x, y), font, scale, (0,255,0), thick, cv2.LINE_AA)

    if bottom_right_text:
        disp = draw_bottom_right(disp, bottom_right_text, alpha=0.6)

    return disp

def draw_bottom_right(img: np.ndarray, text: str, alpha: float=0.6) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 10
    x2, y2 = w - 10, h - 10
    x1, y1 = x2 - (tw + 2*pad), y2 - (th + 2*pad)

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255,255,255), -1)
    cv2.addWeighted(overlay, alpha, out, 1-alpha, 0, out)
    cv2.putText(out, text, (x1 + pad, y2 - pad - 2), font, scale, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(out, text, (x1 + pad, y2 - pad - 2), font, scale, (34,139,34), 1, cv2.LINE_AA)
    return out

def write_csv_fallback(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        Path(path).write_text("", encoding="utf-8"); return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = [str(r.get(k, "")) for k in keys]
            f.write(",".join(vals) + "\n")


# ─────────────────────────────────────────────────────────────
#                           Main
# ─────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    # win.showFullScreen()  # mode kiosque
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
