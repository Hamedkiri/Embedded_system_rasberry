#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI temps réel Raspberry Pi — Détection météo
- PySide6 (Qt) pour l'IHM (responsive, HiDPI, multi-écrans)
- OpenCV pour capture/overlay/vidéo
- ModelRegistry pour charger PyTorch / TFLite / ONNX de façon uniforme

Dépendances conseillées :
  pip install PySide6 opencv-python torch torchvision pandas openpyxl onnxruntime tflite-runtime
  (onnxruntime/tflite-runtime sont optionnels selon vos modèles)
"""

import os
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")  # HiDPI/multi-écrans
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

import sys, json, time, math, glob, threading, queue, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

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

from PySide6.QtCore import Qt, QTimer, QSize, QObject, Signal, Slot, QThread
from PySide6.QtGui  import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QLineEdit,
    QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QMessageBox, QSizePolicy
)

# ─────────────────────────────────────────────────────────────
#                  Pré-traitement ImageNet
# ─────────────────────────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_bgr_imagenet(frame_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    img = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return img  # (H,W,3) float32


# ─────────────────────────────────────────────────────────────
#               Adapters / Registry (chargement modèle)
# ─────────────────────────────────────────────────────────────
class BaseAdapter:
    def __init__(self, tasks: Dict[str, List[str]], device: str = "cpu", input_size: int = 224):
        self.tasks = tasks
        self.device = device
        self.input_size = int(input_size)

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        raise NotImplementedError

class PyTorchMultiTaskAdapter(BaseAdapter):
    """Adapte un modèle PyTorch multi-tâches (dict logits par tâche)."""
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224, extra: Optional[Dict[str, Any]]=None):
        super().__init__(tasks, device, input_size)
        # Import tardif depuis votre projet (MultiHeadAttentionPerTaskModel / load_best_model)
        # Pour un exemple autonome minimal, on va charger un torchscript/pt simple si présent.
        # Ici, on essaie 2 cas:
        #  (A) torch.jit.load -> renvoie un module scripté qui retourne un dict
        #  (B) votre pipeline existant dans __main__ si disponible
        self._jit = None
        self._model = None

        try:
            self._jit = torch.jit.load(weights_path, map_location=device)
            self._jit.eval()
        except Exception:
            # Fallback: essayer d'utiliser vos classes si importables depuis un module local
            try:
                from train_style_disentangle import MultiHeadAttentionPerTaskModel, load_best_model  # ajustez si besoin
                from torchvision import models
                base_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                truncate = int((extra or {}).get("truncate_after_layer", 10))
                use_attention = bool((extra or {}).get("use_attention", True))
                attn_token_dim = (extra or {}).get("attn_token_dim", None)
                cls_hidden_dims = (extra or {}).get("cls_hidden_dims", [])
                cls_num_layers  = int((extra or {}).get("cls_num_layers", 0))
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
                load_best_model(self._model, weights_path, strict_backbone=True)
                self._model.eval()
            except Exception as e:
                raise RuntimeError(f"Impossible de charger le modèle PyTorch: {e}")

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = preprocess_bgr_imagenet(frame_bgr, self.input_size)
        nchw = np.transpose(nhwc, (2,0,1))[np.newaxis, ...].astype(np.float32)
        inp  = torch.from_numpy(nchw)
        with torch.no_grad():
            if self._jit is not None:
                out = self._jit(inp)
                # Assumer que out est dict{task: logits[1,n]}
                return out
            else:
                out = self._model(inp)
                return out

class TFLiteMultiHeadAdapter(BaseAdapter):
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224, extra: Optional[Dict[str, Any]]=None):
        super().__init__(tasks, device, input_size)
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
        nhwc = preprocess_bgr_imagenet(frame_bgr, self.input_size)
        arr = nhwc[np.newaxis, ...].astype(np.float32)  # [1,H,W,3]
        inp = self.input_details[0]
        if inp["dtype"] == np.uint8:
            arr = (np.clip((nhwc * _IMAGENET_STD + _IMAGENET_MEAN), 0, 1) * 255.0).astype(np.uint8)[np.newaxis, ...]
        self.interp.set_tensor(inp["index"], arr)
        self.interp.invoke()
        outs = [self.interp.get_tensor(od["index"]) for od in self.output_details]

        result = {}
        if len(outs) == len(self.output_task_order):
            for task_name, arr in zip(self.output_task_order, outs):
                result[task_name] = torch.from_numpy(arr)
        else:
            total = sum(len(v) for v in self.tasks.values())
            flat = outs[0]
            if flat.shape[-1] != total:
                raise RuntimeError("TFLite: sortie incompatible. Configurez 'output_task_order' dans config.json.")
            off = 0
            for task, cls in self.tasks.items():
                n = len(cls)
                result[task] = torch.from_numpy(flat[:, off:off+n])
                off += n
        return result

class ONNXMultiHeadAdapter(BaseAdapter):
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224, extra: Optional[Dict[str, Any]]=None):
        super().__init__(tasks, device, input_size)
        extra = extra or {}
        import onnxruntime as ort
        providers = extra.get("providers", ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(weights_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.output_task_order = extra.get("output_task_order", list(tasks.keys()))

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = preprocess_bgr_imagenet(frame_bgr, self.input_size)
        nchw = nhwc.transpose(2,0,1)[np.newaxis, ...].astype(np.float32)
        outs = self.sess.run(self.output_names, {self.input_name: nchw})
        result = {}
        if len(outs) == len(self.output_task_order):
            for task_name, arr in zip(self.output_task_order, outs):
                result[task_name] = torch.from_numpy(arr)
        else:
            total = sum(len(v) for v in self.tasks.values())
            flat = outs[0]
            if flat.shape[-1] != total:
                raise RuntimeError("ONNX: sortie incompatible. Configurez 'output_task_order' dans config.json.")
            off = 0
            for task, cls in self.tasks.items():
                n = len(cls)
                result[task] = torch.from_numpy(flat[:, off:off+n])
                off += n
        return result

class ModelRegistry:
    _MAP = { "pytorch": PyTorchMultiTaskAdapter, "tflite": TFLiteMultiHeadAdapter, "onnx": ONNXMultiHeadAdapter }

    @staticmethod
    def _read_config(weights_path: str) -> Dict[str, Any]:
        base = Path(weights_path).parent
        cfg = base / "config.json"
        if cfg.exists():
            return json.loads(cfg.read_text(encoding="utf-8"))
        return {}

    @staticmethod
    def _infer_type_from_ext(weights_path: str) -> Optional[str]:
        ext = Path(weights_path).suffix.lower()
        return {".pt":"pytorch",".pth":"pytorch",".tflite":"tflite",".onnx":"onnx"}.get(ext)

    @classmethod
    def create(cls, weights_path: str, tasks: Dict[str, List[str]], device="cpu", extra: Optional[Dict[str, Any]]=None):
        cfg = cls._read_config(weights_path)
        mtype = cfg.get("type") or cls._infer_type_from_ext(weights_path)
        if mtype not in cls._MAP:
            raise ValueError(f"Type inconnu pour {weights_path}. Ajoutez 'type' dans config.json (pytorch|tflite|onnx).")
        input_size = int(cfg.get("input_size", 224))
        merged_extra = dict(cfg.get("extra", {}))
        if extra:
            merged_extra.update(extra)
        return cls._MAP[mtype](
            weights_path=weights_path,
            tasks=tasks,
            device=device,
            input_size=input_size,
            extra=merged_extra
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
        self.queue = queue.Queue(maxsize=1)  # drop frames si saturé
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
        # Tente d'insérer sans bloquer (drop si plein)
        try:
            if self.queue.full():
                _ = self.queue.get_nowait()
            self.queue.put_nowait(frame)
        except queue.Full:
            pass

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────
#                    Fenêtre principale (Qt)
# ─────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S.T.I Innovation — Real-time weather detection")
        self.setMinimumSize(1000, 600)

        # État
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_capture)
        self.target_fps = 20
        self.last_tick = 0.0

        self.infer_thread = None
        self.infer_worker = None
        self.infer_busy = False

        self.current_adapter: Optional[BaseAdapter] = None
        self.tasks: Dict[str, List[str]] = {}
        self.device = "cpu"

        # Session / enregistrement
        self.recording_video = False
        self.video_writer = None
        self.overlay_on_video = True
        self.output_dir = Path("runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_rows: List[Dict[str, Any]] = []
        self.meta = {}
        self.session_started_at: Optional[datetime.datetime] = None

        self._build_ui()
        self._post_show_setup()
        self.reload_models()

    # ---------- UI building ----------
    from pathlib import Path
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QSizePolicy

    def _build_ui(self):
        # Top bar (logo + titre)
        self.logo = QLabel()
        self.logo.setObjectName("logo")
        self.logo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.logoPath = str(Path(__file__).parent / "images" / "logo_cerema.png")

        self.appTitle = QLabel("S.T.I-WeatherMeasure")
        self.appTitle.setStyleSheet("font-size:20px; font-weight:700;")

        topbar = QHBoxLayout()
        topbar.addWidget(self.logo)
        topbar.addWidget(self.appTitle)
        topbar.addStretch()

        # Zone vidéo
        self.videoLabel = QLabel("Prévisualisation vidéo")
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoLabel.setStyleSheet("background:#000; color:#aaa;")

        # Panneau contrôles
        controls = self._build_controls_panel()

        # Layout principal
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(10)
        grid.addLayout(topbar, 0, 0, 1, 2)
        grid.addWidget(self.videoLabel, 1, 0, 1, 1)
        grid.addWidget(controls, 1, 1, 1, 1)

        # stretch
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 1)

        # Initialise le logo une première fois
        self._update_logo_pixmap(scale=1.0)

    def _build_controls_panel(self) -> QWidget:
        box = QGroupBox("Contrôles")
        lay = QVBoxLayout(box)

        # Modèle
        self.modelCombo = QComboBox()
        self.reloadBtn = QPushButton("Recharger modèles")
        self.reloadBtn.clicked.connect(self.reload_models)
        self.deviceLabel = QLabel("Device: CPU")

        # Fichier classes (si non encodé dans le modèle/config)
        self.classesEdit = QLineEdit()
        self.classesEdit.setPlaceholderText("build_classifier.json (facultatif)")
        chooseClasses = QPushButton("Parcourir classes…")
        chooseClasses.clicked.connect(self._choose_classes)

        # Caméra
        self.cameraCombo = QComboBox()
        self.cameraCombo.addItems([f"Caméra {i}" for i in range(4)])
        self.fpsSpin = QSpinBox(); self.fpsSpin.setRange(5, 60); self.fpsSpin.setValue(20)
        self.threshSpin = QDoubleSpinBox(); self.threshSpin.setRange(0.0, 1.0); self.threshSpin.setSingleStep(0.05); self.threshSpin.setValue(0.5)

        # Sorties
        self.formatCombo = QComboBox(); self.formatCombo.addItems(["json", "csv", "xlsx", "txt"])
        self.sessionEdit = QLineEdit()
        self.sessionEdit.setPlaceholderText("Nom de session (défaut: <modèle>_<timestamp>)")
        self.dirEdit = QLineEdit(str(self.output_dir))
        chooseDir = QPushButton("Dossier sortie…")
        chooseDir.clicked.connect(self._choose_dir)

        # Boutons
        self.startBtn = QPushButton("▶ Start")
        self.stopBtn  = QPushButton("■ Stop")
        self.stopBtn.setEnabled(False)
        self.recBtn   = QPushButton("● Enregistrer vidéo")
        self.recBtn.setCheckable(True)
        self.recBtn.setChecked(False)

        self.startBtn.clicked.connect(self.start_session)
        self.stopBtn.clicked.connect(self.stop_session)
        self.recBtn.toggled.connect(self.toggle_video_record)

        # Layouts
        lay.addWidget(QLabel("Modèle :"))
        lay.addWidget(self.modelCombo)
        lay.addWidget(self.reloadBtn)
        lay.addWidget(self.deviceLabel)
        lay.addSpacing(10)

        lay.addWidget(QLabel("Fichier classes (JSON) :"))
        row1 = QHBoxLayout()
        row1.addWidget(self.classesEdit, 1)
        row1.addWidget(chooseClasses)
        lay.addLayout(row1)

        lay.addSpacing(10)
        lay.addWidget(QLabel("Caméra & runtime :"))
        r2 = QGridLayout()
        r2.addWidget(QLabel("Caméra :"), 0, 0); r2.addWidget(self.cameraCombo, 0, 1)
        r2.addWidget(QLabel("FPS cible :"), 1, 0); r2.addWidget(self.fpsSpin, 1, 1)
        r2.addWidget(QLabel("Seuil proba :"), 2, 0); r2.addWidget(self.threshSpin, 2, 1)
        lay.addLayout(r2)

        lay.addSpacing(10)
        lay.addWidget(QLabel("Export résumé :"))
        r3 = QGridLayout()
        r3.addWidget(QLabel("Format :"), 0,0); r3.addWidget(self.formatCombo, 0,1)
        r3.addWidget(QLabel("Nom session :"), 1,0); r3.addWidget(self.sessionEdit, 1,1)
        r3.addWidget(QLabel("Dossier :"), 2,0); r3.addWidget(self.dirEdit, 2,1)
        r3.addWidget(QWidget(), 3,0)  # spacer
        lay.addLayout(r3)

        lay.addSpacing(10)
        lay.addWidget(self.startBtn)
        lay.addWidget(self.stopBtn)
        lay.addWidget(self.recBtn)
        lay.addStretch(1)
        return box

    def showEvent(self, e):
        """Appelé quand la fenêtre devient visible → le QWindow existe."""
        super().showEvent(e)
        # Attendre 0 ms garantit que le handle est prêt sur Wayland/X11
        QTimer.singleShot(0, self._post_show_setup)

    def _post_show_setup(self):
        w = self.windowHandle()
        if not w:
            # Fallback si aucun handle (offscreen/eglfs ou timing rare)
            screen = QApplication.primaryScreen()
            if screen:
                self._on_screen_changed(screen)
            return
        # Se reconnecter qu’une seule fois si vous rappelez _post_show_setup
        try:
            w.screenChanged.disconnect(self._on_screen_changed)
        except Exception:
            pass
        w.screenChanged.connect(self._on_screen_changed)
        self._on_screen_changed(w.screen())

    def _on_screen_changed(self, screen):
        dpi = screen.logicalDotsPerInch() or 96.0
        scale = dpi / 96.0
        f = QFont();
        f.setPointSizeF(11.0 * scale)
        self.setFont(f)
        # Recalibre le logo quand on déplace la fenêtre sur un autre écran
        self._update_logo_pixmap(scale)

    def _update_logo_pixmap(self, scale: float):
        """Charge/redimensionne le logo selon l’échelle DPI."""
        base_h = 36  # hauteur de base (en points) ; ajuste si tu veux plus grand
        target_h = max(24, int(base_h * max(0.75, scale)))
        if os.path.exists(self.logoPath):
            pm = QPixmap(self.logoPath)
            if not pm.isNull():
                pm = pm.scaledToHeight(target_h, Qt.SmoothTransformation)
                self.logo.setPixmap(pm)
                self.logo.setMinimumSize(pm.size())
                self.logo.setToolTip(self.logoPath)
                return
        # fallback texte si l’image est introuvable
        self.logo.setText("LOGO")
        self.logo.setStyleSheet("font-weight:600; padding:6px;")

    # ---------- Gestion des modèles ----------
    def reload_models(self):
        self.modelCombo.clear()
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        files = []
        for ext in ("*.pt", "*.pth", "*.tflite", "*.onnx"):
            files += glob.glob(str(models_dir / ext))
        files = sorted(files)
        if not files:
            self.modelCombo.addItem("(aucun modèle trouvé…)")
            self.modelCombo.setEnabled(False)
        else:
            self.modelCombo.setEnabled(True)
            for f in files:
                self.modelCombo.addItem(f)

    def _choose_classes(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir build_classifier.json", "", "JSON (*.json)")
        if path:
            self.classesEdit.setText(path)

    def _choose_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choisir dossier de sortie", str(self.output_dir))
        if path:
            self.dirEdit.setText(path)

    def _load_tasks(self, model_path: str) -> Dict[str, List[str]]:
        # 1) si un JSON de classes a été choisi
        classes_path = self.classesEdit.text().strip()
        if classes_path and Path(classes_path).exists():
            data = json.loads(Path(classes_path).read_text(encoding="utf-8"))
            return {k: list(v) for k, v in data.items()}

        # 2) sinon tenter models/config.json -> "tasks"
        cfg = ModelRegistry._read_config(model_path)
        if "tasks" in cfg:
            return {k: list(v) for k, v in cfg["tasks"].items()}

        # 3) fallback: 1 seule tâche binaire (Unknown/Detected)
        return {"Weather": ["No", "Yes"]}

    def _build_default_session_name(self, model_path: str) -> str:
        base = Path(model_path).stem
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{ts}"

    # ---------- Start / Stop ----------
    def start_session(self):
        model_path = self.modelCombo.currentText().strip()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "Modèle", "Aucun modèle valide sélectionné.")
            return

        # Charger les tâches
        self.tasks = self._load_tasks(model_path)

        # Créer l'adapter (redirige auto selon type)
        self.device = "cuda" if hasattr(torch, "cuda") and getattr(torch.cuda, "is_available", lambda: False)() else "cpu"
        try:
            self.current_adapter = ModelRegistry.create(
                weights_path=model_path,
                tasks=self.tasks,
                device=self.device,
                extra=None
            )
        except Exception as e:
            QMessageBox.critical(self, "Chargement modèle", str(e))
            return

        self.deviceLabel.setText(f"Device: {'GPU' if self.device=='cuda' else 'CPU'}")

        # Ouvrir caméra
        cam_index = self.cameraCombo.currentIndex()
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Caméra", f"Impossible d'ouvrir la caméra {cam_index}.")
            self.cap = None
            return

        # Démarrer worker d'inférence
        if self.infer_thread:
            self._stop_infer_thread()
        self.infer_thread = QThread(self)
        self.infer_worker = InferWorker(self.current_adapter, prob_threshold=self.threshSpin.value())
        self.infer_worker.moveToThread(self.infer_thread)
        self.infer_thread.started.connect(self.infer_worker.run)
        self.infer_worker.resultReady.connect(self._on_infer_result)
        self.infer_thread.start()

        # Init session
        outdir = Path(self.dirEdit.text().strip() or self.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        self.output_dir = outdir
        if not self.sessionEdit.text().strip():
            self.sessionEdit.setText(self._build_default_session_name(model_path))
        self.summary_rows.clear()
        self.session_started_at = datetime.datetime.now()
        self.meta = {
            "model_path": model_path,
            "tasks": list(self.tasks.keys()),
            "start_time": self.session_started_at.isoformat(timespec="seconds")
        }

        # Timer capture
        self.target_fps = self.fpsSpin.value()
        self.last_tick = 0.0
        self.timer.start(int(1000 / max(1, self.target_fps)))

        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)

    def stop_session(self):
        # Arrêt capture
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

        # Arrêt vidéo
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording_video = False
            self.recBtn.setChecked(False)

        # Arrêt worker
        self._stop_infer_thread()

        # Écrire le fichier résumé
        if self.session_started_at is not None:
            self._write_summary_file()
            self.session_started_at = None

        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)

    def _stop_infer_thread(self):
        if self.infer_worker:
            self.infer_worker.stop()
        if self.infer_thread:
            self.infer_thread.quit()
            self.infer_thread.wait()
        self.infer_worker = None
        self.infer_thread = None

    # ---------- Capture & inférence ----------
    def _on_timer_capture(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        # Soumettre à l'inférence (non bloquant)
        if self.infer_worker:
            self.infer_worker.prob_threshold = self.threshSpin.value()
            self.infer_worker.submit(frame)

    @Slot(np.ndarray, dict, float)
    def _on_infer_result(self, frame_bgr: np.ndarray, outputs: dict, elapsed: float):
        # Post-traitement : softmax + libellés + overlay
        lines = []
        per_task = {}
        for task, logits in outputs.items():
            # logits : [1, n]
            logits_np = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
            probs = softmax_np(logits_np[0])
            idx = int(np.argmax(probs))
            score = float(probs[idx])
            label = self.tasks.get(task, ["Unknown"]* (idx+1))[idx] if score >= self.threshSpin.value() else "Unknown"
            lines.append(f"{task}: {label} ({score:.2f})")
            per_task[task] = {"label": label, "score": score}

        # Overlay (texte + bandeau translucide)
        disp = draw_overlay(frame_bgr, lines)

        # Vidéo
        if self.recording_video:
            h, w = disp.shape[:2]
            if self.video_writer is None:
                save_name = self.sessionEdit.text().strip() or self._build_default_session_name(self.meta.get("model_path","session"))
                out_path = self.output_dir / f"{save_name}.avi"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.video_writer = cv2.VideoWriter(str(out_path), fourcc, float(self.target_fps), (w, h))
            if self.video_writer:
                self.video_writer.write(disp)

        # Résumé par frame
        now = datetime.datetime.now()
        ts_ms = int(now.timestamp() * 1000)
        row = {
            "timestamp_ms": ts_ms,
            "iso_time": now.isoformat(timespec="milliseconds"),
            "latency_s": round(elapsed, 4),
            "camera_index": self.cameraCombo.currentIndex(),
            "model": Path(self.meta.get("model_path","")).name,
        }
        for task, info in per_task.items():
            row[f"{task}_label"] = info["label"]
            row[f"{task}_score"] = round(float(info["score"]), 4)
        self.summary_rows.append(row)

        # Affichage
        self._set_pixmap(self.videoLabel, disp)

    # ---------- Vidéo on/off ----------
    def toggle_video_record(self, checked: bool):
        self.recording_video = checked
        if not checked and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    # ---------- Helpers ----------
    def _set_pixmap(self, label: QLabel, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)

    def closeEvent(self, e):
        try:
            self.stop_session()
        except Exception:
            pass
        super().closeEvent(e)

    def _write_summary_file(self):
        end_time = datetime.datetime.now()
        self.meta["end_time"] = end_time.isoformat(timespec="seconds")
        start_dt = datetime.datetime.fromisoformat(self.meta["start_time"])
        self.meta["duration_s"] = (end_time - start_dt).total_seconds()

        name = self.sessionEdit.text().strip() or self._build_default_session_name(self.meta.get("model_path","session"))
        fmt  = self.formatCombo.currentText()
        base = self.output_dir / f"{name}"

        # Toujours produire un JSON méta minimal
        meta_path = base.with_suffix(".meta.json")
        meta_obj = {
            "meta": self.meta,
            "summary_count": len(self.summary_rows),
        }
        meta_path.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")

        if fmt == "json":
            out = {"meta": self.meta, "frames": self.summary_rows}
            (base.with_suffix(".json")).write_text(json.dumps(out, indent=2), encoding="utf-8")
        elif fmt == "csv":
            if pd is not None:
                df = pd.DataFrame(self.summary_rows)
                df.to_csv(base.with_suffix(".csv"), index=False)
            else:
                # fallback manuel CSV
                write_csv_fallback(base.with_suffix(".csv"), self.summary_rows)
        elif fmt == "xlsx":
            if pd is not None:
                df = pd.DataFrame(self.summary_rows)
                try:
                    df.to_excel(base.with_suffix(".xlsx"), index=False)
                except Exception:
                    # si openpyxl absent → CSV en secours
                    df.to_csv(base.with_suffix(".csv"), index=False)
            else:
                # fallback → CSV
                write_csv_fallback(base.with_suffix(".csv"), self.summary_rows)
        elif fmt == "txt":
            with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write("# META\n")
                for k, v in self.meta.items():
                    f.write(f"{k}: {v}\n")
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
    return e / np.sum(e)

def draw_overlay(frame_bgr: np.ndarray, lines: List[str]) -> np.ndarray:
    disp = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.7, 2
    pad_x, pad_y = 12, 10
    y0, y_step = 30, 28

    # Taille bandeau
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
    return disp

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
    # Plein écran “kiosk” possible selon votre déploiement
    # win.showFullScreen()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
