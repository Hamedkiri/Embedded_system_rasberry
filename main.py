#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI temps réel Raspberry Pi — Détection météo (sélection/prune des tâches + choix caméra)
"""

import os
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

import sys, json, time, glob, queue, datetime, shutil, ssl, smtplib, math
import subprocess, threading, shutil as _shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

import numpy as np
import cv2

# Limiter le nombre de threads OpenCV
try:
    cv2.setNumThreads(int(os.getenv("OPENCV_THREADS", "1")))
except Exception:
    pass

def _ts(): return datetime.datetime.now().strftime("[%H:%M:%S.%f]")[:-3]
def log(msg): print(_ts(), msg, flush=True)

log("==== START ====")
log(f"Python: {sys.version.split()[0]} | OS: {sys.platform}")
log(f"OpenCV: {cv2.__version__}")
try:
    import PySide6
    log(f"PySide6: {PySide6.__version__}")
except Exception:
    pass
log(f"DISPLAY: {os.getenv('DISPLAY')} WAYLAND_DISPLAY: {os.getenv('WAYLAND_DISPLAY')} XDG_SESSION_TYPE: {os.getenv('XDG_SESSION_TYPE')}")
videos = sorted(glob.glob("/dev/video*"))
if videos: log(f"Available /dev/video*: {videos}")

# SMTP config (Gmail)
DEFAULT_SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
DEFAULT_SMTP_PORT   = int(os.getenv("SMTP_PORT", "465"))
DEFAULT_EMAIL_USER  = os.getenv("SMTP_USER",  "votre@gmail.com")
DEFAULT_EMAIL_PASS  = os.getenv("SMTP_PASSWORD", "mot_de_passe_application")
DEFAULT_EMAIL_FROM  = os.getenv("SMTP_FROM", DEFAULT_EMAIL_USER)

# Optionnels
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import torch
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(int(os.getenv("TORCH_THREADS", "1")))
    # --- Compat PyTorch: ModuleDict.get (pour ResNet50_truncated_module) ---
    import torch.nn as nn
    if not hasattr(nn.ModuleDict, "get"):
        def _md_get(self, key, default=None):
            return self[key] if key in self else default
        nn.ModuleDict.get = _md_get  # monkey-patch pour anciennes versions
except Exception:
    class _DummyTorch:
        def no_grad(self): return self
        def __enter__(self): return self
        def __exit__(self, *a): pass
        device = "cpu"
        from numpy import ndarray
    torch = _DummyTorch()

from PySide6.QtCore import Qt, QTimer, QObject, Signal, Slot, QThread, QSize, QRect, QPropertyAnimation, QEasingCurve
from PySide6.QtGui  import QImage, QPixmap, QFont, QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QLineEdit,
    QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QMessageBox, QSizePolicy, QDialog, QFormLayout,
    QDialogButtonBox, QTextEdit, QScrollArea, QFrame, QLayout, QToolButton,
    QGraphicsDropShadowEffect, QSpacerItem
)

# ───────── Pré-traitements images ─────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_bgr_imagenet(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return img

def preprocess_bgr_tanh(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 127.5) - 1.0
    return img

# ───────── Adapters / Registry (modèles) ─────────
class BaseAdapter:
    def __init__(self, tasks: Dict[str, List[str]], device: str = "cpu", input_size: int = 224, preprocess: str = "imagenet"):
        self.tasks = tasks; self.device = device; self.input_size = int(input_size); self.preprocess = preprocess
    def _prep(self, frame_bgr: np.ndarray) -> np.ndarray:
        return preprocess_bgr_tanh(frame_bgr, self.input_size) if self.preprocess == "tanh" else preprocess_bgr_imagenet(frame_bgr, self.input_size)
    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        raise NotImplementedError

class PyTorchMultiTaskAdapter(BaseAdapter):
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224, extra: Optional[Dict[str, Any]]=None, preprocess: str="imagenet"):
        super().__init__(tasks, device, input_size, preprocess)
        self._jit = None; self._model = None
        try:
            self._jit = torch.jit.load(weights_path, map_location=device); self._jit.eval()
            log("[Model] Loaded TorchScript")
        except Exception:
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
                    base_encoder=base_encoder, truncate_after_layer=truncate,
                    tasks=tasks, device=device, use_attention=use_attention,
                    attn_token_dim=attn_token_dim, cls_hidden_dims=cls_hidden_dims, cls_num_layers=cls_num_layers
                )
                load_best_model(self._model, weights_path, strict_backbone=False, verbose=False)
                self._model.eval()
                log("[Model] Loaded PyTorch state dict")
            except Exception as e:
                raise RuntimeError(f"Impossible de charger le modèle PyTorch: {e}")
    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = self._prep(frame_bgr); nchw = np.transpose(nhwc, (2,0,1))[np.newaxis, ...].astype(np.float32)
        inp  = torch.from_numpy(nchw)
        with torch.no_grad():
            return self._jit(inp) if self._jit is not None else self._model(inp)

class TFLiteMultiHeadAdapter(BaseAdapter):
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224, extra: Optional[Dict[str, Any]]=None, preprocess: str="imagenet"):
        super().__init__(tasks, device, input_size, preprocess); extra = extra or {}
        try:
            import tflite_runtime.interpreter as tflite
        except Exception:
            import tensorflow.lite as tflite
        self.interp = tflite.Interpreter(model_path=weights_path, num_threads=int(extra.get("threads", 2)))
        self.interp.allocate_tensors()
        self.input_details  = self.interp.get_input_details()
        self.output_details = self.interp.get_output_details()
        self.output_task_order = extra.get("output_task_order", list(tasks.keys()))
        log("[Model] Loaded TFLite")
    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = self._prep(frame_bgr); arr = nhwc[np.newaxis, ...].astype(np.float32)
        inp = self.input_details[0]
        if inp["dtype"] == np.uint8:
            if self.preprocess == "imagenet": rgb01 = np.clip((nhwc * _IMAGENET_STD + _IMAGENET_MEAN), 0, 1)
            else:                             rgb01 = np.clip((nhwc + 1.0) / 2.0, 0, 1)
            arr = (rgb01 * 255.0).astype(np.uint8)[np.newaxis, ...]
        self.interp.set_tensor(inp["index"], arr); self.interp.invoke()
        outs = [self.interp.get_tensor(od["index"]) for od in self.output_details]
        result = {}
        if len(outs) == len(self.output_task_order):
            for task_name, a in zip(self.output_task_order, outs): result[task_name] = torch.from_numpy(a)
        else:
            total = sum(len(v) for v in self.tasks.values()); flat = outs[0]
            if flat.shape[-1] != total: raise RuntimeError("TFLite: sortie incompatible. Configurez 'output_task_order' dans config.")
            off = 0
            for task, cls in self.tasks.items():
                n = len(cls); result[task] = torch.from_numpy(flat[:, off:off+n]); off += n
        return result

class ONNXMultiHeadAdapter(BaseAdapter):
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224, extra: Optional[Dict[str, Any]]=None, preprocess: str="imagenet"):
        super().__init__(tasks, device, input_size, preprocess); extra = extra or {}
        import onnxruntime as ort
        providers = extra.get("providers", ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(weights_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.output_task_order = extra.get("output_task_order", list(tasks.keys()))
        log("[Model] Loaded ONNX")
    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = self._prep(frame_bgr); nchw = nhwc.transpose(2,0,1)[np.newaxis, ...].astype(np.float32)
        outs = self.sess.run(self.output_names, {self.input_name: nchw})
        result = {}
        if len(outs) == len(self.output_task_order):
            for task_name, a in zip(self.output_task_order, outs): result[task_name] = torch.from_numpy(a)
        else:
            total = sum(len(v) for v in self.tasks.values()); flat = outs[0]
            if flat.shape[-1] != total: raise RuntimeError("ONNX: sortie incompatible. Configurez 'output_task_order' dans config.")
            off = 0
            for task, cls in self.tasks.items():
                n = len(cls); result[task] = torch.from_numpy(flat[:, off:off+n]); off += n
        return result

class PatchGANAdapter(BaseAdapter):
    def __init__(self, model_path: str, device="cpu", selected_tasks: Optional[List[str]]=None):
        p = Path(model_path); manifest = p.parent / "manifest.json"
        if not manifest.exists(): raise RuntimeError(f"manifest.json introuvable à côté de {model_path}")
        try:
            from loader_patchgan_from_manifest import create_patchgan_from_manifest
        except Exception as e:
            raise RuntimeError(f"Impossible d'importer loader_patchgan_from_manifest.py : {e}")
        # → passer selected_tasks au constructeur PatchGAN pour PRUNE
        model, tasks = create_patchgan_from_manifest(str(manifest), device=device, selected_tasks=selected_tasks)
        man = json.loads(manifest.read_text(encoding="utf-8"))
        extra = man.get("extra", {}); input_size = int(man.get("input_size", extra.get("input_size", 224)))
        preprocess = str(extra.get("preprocess", "tanh")).lower()
        super().__init__(tasks=tasks, device=device, input_size=input_size, preprocess=preprocess)
        self.model = model.eval(); log("[Model] Loaded PatchGAN")
    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = self._prep(frame_bgr); nchw = np.transpose(nhwc, (2,0,1))[np.newaxis, ...].astype(np.float32)
        x = torch.from_numpy(nchw)
        with torch.no_grad(): out = self.model(x)
        return out

class ModelRegistry:
    _MAP = {"pytorch":PyTorchMultiTaskAdapter,"tflite":TFLiteMultiHeadAdapter,"onnx":ONNXMultiHeadAdapter}
    @staticmethod
    def _labels_from_tasks(tasks_dict: Optional[Dict[str, int]]) -> Dict[str, List[str]]:
        if not isinstance(tasks_dict, dict): return {"Weather": ["No", "Yes"]}
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
        for k in ("patch_size","attn_tau","attn_use_se","ndf","discriminator"):
            if k in man or k in man.get("extra", {}): return True
        return False
    @staticmethod
    def _read_config(model_path: str) -> Dict[str, Any]:
        p = Path(model_path); base = p.parent
        cfg_path = base / "config.json"
        if cfg_path.exists():
            try: return json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception: pass
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
                if man.get("adapter", "").lower() == "patchgan" or ModelRegistry._looks_like_patchgan(man, str(p)): cfg["adapter"] = "patchgan"
                if cfg["tasks"] and isinstance(list(cfg["tasks"].values())[0], int):
                    cfg["tasks"] = ModelRegistry._labels_from_tasks(cfg["tasks"])
                return cfg
            except Exception: pass
        return {}
    @staticmethod
    def _infer_type_from_ext(model_path: str) -> Optional[str]:
        ext = Path(model_path).suffix.lower()
        return {".pt":"pytorch",".pth":"pytorch",".tflite":"tflite",".onnx":"onnx"}.get(ext)
    @classmethod
    def create(cls, model_path: str, tasks: Optional[Dict[str, List[str]]], device="cpu", extra: Optional[Dict[str, Any]]=None,
               selected_tasks: Optional[List[str]]=None):
        cfg = cls._read_config(model_path); adapter = cfg.get("adapter")
        mtype = cfg.get("type") or cls._infer_type_from_ext(model_path)
        input_size = int(cfg.get("input_size", 224))
        merged_extra = dict(cfg.get("extra", {}));
        if extra: merged_extra.update(extra)
        preprocess = merged_extra.get("preprocess", "imagenet")

        # PatchGAN: déléguer au chargeur (PRUNE dans le loader)
        if adapter == "patchgan":
            return PatchGANAdapter(model_path, device=device, selected_tasks=selected_tasks)

        # Autres: construire avec seulement les tâches sélectionnées (PRUNE par construction)
        if not tasks:
            tasks = cfg.get("tasks") or {"Weather": ["No", "Yes"]}
        if selected_tasks:
            tasks = {k: v for k, v in tasks.items() if k in selected_tasks}

        if mtype not in cls._MAP: raise ValueError(f"Type inconnu pour {model_path}. Ajoutez 'type' dans config.json (pytorch|tflite|onnx).")
        return cls._MAP[mtype](weights_path=model_path,tasks=tasks,device=device,input_size=input_size,extra=merged_extra,preprocess=preprocess)

# ───────── Backend rpicam-vid/libcamera-vid (MJPEG) ─────────
class LibcameraMJPEGCapture:
    def __init__(self, width=1280, height=720, fps=30):
        self.width = int(width); self.height = int(height); self.fps = int(fps)
        self.proc: Optional[subprocess.Popen] = None; self._buf = bytearray()
        self._running = False; self._th: Optional[threading.Thread] = None
        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self.bin = None
    def _pick_bin(self) -> Optional[str]:
        env = os.getenv("PI_CAM_BIN"); cand = [env] if env else ["rpicam-vid", "libcamera-vid"]
        for b in cand:
            if b and _shutil.which(b): return b
        return None
    def start(self) -> bool:
        self.bin = self._pick_bin()
        if not self.bin:
            log("[LCMJPEG] Neither rpicam-vid nor libcamera-vid was found in PATH")
            return False
        cmd = [self.bin, "-t", "0", "--nopreview", "--codec", "mjpeg",
               "--width", str(self.width), "--height", str(self.height),
               "--framerate", str(self.fps), "-o", "-"]
        try:
            log(f"[LCMJPEG] Start: {' '.join(cmd)}")
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        except Exception as e:
            log(f"[LCMJPEG] Launch failed: {e}"); self.proc = None; return False
        self._running = True
        self._th = threading.Thread(target=self._reader_loop, daemon=True); self._th.start()
        return True
    def _reader_loop(self):
        SOI = b"\xff\xd8"; EOI = b"\xff\xd9"; stdout = self.proc.stdout if self.proc else None
        while self._running and self.proc and stdout:
            try:
                chunk = stdout.read(4096)
                if not chunk:
                    time.sleep(0.005)
                    if self.proc.poll() is not None:
                        log("[LCMJPEG] Process ended"); break
                    continue
                self._buf.extend(chunk)
                while True:
                    soi = self._buf.find(SOI)
                    if soi < 0:
                        if len(self._buf) > 1_000_000: self._buf[:] = self._buf[-200_000:]
                        break
                    eoi = self._buf.find(EOI, soi+2)
                    if eoi < 0:
                        if soi > 0: del self._buf[:soi]
                        break
                    eoi += 2
                    jpg = self._buf[soi:eoi]; del self._buf[:eoi]
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        try:
                            if self.q.full(): _ = self.q.get_nowait()
                        except Exception: pass
                        self.q.put_nowait(frame)
            except Exception as e:
                log(f"[LCMJPEG] Reader error: {e}"); time.sleep(0.02)
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frame = self.q.get_nowait(); return True, frame
        except queue.Empty:
            return False, None
    def release(self):
        self._running = False
        if self._th: self._th.join(timeout=0.5)
        if self.proc:
            try:
                if self.proc.poll() is None:
                    self.proc.terminate()
                    try: self.proc.wait(timeout=0.3)
                    except subprocess.TimeoutExpired: self.proc.kill()
                if self.proc.stdout: self.proc.stdout.close()
                if self.proc.stderr: self.proc.stderr.close()
            except Exception: pass
        self.proc = None; self._buf.clear()
        with self.q.mutex: self.q.queue.clear()
        log("[LCMJPEG] Released")

# ───────── Caméra : GST / LC-MJPEG / V4L2 ─────────
def cv_has_gstreamer() -> bool:
    try: bi = cv2.getBuildInformation()
    except Exception: return False
    for line in bi.splitlines():
        if "GStreamer" in line and "YES" in line: return True
    return False

def default_gst_pipeline() -> str:
    w = int(os.getenv("PI_CAM_WIDTH", "1280")); h = int(os.getenv("PI_CAM_HEIGHT", "720")); fps = int(os.getenv("PI_CAM_FPS", "30"))
    pipe_env = os.getenv("PI_GST_PIPELINE")
    if pipe_env: return pipe_env
    return (f"libcamerasrc ! video/x-raw,width={w},height={h},framerate={fps}/1,format=BGRx "
            f"! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1")

def list_v4l2_devices() -> List[Tuple[str, int, str]]:
    """
    Retourne une liste [(label, index, devpath)] pour /dev/video*.
    Essaie d'utiliser v4l2-ctl pour un label plus parlant, sinon fallback.
    """
    devs = sorted(glob.glob("/dev/video*"))
    out: List[Tuple[str,int,str]] = []
    name_by_dev = {}
    if _shutil.which("v4l2-ctl"):
        try:
            txt = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True)
            cur = None
            for line in txt.splitlines():
                if not line.strip(): continue
                if not line.startswith("\t"):
                    cur = line.strip()
                else:
                    d = line.strip()
                    if d.startswith("/dev/video"):
                        name_by_dev[d] = cur or d
        except Exception:
            pass
    for d in devs:
        try: idx = int(Path(d).name.replace("video",""))
        except Exception: continue
        label = f"{name_by_dev.get(d, 'V4L2')} ({d})"
        out.append((label, idx, d))
    return out

# ───────── Thread d’inférence ─────────
class InferWorker(QObject):
    resultReady = Signal(np.ndarray, dict, float)
    def __init__(self, infer_adapter: BaseAdapter, prob_threshold: float):
        super().__init__(); self.infer = infer_adapter; self.prob_threshold = float(prob_threshold)
        self.queue = queue.Queue(maxsize=1); self._running = True
    @Slot()
    def run(self):
        while self._running:
            try: frame = self.queue.get(timeout=0.1)
            except queue.Empty: continue
            t0 = time.time(); outputs = self.infer.predict_bgr(frame); elapsed = time.time() - t0
            self.resultReady.emit(frame, outputs, elapsed)
    def submit(self, frame: np.ndarray):
        try:
            if self.queue.full(): _ = self.queue.get_nowait()
            self.queue.put_nowait(frame)
        except queue.Full: pass
    def stop(self): self._running = False

# ───────── Petit widget : section repliable (mobile) ─────────
class Collapsible(QWidget):
    def __init__(self, title: str, parent=None, expanded=False):
        super().__init__(parent)
        self.btn = QPushButton(title); self.btn.setCheckable(True); self.btn.setChecked(expanded)
        self.btn.setObjectName("collapsibleHeader")
        self.body = QWidget(); self.body.setVisible(expanded)
        self.v = QVBoxLayout(self); self.v.setContentsMargins(0,0,0,0); self.v.setSpacing(6)
        self.v.addWidget(self.btn); self.v.addWidget(self.body)
        self.btn.toggled.connect(self.body.setVisible)
    def setContentLayout(self, layout: QLayout):
        self.body.setLayout(layout)

# ───────── Scrim cliquable (voile) ─────────
class ClickableScrim(QFrame):
    def __init__(self, on_click, parent=None):
        super().__init__(parent)
        self._on_click = on_click
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
    def mousePressEvent(self, e):
        if callable(self._on_click): self._on_click()
        e.accept()

# ───────── Drawer réutilisable (UN SEUL, côté gauche) ─────────
class DrawerSection(QGroupBox):
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(10, 8, 10, 10)
        self.layout().setSpacing(8)

class SideDrawer(QFrame):
    def __init__(self, parent, side: str = "left", title: str = "", on_close=None):
        super().__init__(parent)
        assert side in ("left", "right")
        self.side = side
        self.on_close = on_close
        self.setObjectName("drawerLeft" if side == "left" else "drawerRight")
        self.setVisible(False)

        # Shadow (léger pour Pi)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(22)
        shadow.setOffset(-2 if side == "right" else 2, 0)
        shadow.setColor(Qt.black)
        self.setGraphicsEffect(shadow)

        # Contenu
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        header = QFrame(); header.setObjectName("drawerHeader")
        h = QHBoxLayout(header); h.setContentsMargins(12,12,12,12); h.setSpacing(8)
        self.titleLbl = QLabel(title); self.titleLbl.setObjectName("drawerTitle")
        self.subtitleLbl = QLabel("Configuration"); self.subtitleLbl.setObjectName("drawerSubtitle")
        titleWrap = QVBoxLayout(); titleWrap.setContentsMargins(0,0,0,0); titleWrap.setSpacing(0)
        titleWrap.addWidget(self.titleLbl); titleWrap.addWidget(self.subtitleLbl)
        closeBtn = QToolButton(); closeBtn.setText("✕"); closeBtn.setCursor(Qt.PointingHandCursor)
        closeBtn.clicked.connect(self.close)
        h.addLayout(titleWrap); h.addStretch(); h.addWidget(closeBtn)
        root.addWidget(header)

        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.inner = QWidget(); self.scroll.setWidget(self.inner)
        self.inner.setLayout(QVBoxLayout()); self.inner.layout().setContentsMargins(12,12,12,12); self.inner.layout().setSpacing(12)
        root.addWidget(self.scroll)

        footer = QFrame(); footer.setObjectName("drawerFooter")
        f = QHBoxLayout(footer); f.setContentsMargins(12,10,12,12); f.setSpacing(8)
        self.primaryBtn = QPushButton("Fermer"); self.primaryBtn.clicked.connect(self.close)
        f.addStretch(); f.addWidget(self.primaryBtn)
        root.addWidget(footer)

        # Animation
        self.anim = QPropertyAnimation(self, b"geometry", self)
        self.anim.setDuration(220)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)

    def setContent(self, widget_or_layout):
        lay = self.inner.layout()
        # IMPORTANT: ne pas détruire les widgets partagés → le MainWindow les détache avant.
        while lay.count():
            child = lay.takeAt(0)
            w = child.widget()
            if w: w.setParent(None)
        if isinstance(widget_or_layout, QLayout):
            wrap = QWidget(); wrap.setLayout(widget_or_layout)
            lay.addWidget(wrap)
        else:
            lay.addWidget(widget_or_layout)
        lay.addStretch(1)

    def _target_rects(self):
        central = self.parent()
        dw = max(320, min(520, int(central.width() * 0.78)))
        ch = central.height()
        if self.side == "right":
            show = QRect(central.width() - dw, 0, dw, ch)
            hide = QRect(central.width(), 0, dw, ch)
        else:
            show = QRect(0, 0, dw, ch)
            hide = QRect(-dw, 0, dw, ch)
        return show, hide

    def layout_now(self, initial=False):
        show, hide = self._target_rects()
        if initial or not self.isVisible():
            self.setGeometry(hide)
        self._geom_show = show; self._geom_hide = hide

    def open(self):
        self.layout_now()
        self.setVisible(True)
        self.raise_()
        self.anim.stop()
        self.anim.setStartValue(self._geom_hide)
        self.anim.setEndValue(self._geom_show)
        self.anim.start()

    def close(self):
        self.layout_now()
        self.anim.stop()
        self.anim.setStartValue(self._geom_show)
        self.anim.setEndValue(self._geom_hide)
        self.anim.start()
        self.anim.finished.connect(self._after_close)

    def _after_close(self):
        try: self.anim.finished.disconnect(self._after_close)
        except Exception: pass
        self.setVisible(False)
        if callable(self.on_close): self.on_close()

# ───────── Sélecteur de tâches (dialog desktop) ─────────
class TaskSelectionDialog(QDialog):
    def __init__(self, parent, all_tasks: Dict[str, List[str]], selected: List[str]):
        super().__init__(parent)
        self.setWindowTitle("Sélection des tâches")
        self.all_tasks = all_tasks
        self.checks: Dict[str, QCheckBox] = {}
        v = QVBoxLayout(self); v.setContentsMargins(12,12,12,12); v.setSpacing(8)
        for t in all_tasks.keys():
            cb = QCheckBox(t); cb.setChecked(t in selected)
            self.checks[t] = cb; v.addWidget(cb)
        row = QHBoxLayout()
        btnAll = QPushButton("Tout"); btnNone = QPushButton("Aucun")
        btnAll.clicked.connect(lambda: [c.setChecked(True) for c in self.checks.values()])
        btnNone.clicked.connect(lambda: [c.setChecked(False) for c in self.checks.values()])
        row.addWidget(btnAll); row.addWidget(btnNone); row.addStretch(1)
        v.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        v.addWidget(bb)
    def selected_tasks(self) -> List[str]:
        return [t for t, cb in self.checks.items() if cb.isChecked()]

# ───────── Fenêtre principale ─────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S.T.I Innovation — Real-time weather detection")
        self.setMinimumSize(920, 560)

        # Caméra
        self.cap = None; self.cap_backend = None
        self.lc_mjpeg: Optional[LibcameraMJPEGCapture] = None
        self._read_fail = 0; self._tick = 0

        # Timer affichage
        self.timer = QTimer(self); self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._on_timer_capture); self.target_fps = 20

        # IA
        self.infer_thread = None; self.infer_worker = None
        self.current_adapter: Optional[BaseAdapter] = None
        self.tasks: Dict[str, List[str]] = {}        # tâches en cours (affichage)
        self.all_tasks: Dict[str, List[str]] = {}    # toutes les tâches du modèle
        self.tasks_order: List[str] = []
        self.selected_tasks: List[str] = []          # sélection utilisateur (pour PRUNE)
        self.device = "cpu"

        # Dernières prédictions
        self.last_items: List[Tuple[str,str,float]] = []; self.last_speed_text: str = ""
        self._last_disp_t = time.time(); self._disp_fps_ema = None

        # Session & I/O
        self.recording_video = False; self.video_writer = None
        self.output_dir = Path("runs"); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_rows: List[Dict[str, Any]] = []; self.meta = {}
        self.session_started_at: Optional[datetime.datetime] = None

        # Lecture
        self.playback_mode = False; self.play_source_path: Optional[Path] = None
        self.play_fps = 25.0

        # Vitesse IA
        self.lat_ema_s = None; self.fps_hist = deque(maxlen=30)

        # Perf : inférence 1/N
        self.infer_stride = 1

        # Drawers (MOBILE)
        # Drawers (mobile) — APRES tes lignes:
        self.scrim: Optional[ClickableScrim] = None
        self.drawerEss: Optional[SideDrawer] = None  # un seul drawer (à gauche)
        self._open_drawer: Optional[str] = None  # "ess" | None
        self._adv_anchor: Optional[QWidget] = None  # ancre “Avancé”

        # ➜ AJOUTER cette ligne (manquante) :
        self._ui_mode: Optional[str] = None  # "mobile" | "desktop"

        # UI
        self._build_shared_widgets()
        self._build_ui()               # construit panneau + drawer
        self.reload_cameras()          # remplit la liste de caméras disponibles
        self.reload_models()           # remplit la liste de modèles
        self._on_model_combo_changed() # met en place les tâches (défaut: 1ère tâche)

        # Fullscreen toggle
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)

    # ── Widgets partagés (créés 1 fois) ──
    def _build_shared_widgets(self):
        # Haut
        self.logo = QLabel(); self.logo.setObjectName("logo")
        self.logo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.logoPath = str(Path(__file__).parent / "images" / "logo_cerema.png")
        self.appTitle = QLabel("S.T.I-WeatherMeasure"); self.appTitle.setStyleSheet("font-size:20px; font-weight:700;")

        # Vidéo
        self.videoLabel = QLabel("Prévisualisation vidéo")
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoLabel.setStyleSheet("background:#000; color:#aaa;")

        # Contrôles (partagés)
        self.modelCombo = QComboBox()
        self.modelCombo.currentIndexChanged.connect(self._on_model_combo_changed)
        self.reloadBtn = QPushButton("Recharger modèles"); self.reloadBtn.clicked.connect(self.reload_models)
        self.deviceLabel = QLabel("Device: CPU")
        self.classesEdit = QLineEdit(); self.classesEdit.setPlaceholderText("classes JSON (facultatif)")
        self.chooseClasses = QPushButton("Parcourir classes…"); self.chooseClasses.clicked.connect(self._choose_classes)

        self.cameraCombo = QComboBox()
        self.rescanCamBtn = QPushButton("Rafraîchir caméras"); self.rescanCamBtn.clicked.connect(self.reload_cameras)

        self.fpsSpin = QSpinBox(); self.fpsSpin.setRange(5, 60); self.fpsSpin.setValue(20)
        self.threshSpin = QDoubleSpinBox(); self.threshSpin.setRange(0.0, 1.0); self.threshSpin.setSingleStep(0.05); self.threshSpin.setValue(0.5)
        self.inferEverySpin = QSpinBox(); self.inferEverySpin.setRange(1, 8); self.inferEverySpin.setValue(1)
        self.inferEverySpin.valueChanged.connect(lambda v: setattr(self, "infer_stride", int(v)))
        self.formatCombo = QComboBox(); self.formatCombo.addItems(["json", "csv", "xlsx", "txt"])
        self.sessionEdit = QLineEdit(); self.sessionEdit.setPlaceholderText("Nom de session (défaut: <modèle>_<timestamp>)")
        self.dirEdit = QLineEdit(str(self.output_dir))
        self.chooseDir = QPushButton("Dossier sortie…"); self.chooseDir.clicked.connect(self._choose_dir)
        self.speedCheck = QCheckBox("Afficher vitesse (IA/affichage)"); self.speedCheck.setChecked(False)

        # Actions (mobile — source de vérité)
        self.burgerBtn = QToolButton(); self.burgerBtn.setText("☰"); self.burgerBtn.setObjectName("burgerBtn")
        self.burgerBtn.setToolTip("Options")
        self.burgerBtn.clicked.connect(lambda: self._toggle_drawer("ess"))

        self.gearBtn = QToolButton(); self.gearBtn.setText("⚙"); self.gearBtn.setObjectName("gearBtn")
        self.gearBtn.setToolTip("Options avancées")
        # Ouvre le même drawer et défile vers l'ancre "Avancé"
        self.gearBtn.clicked.connect(lambda: self._toggle_drawer("adv"))

        self.startBtn = QPushButton("▶ Démarrer"); self.startBtn.setObjectName("startBtn")
        self.stopBtn  = QPushButton("■ Stop");      self.stopBtn.setObjectName("stopBtn"); self.stopBtn.setEnabled(False)
        self.recBtn   = QPushButton("● Enregistrer"); self.recBtn.setObjectName("recBtn"); self.recBtn.setCheckable(True); self.recBtn.setChecked(False)
        self.fullBtn  = QPushButton("⤢ Plein écran"); self.fullBtn.setObjectName("fullBtn")
        self.startBtn.clicked.connect(self.start_session)
        self.stopBtn.clicked.connect(self.stop_session)
        self.recBtn.toggled.connect(self.toggle_video_record)
        self.fullBtn.clicked.connect(self._toggle_fullscreen)

        # Actions (desktop — miroirs qui pilotent les boutons mobile)
        self.startBtnDesk = QPushButton("▶ Démarrer"); self.startBtnDesk.clicked.connect(self.start_session)
        self.stopBtnDesk  = QPushButton("■ Stop");      self.stopBtnDesk.clicked.connect(self.stop_session)
        self.recBtnDesk   = QPushButton("● Enregistrer"); self.recBtnDesk.setCheckable(True)
        self.recBtnDesk.toggled.connect(lambda c: self.recBtn.setChecked(c))
        self.fullBtnDesk  = QPushButton("⤢ Plein écran"); self.fullBtnDesk.clicked.connect(self._toggle_fullscreen)
        self.recBtn.toggled.connect(lambda c: self.recBtnDesk.setChecked(c))

        # Desktop: bouton pour ouvrir la sélection des tâches
        self.taskDialogBtn = QPushButton("Sélectionner tâches…")
        self.taskDialogBtn.clicked.connect(self._open_task_dialog)

        self.playBtn = QPushButton("Lire un enregistrement…"); self.playBtn.clicked.connect(self.open_recording_file)
        self.transferBtn = QPushButton("Transférer fichiers…"); self.transferBtn.clicked.connect(self.transfer_files)
        self.emailBtn = QPushButton("Envoyer par e-mail…"); self.emailBtn.clicked.connect(self.send_email_dialog)

    # Détacher tous les widgets partagés avant de changer de layout/parent
    def _detach_shared_controls(self):
        shared = [
            self.modelCombo, self.reloadBtn, self.deviceLabel, self.taskDialogBtn,
            self.classesEdit, self.chooseClasses,
            self.cameraCombo, self.rescanCamBtn,
            self.fpsSpin, self.threshSpin, self.inferEverySpin, self.speedCheck,
            self.formatCombo, self.sessionEdit, self.dirEdit, self.chooseDir,
            self.playBtn, self.transferBtn, self.emailBtn
        ]
        for w in shared:
            try:
                if w and w.parent() is not None:
                    w.setParent(None)
            except Exception:
                pass

    # ── Construction UI conteneurs ──
    def _build_ui(self):
        # Topbar
        topbar = QHBoxLayout()
        topbar.addWidget(self.logo); topbar.addWidget(self.appTitle); topbar.addStretch()

        # Desktop controls (scroll)
        self.controlsScroll = QScrollArea(); self.controlsScroll.setWidgetResizable(True)
        self.controlsScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controlsPanelDesktop = QWidget(); self.controlsScroll.setWidget(self.controlsPanelDesktop)
        self._mount_controls_desktop()

        # Mobile controls (barre ; les options sont dans le drawer)
        self.mobilePanel = QWidget()
        self._mount_controls_mobile()

        # Grille centrale
        central = QWidget(); self.setCentralWidget(central)
        self.grid = QGridLayout(central); self.grid.setContentsMargins(10, 10, 10, 10); self.grid.setSpacing(10)
        self.grid.addLayout(topbar, 0, 0, 1, 2)
        self.grid.addWidget(self.videoLabel, 1, 0, 1, 1)
        self.grid.addWidget(self.controlsScroll, 1, 1, 1, 1)
        self.grid.addWidget(self.mobilePanel,    2, 0, 1, 2)   # masquée par défaut en desktop
        self.grid.setRowStretch(1, 1)
        self.grid.setColumnStretch(0, 3); self.grid.setColumnStretch(1, 1)

        # Menu
        actOpenOut = QAction("Ouvrir dossier sorties", self)
        actOpenOut.triggered.connect(lambda: os.system(f'xdg-open "{self.output_dir}" 2>/dev/null || open "{self.output_dir}"'))
        self.menuBar().addAction(actOpenOut)

        # Drawer unique (MOBILE)
        self._build_drawers()

        self._update_logo_pixmap(1.0)
        self._apply_ui_scale(); self._apply_responsive_layout(force=True)

    # ── Montage Desktop ──
    def _clear_layout(self, layout: QLayout):
        if not layout: return
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w: w.setParent(None)

    def _mount_controls_desktop(self):
        # Toujours détacher d'abord les widgets partagés de tout parent (drawer)
        self._detach_shared_controls()

        panel = self.controlsPanelDesktop
        v = panel.layout() or QVBoxLayout(panel)      # réutilise le layout s'il existe
        self._clear_layout(v)                          # supprime les items, pas le layout

        # Bloc modèle
        boxModel = QGroupBox("Modèle"); l = QVBoxLayout(boxModel)
        row0 = QHBoxLayout(); row0.addWidget(self.modelCombo, 1); row0.addWidget(self.reloadBtn)
        l.addLayout(row0); l.addWidget(self.deviceLabel)
        l.addWidget(self.taskDialogBtn)
        v.addWidget(boxModel)

        # Classes
        boxCls = QGroupBox("Classes"); l2 = QVBoxLayout(boxCls)
        row1 = QHBoxLayout(); row1.addWidget(self.classesEdit, 1); row1.addWidget(self.chooseClasses)
        l2.addLayout(row1)
        v.addWidget(boxCls)

        # Caméra/runtime
        boxCam = QGroupBox("Caméra & runtime"); form = QFormLayout(boxCam)
        form.addRow("Caméra :", self.cameraCombo)
        form.addRow("", self.rescanCamBtn)
        form.addRow("FPS affichage :", self.fpsSpin)
        form.addRow("Seuil proba :", self.threshSpin)
        form.addRow("Inférence 1 image sur :", self.inferEverySpin)
        v.addWidget(boxCam)
        v.addWidget(self.speedCheck)

        # Export
        boxExp = QGroupBox("Export résumé"); form2 = QFormLayout(boxExp)
        form2.addRow("Format :", self.formatCombo)
        form2.addRow("Nom session :", self.sessionEdit)
        dirRow = QHBoxLayout(); dirRow.addWidget(self.dirEdit, 1); dirRow.addWidget(self.chooseDir)
        wdir = QWidget(); wdir.setLayout(dirRow)
        form2.addRow("Dossier :", wdir)
        v.addWidget(boxExp)

        # Actions (miroir desktop)
        actionBox = QGroupBox("Actions"); ha = QHBoxLayout(actionBox)
        ha.addWidget(self.startBtnDesk); ha.addWidget(self.stopBtnDesk)
        ha.addWidget(self.recBtnDesk); ha.addWidget(self.fullBtnDesk)
        v.addWidget(actionBox)

        # Lecture & partage
        share = QGroupBox("Enregistrements & partage"); ls = QVBoxLayout(share)
        ls.addWidget(self.playBtn); ls.addWidget(self.transferBtn); ls.addWidget(self.emailBtn)
        v.addWidget(share)

        v.addStretch(1)

    # ── Montage Mobile (barre ; drawer pour les options) ──
    def _mount_controls_mobile(self):
        mob = self.mobilePanel
        v = QVBoxLayout(mob); v.setContentsMargins(0,0,0,0); v.setSpacing(0)

        # Barre d'actions (gros boutons) + ☰ à gauche + ⚙ à droite (même drawer, nav rapide)
        bar = QFrame(); bar.setObjectName("mobileBar")
        hb = QHBoxLayout(bar); hb.setContentsMargins(10,10,10,10); hb.setSpacing(10)
        hb.addWidget(self.burgerBtn)
        hb.addWidget(self.startBtn); hb.addWidget(self.stopBtn); hb.addWidget(self.recBtn); hb.addWidget(self.fullBtn)
        hb.addWidget(self.gearBtn)
        v.addWidget(bar)

        # Zone d'info
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); scroll.setWidget(inner)
        vv = QVBoxLayout(inner); vv.setContentsMargins(10,10,10,10); vv.setSpacing(10)
        info = QLabel("Appuyez sur ☰ pour les options. Le bouton ⚙ ouvre directement la section Avancé.")
        info.setStyleSheet("color:#9aa; padding:10px;")
        vv.addWidget(info); vv.addStretch(1)
        v.addWidget(scroll)

        # Style mobile
        self._style_buttons(mobile=True)

    # ── Drawer unique (contenu : Essentiel + Avancé) ──
    def _build_drawers(self):
        central = self.centralWidget()
        if not central: return

        # Scrim unique
        self.scrim = ClickableScrim(lambda: self._close_any_drawer(), parent=central)
        self.scrim.setVisible(False)
        self.scrim.setStyleSheet("background: rgba(0,0,0,160);")

        # Drawer Essentiel (gauche) — contient aussi Avancé plus bas
        self.drawerEss = SideDrawer(central, side="left", title="Options", on_close=self._on_drawer_closed)

        # Styles
        self._style_drawers()

        # Position initiale
        self._layout_drawers(initial=True)

        # Remplir avec le contenu initial
        self._populate_drawers_content()

    def _style_drawers(self):
        self.setStyleSheet(self.styleSheet() + """
        QFrame#drawerLeft {
            background: #0F172A; color: #e5e7eb;
            border-top-left-radius:16px; border-bottom-left-radius:16px;
        }
        #drawerHeader {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0EA5E9, stop:1 #6366F1);
            border-top-left-radius:16px; border-top-right-radius:16px;
        }
        #drawerTitle { font-size:18px; font-weight:800; color:#fff; }
        #drawerSubtitle { font-size:12px; color: rgba(255,255,255,0.85); }
        QGroupBox {
            border:1px solid #1f2937; border-radius:12px; margin-top:12px; padding-top:10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left:8px; padding: 0 6px; color:#93c5fd; font-weight:700;
        }
        #drawerFooter { background:#0b1327; border-bottom-left-radius:16px; border-bottom-right-radius:16px; }
        """)

    def _make_tasks_widget(self) -> QWidget:
        """Bloc 'Tâches' (checkboxes) dans le drawer."""
        wrap = QWidget(); lay = QVBoxLayout(wrap); lay.setContentsMargins(4,0,4,0); lay.setSpacing(6)
        self._taskChecks: Dict[str, QCheckBox] = {}
        if not self.all_tasks:
            lay.addWidget(QLabel("Aucune tâche (sélectionnez un modèle)."))
            return wrap
        info = QLabel("Tâches affichées / entraînées (prune au rechargement) :")
        info.setStyleSheet("color:#cbd5e1;")
        lay.addWidget(info)
        for t in self.all_tasks.keys():
            cb = QCheckBox(t); cb.setChecked(t in self.selected_tasks); self._taskChecks[t] = cb
            lay.addWidget(cb)
        row = QHBoxLayout()
        btnAll = QPushButton("Tout"); btnNone = QPushButton("Aucun"); btnApply = QPushButton("Appliquer")
        btnAll.clicked.connect(lambda: [c.setChecked(True) for c in self._taskChecks.values()])
        btnNone.clicked.connect(lambda: [c.setChecked(False) for c in self._taskChecks.values()])
        btnApply.clicked.connect(self._apply_task_selection_from_drawer)
        row.addWidget(btnAll); row.addWidget(btnNone); row.addStretch(1); row.addWidget(btnApply)
        lay.addLayout(row)
        return wrap

    def _populate_drawers_content(self):
        self._detach_shared_controls()

        essLay = QVBoxLayout()

        # mini-nav interne
        nav = QHBoxLayout()
        btnEss = QPushButton("Essentiel")
        btnAdv = QPushButton("Avancé")
        btnEss.clicked.connect(lambda: self.drawerEss.scroll.ensureWidgetVisible(self.drawerEss.inner, 0, 0))
        btnAdv.clicked.connect(lambda: self._scroll_to_advanced())
        nav.addWidget(btnEss);
        nav.addWidget(btnAdv);
        nav.addStretch(1)
        essLay.addLayout(nav)

        # --- Section Essentiel ---
        secModel = DrawerSection("Modèle & tâches")
        rowm = QHBoxLayout();
        rowm.addWidget(self.modelCombo, 1);
        rowm.addWidget(self.reloadBtn)
        secModel.layout().addLayout(rowm)
        secModel.layout().addWidget(self.deviceLabel)
        secModel.layout().addWidget(self._make_tasks_widget())
        essLay.addWidget(secModel)

        secCam = DrawerSection("Caméra & runtime")
        formc = QFormLayout();
        formc.setLabelAlignment(Qt.AlignRight)
        formc.addRow("Caméra :", self.cameraCombo)
        formc.addRow("", self.rescanCamBtn)
        formc.addRow("FPS :", self.fpsSpin)
        formc.addRow("Seuil :", self.threshSpin)
        formc.addRow("Inférence 1/N :", self.inferEverySpin)
        secCam.layout().addLayout(formc)
        secCam.layout().addWidget(self.speedCheck)
        essLay.addWidget(secCam)

        # --- Ancre Avancé ---
        self._adv_anchor = QLabel("")
        self._adv_anchor.setFixedHeight(1)
        essLay.addWidget(self._adv_anchor)

        # --- Section Avancé ---
        secCls = DrawerSection("Classes")
        row1 = QHBoxLayout();
        row1.addWidget(self.classesEdit, 1);
        row1.addWidget(self.chooseClasses)
        secCls.layout().addLayout(row1)
        essLay.addWidget(secCls)

        secExp = DrawerSection("Export / Session")
        form2 = QFormLayout();
        form2.setLabelAlignment(Qt.AlignRight)
        form2.addRow("Format :", self.formatCombo)
        form2.addRow("Nom session :", self.sessionEdit)
        dirRow = QHBoxLayout();
        dirRow.addWidget(self.dirEdit, 1);
        dirRow.addWidget(self.chooseDir)
        wdir = QWidget();
        wdir.setLayout(dirRow)
        form2.addRow("Dossier :", wdir)
        secExp.layout().addLayout(form2)
        essLay.addWidget(secExp)

        secShare = DrawerSection("Enregistrements & partage")
        secShare.layout().addWidget(self.playBtn)
        secShare.layout().addWidget(self.transferBtn)
        secShare.layout().addWidget(self.emailBtn)
        essLay.addWidget(secShare)

        self.drawerEss.setContent(essLay)

    def _scroll_to_advanced(self):
        if self.drawerEss and self._adv_anchor:
            self.drawerEss.scroll.ensureWidgetVisible(self._adv_anchor, 0, 24)

    def _layout_drawers(self, initial=False):
        if self.drawerEss: self.drawerEss.layout_now(initial=initial)
        if self.scrim and self.centralWidget():
            self.scrim.setGeometry(self.centralWidget().rect())

    def _scroll_to_advanced(self):
        if self.drawerEss and self._adv_anchor:
            self.drawerEss.scroll.ensureWidgetVisible(self._adv_anchor, 0, 24)

    def _toggle_drawer(self, which: str):
        if not self._is_mobile():
            return
        if which == self._open_drawer:
            if which == "adv":
                self._scroll_to_advanced()
            else:
                self._close_any_drawer()
            return
        self._close_any_drawer(silent=True)
        if self.drawerEss:
            self._show_scrim(True);
            self.drawerEss.open();
            self._open_drawer = "ess"
            if which == "adv":
                QTimer.singleShot(250, self._scroll_to_advanced)  # après l’anim

    def _close_any_drawer(self, silent=False):
        if self._open_drawer == "ess" and self.drawerEss:
            self.drawerEss.close()
        if not silent: self._open_drawer = None

    def _on_drawer_closed(self):
        self._open_drawer = None
        self._show_scrim(False)

    def _show_scrim(self, show: bool):
        if not self.scrim: return
        self.scrim.setVisible(show)
        if show: self.scrim.raise_()

    # ── Responsive ──
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
        self._apply_responsive_layout(force=True)
        self._layout_drawers(initial=True)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply_responsive_layout()
        self._apply_ui_scale()
        self._layout_drawers()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self._open_drawer:
            self._close_any_drawer(); e.accept(); return
        super().keyPressEvent(e)

    def _on_screen_changed(self, screen):
        dpi = screen.logicalDotsPerInch() or 96.0
        scale = dpi / 96.0
        f = QFont(); f.setPointSizeF(11.0 * scale); self.setFont(f)
        log(f"[UI] Screen changed. DPI: {dpi} scale: {scale}")
        self._update_logo_pixmap(scale)
        self._apply_ui_scale()

    def _is_mobile(self) -> bool:
        w, h = self.width(), self.height()
        return (w < 980) or (h < 620)

    def _apply_responsive_layout(self, force=False):
        mobile = self._is_mobile()
        mode = getattr(self, "_ui_mode", None)

        if mobile and mode != "mobile":
            self.logo.setVisible(False);
            self.appTitle.setVisible(False)
            self.burgerBtn.setVisible(True);
            self.gearBtn.setVisible(True)
            self.controlsScroll.setVisible(False)
            self.mobilePanel.setVisible(True)
            self.grid.removeWidget(self.videoLabel)
            self.grid.addWidget(self.videoLabel, 1, 0, 1, 2)
            self.grid.setColumnStretch(0, 1);
            self.grid.setColumnStretch(1, 0)
            self.grid.setRowStretch(1, 100);
            self.grid.setRowStretch(2, 1)
            self.videoLabel.setMinimumHeight(int(max(360, max(480, self.height()) * 0.72)))
            self.grid.setContentsMargins(6, 6, 6, 6)
            self._style_buttons(mobile=True)
            self._populate_drawers_content()  # (re)peuple le tiroir
            self._ui_mode = "mobile"

        elif (not mobile) and mode != "desktop":
            self.logo.setVisible(True);
            self.appTitle.setVisible(True)
            self.burgerBtn.setVisible(False);
            self.gearBtn.setVisible(False)
            self.mobilePanel.setVisible(False)
            self.controlsScroll.setVisible(True)
            self.grid.removeWidget(self.videoLabel)
            self.grid.addWidget(self.videoLabel, 1, 0, 1, 1)
            self.grid.setColumnStretch(0, 3);
            self.grid.setColumnStretch(1, 1)
            self.grid.setRowStretch(1, 1);
            self.grid.setRowStretch(2, 0)
            self.videoLabel.setMinimumHeight(0)
            self.grid.setContentsMargins(10, 10, 10, 10)
            self._style_buttons(mobile=False)
            self._close_any_drawer(silent=True)
            self._mount_controls_desktop()
            self._ui_mode = "desktop"

        if (not mobile) and getattr(self, "_open_drawer", None):
            self._close_any_drawer()

    def _apply_ui_scale(self):
        w, h = max(640, self.width()), max(480, self.height())
        ui_scale = max(0.85, min(1.35, min(w / 1200.0, h / 700.0)))
        base_pt = 11.0 * ui_scale
        f = QFont(self.font()); f.setPointSizeF(base_pt); self.setFont(f)
        pad = max(6, int(8 * ui_scale))
        ss = f"""
        QPushButton {{ padding:{pad}px; }}
        QToolButton {{ padding:{max(6,int(6*ui_scale))}px; border-radius:12px; }}
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ padding:{max(4,int(5*ui_scale))}px; }}
        #collapsibleHeader {{ text-align:left; font-weight:700; padding:{pad}px; border-radius:10px; background:#0ea5e9; color:white; }}
        #mobileBar {{ background:#0b1020; }}
        QToolButton#burgerBtn, QToolButton#gearBtn {{
            background:#1f2937; color:white; font-weight:900; font-size:{int(18*ui_scale)}px;
            min-width: 44px; min-height: 44px;
        }}
        """
        self.setStyleSheet(ss)

    def _style_buttons(self, mobile: bool):
        if mobile:
            style = """
            QPushButton#startBtn { background:#16a34a; color:white; font-weight:800; font-size:18px; padding:14px 18px; border-radius:16px; }
            QPushButton#stopBtn  { background:#dc2626; color:white; font-weight:800; font-size:18px; padding:14px 18px; border-radius:16px; }
            QPushButton#recBtn   { background:#ef4444; color:white; font-weight:800; font-size:18px; padding:14px 18px; border-radius:16px; }
            QPushButton#fullBtn  { background:#334155; color:white; font-weight:800; font-size:18px; padding:14px 18px; border-radius:16px; }
            """
        else:
            style = """
            QPushButton#startBtn { background:#22c55e; color:white; font-weight:700; font-size:15px; padding:10px 14px; border-radius:14px; }
            QPushButton#stopBtn  { background:#f43f5e; color:white; font-weight:700; font-size:15px; padding:10px 14px; border-radius:14px; }
            QPushButton#recBtn   { background:#ef4444; color:white; font-weight:700; font-size:15px; padding:10px 14px; border-radius:14px; }
            QPushButton#fullBtn  { background:#334155; color:white; font-weight:700; font-size:15px; padding:10px 14px; border-radius:14px; }
            """
        self.setStyleSheet(self.styleSheet() + "\n" + style)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _update_logo_pixmap(self, scale: float):
        base_h = 36;
        target_h = max(24, int(base_h * max(0.75, scale)))
        if os.path.exists(self.logoPath):
            pm = QPixmap(self.logoPath)
            if not pm.isNull():
                pm = pm.scaledToHeight(target_h, Qt.SmoothTransformation)
                self.logo.setPixmap(pm);
                self.logo.setMinimumSize(pm.size());
                self.logo.setToolTip(self.logoPath);
                return
        self.logo.setText("LOGO");
        self.logo.setStyleSheet("font-weight:600; padding:6px;")

        # ── Modèles & tâches ──

    def reload_models(self):
        self.modelCombo.clear()
        models_dir = Path("models").resolve();
        models_dir.mkdir(exist_ok=True)
        patterns = ("**/*.pt", "**/*.pth", "**/*.tflite", "**/*.onnx")
        files: List[str] = []
        for pat in patterns: files += glob.glob(str(models_dir / pat), recursive=True)
        files = sorted(files);
        log(f"[Models] Found: {len(files)}")
        if not files:
            self.modelCombo.addItem("(aucun modèle trouvé…)");
            self.modelCombo.setEnabled(False)
        else:
            self.modelCombo.setEnabled(True)
            for f in files:
                fpath = Path(f).resolve()
                try:
                    rel = str(fpath.relative_to(models_dir))
                except Exception:
                    rel = str(fpath)
                self.modelCombo.addItem(rel, str(fpath))

    def _on_model_combo_changed(self):
        model_path = self.modelCombo.currentData() or self.modelCombo.currentText()
        model_path = str(model_path).strip()
        if not model_path or not Path(model_path).exists():
            self.all_tasks = {};
            self.tasks = {};
            self.selected_tasks = [];
            self.tasks_order = []
        else:
            self.all_tasks = self._load_tasks(model_path)
            # sélection par défaut = toutes
            self.selected_tasks = list(self.all_tasks.keys())
            self.tasks = dict(self.all_tasks)
            self.tasks_order = list(self.selected_tasks)
        # rafraîchir la zone de sélection des tâches (drawer/mobile)
        if self._is_mobile():
            self._populate_drawers_content()

    def _apply_task_selection_from_drawer(self):
        if not hasattr(self, "_taskChecks"): return
        self.selected_tasks = [t for t, cb in self._taskChecks.items() if cb.isChecked()]
        if not self.selected_tasks:
            # forcer au moins une tâche
            if self._taskChecks:
                first = next(iter(self._taskChecks.values()))
                first.setChecked(True)
                self.selected_tasks = [next(iter(self._taskChecks.keys()))]
        # Mettre à jour l’ordre et l’affichage courant (sans recharger le modèle)
        self.tasks_order = list(self.selected_tasks)
        self.tasks = {k: v for k, v in self.all_tasks.items() if k in self.selected_tasks}
        QMessageBox.information(self, "Tâches", "Sélection appliquée pour l’affichage.\n"
                                                "Remarque: la PRUNE effective du modèle aura lieu au prochain démarrage.")

    def _open_task_dialog(self):
        if not self.all_tasks:
            QMessageBox.information(self, "Tâches", "Aucune tâche disponible. Sélectionnez d’abord un modèle.")
            return
        dlg = TaskSelectionDialog(self, self.all_tasks, self.selected_tasks or list(self.all_tasks.keys()))
        if dlg.exec() == QDialog.Accepted:
            self.selected_tasks = dlg.selected_tasks()
            if not self.selected_tasks:
                self.selected_tasks = list(self.all_tasks.keys())
            self.tasks_order = list(self.selected_tasks)
            self.tasks = {k: v for k, v in self.all_tasks.items() if k in self.selected_tasks}
            if self._is_mobile():
                self._populate_drawers_content()

    def _choose_classes(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir classes.json", "", "JSON (*.json)")
        if path: self.classesEdit.setText(path)

    def _choose_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choisir dossier de sortie", str(self.output_dir))
        if path: self.dirEdit.setText(path)

    def _load_tasks(self, model_path: str) -> Dict[str, List[str]]:
        # 1) classes.json (si fourni)
        classes_path = self.classesEdit.text().strip()
        if classes_path and Path(classes_path).exists():
            data = json.loads(Path(classes_path).read_text(encoding="utf-8"))
            return {k: list(v) for k, v in data.items()}
        # 2) manifest/config à côté du modèle
        cfg = ModelRegistry._read_config(model_path)
        if "tasks" in cfg and isinstance(cfg["tasks"], dict):
            return {k: list(v) for k, v in cfg["tasks"].items()}
        # 3) fallback
        return {"Weather": ["No", "Yes"]}

    def _build_default_session_name(self, model_path: str) -> str:
        base = Path(model_path).stem;
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{ts}"

        # ── Caméras ──

    def reload_cameras(self):
        self.cameraCombo.clear()
        # presets haut-niveau
        self.cameraCombo.addItem("Auto (GStreamer ▶ libcamera-vid ▶ V4L2)", ("auto", None))
        self.cameraCombo.addItem("GStreamer (libcamerasrc)", ("gst", None))
        self.cameraCombo.addItem("rpicam-vid / libcamera-vid (MJPEG pipe)", ("lc-mjpeg", None))
        # V4L2 devices
        for label, idx, dev in list_v4l2_devices():
            self.cameraCombo.addItem(f"V4L2: {label}", ("v4l2", idx))

    def _try_open_v4l2(self, cam_index: int, w: int, h: int, fps: int):
        trials = [("MJPG", cv2.VideoWriter_fourcc(*"MJPG")), ("YUYV", cv2.VideoWriter_fourcc(*"YUYV")), ("AUTO", None)]
        for label, fourcc in trials:
            cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
            if not cap or not cap.isOpened(): continue
            if fourcc is not None: cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w);
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h);
            cap.set(cv2.CAP_PROP_FPS, fps)
            ok, frame = cap.read()
            if ok and frame is not None:
                log(f"[Camera] V4L2 OK with FOURCC={label}");
                return cap
            else:
                log(f"[Camera] V4L2 read failed with FOURCC={label}; retry");
                cap.release()
        return None

    def _open_camera_auto(self) -> bool:
        # essaie GST ▶ lc-mjpeg ▶ v4l2:0
        for pref in [("gst", None), ("lc-mjpeg", None), ("v4l2", 0)]:
            if self._open_camera_backend(pref[0], pref[1]): return True
        return False

    def _open_camera_backend(self, backend: str, index: Optional[int]) -> bool:
        self._release_camera()
        w = int(os.getenv("PI_CAM_WIDTH", "1280"));
        h = int(os.getenv("PI_CAM_HEIGHT", "720"));
        fps = int(os.getenv("PI_CAM_FPS", "30"))

        if backend == "gst":
            if cv_has_gstreamer():
                pipe = default_gst_pipeline();
                log(f"[Camera] Try GStreamer:\n{pipe}")
                cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                if cap is not None and cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        self.cap = cap;
                        self.cap_backend = "gst";
                        log("[Camera] GStreamer OK");
                        return True
                    log("[Camera] GStreamer opened but first read failed");
                    cap.release()
            else:
                log("[Camera] OpenCV built without GStreamer")
            return False

        if backend == "lc-mjpeg":
            self.lc_mjpeg = LibcameraMJPEGCapture(w, h, fps)
            if self.lc_mjpeg.start():
                t0 = time.time();
                ok, frame = False, None
                while time.time() - t0 < 1.2 and not ok:
                    ok, frame = self.lc_mjpeg.read()
                    if not ok: time.sleep(0.02)
                if ok and frame is not None:
                    self.cap_backend = "lc-mjpeg";
                    log("[Camera] rpicam-vid/libcamera-vid MJPEG OK");
                    return True
                else:
                    log("[Camera] libcamera-vid started but no frame");
                    self.lc_mjpeg.release();
                    self.lc_mjpeg = None
            else:
                log("[Camera] rpicam-vid/libcamera-vid backend not available")
            return False

        if backend == "v4l2":
            if index is None: index = 0
            log(f"[Camera] Try V4L2 index {index}")
            cap = self._try_open_v4l2(index, w, h, fps)
            if cap:
                self.cap = cap;
                self.cap_backend = "v4l2";
                return True
            log("[Camera] V4L2 failed on all tried FOURCCs")
            return False

        if backend == "auto":
            return self._open_camera_auto()

        return False

    def _open_camera_selected(self) -> bool:
        data = self.cameraCombo.currentData()
        if not data: data = ("auto", None)
        backend, idx = data
        ok = self._open_camera_backend(backend, idx)
        if not ok and backend != "auto":
            log("[Camera] Selected backend failed, fallback to AUTO")
            ok = self._open_camera_auto()
        return ok

    def _release_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.lc_mjpeg is not None:
            try:
                self.lc_mjpeg.release()
            except Exception:
                pass
            self.lc_mjpeg = None
        self.cap_backend = None;
        self._read_fail = 0

    def _reopen_camera_if_needed(self):
        self._read_fail += 1
        if self._read_fail % 20 == 0: log(f"[Capture] READ FAIL x{self._read_fail} (tick {self._tick})")
        if self._read_fail >= 60:
            log("[Capture] Too many read fails → reopen camera")
            self._open_camera_selected()
            self._read_fail = 0

        # ── Start/Stop ──

    def start_session(self):
        if self.playback_mode: self._stop_playback()
        model_path = self.modelCombo.currentData() or self.modelCombo.currentText()
        model_path = str(model_path).strip()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "Modèle", "Aucun modèle valide sélectionné.");
            return

        # Déterminer tâches (toutes + sélection)
        self.all_tasks = self._load_tasks(model_path)
        if not self.selected_tasks:
            self.selected_tasks = list(self.all_tasks.keys())  # default: toutes
        self.tasks = {k: v for k, v in self.all_tasks.items() if k in self.selected_tasks}
        self.tasks_order = list(self.selected_tasks)

        # Device
        self.device = "cuda" if hasattr(torch, "cuda") and getattr(torch.cuda, "is_available",
                                                                   lambda: False)() else "cpu"

        # Créer l'adapter avec PRUNE (selected_tasks) si supporté
        try:
            self.current_adapter = ModelRegistry.create(
                model_path=model_path,
                tasks=self.all_tasks,
                device=self.device,
                extra=None,
                selected_tasks=self.selected_tasks
            )
            # Adapter PatchGAN peut renvoyer tâches réelles (pruned)
            self.tasks = self.current_adapter.tasks
            self.tasks_order = list(self.tasks.keys())
        except Exception as e:
            QMessageBox.critical(self, "Chargement modèle", str(e));
            return
        self.deviceLabel.setText(f"Device: {'GPU' if self.device == 'cuda' else 'CPU'}")

        # Ouvrir la caméra sélectionnée
        log(f"[Start] Opening camera: {self.cameraCombo.currentText()}")
        ok = self._open_camera_selected()
        if not ok:
            QMessageBox.critical(self, "Caméra", "Impossible d’ouvrir une source vidéo (GStreamer/rpicam-vid/V4L2).");
            return

        # Thread d'inférence
        if self.infer_thread: self._stop_infer_thread()
        self.infer_thread = QThread(self);
        self.infer_worker = InferWorker(self.current_adapter, prob_threshold=self.threshSpin.value())
        self.infer_worker.moveToThread(self.infer_thread)
        self.infer_thread.started.connect(self.infer_worker.run)
        self.infer_worker.resultReady.connect(self._on_infer_result)
        self.infer_thread.start()

        # Dossiers & session
        outdir = Path(self.dirEdit.text().strip() or self.output_dir);
        outdir.mkdir(parents=True, exist_ok=True);
        self.output_dir = outdir
        if not self.sessionEdit.text().strip(): self.sessionEdit.setText(self._build_default_session_name(model_path))
        self.summary_rows.clear()
        self.session_started_at = datetime.datetime.now()
        self.meta = {"model_path": model_path, "tasks": self.tasks_order.copy(),
                     "start_time": self.session_started_at.isoformat(timespec="seconds")}
        self.lat_ema_s = None;
        self.fps_hist.clear()

        # Timer
        self.target_fps = self.fpsSpin.value()
        self.timer.start(int(1000 / max(1, self.target_fps)))
        # boutons
        self.startBtn.setEnabled(False);
        self.stopBtn.setEnabled(True)
        self.startBtnDesk.setEnabled(False);
        self.stopBtnDesk.setEnabled(True)
        log(f"[Start] Display timer @ {self.target_fps} FPS | backend={self.cap_backend}")

    def stop_session(self):
        log("[Stop] Stopping session")
        self.timer.stop();
        self._release_camera()
        if self.video_writer: self.video_writer.release(); self.video_writer = None; self.recording_video = False
        if self.recBtn.isChecked(): self.recBtn.setChecked(False)
        self._stop_infer_thread()
        if self.session_started_at is not None:
            self._write_summary_file();
            self.session_started_at = None
        self.startBtn.setEnabled(True);
        self.stopBtn.setEnabled(False)
        self.startBtnDesk.setEnabled(True);
        self.stopBtnDesk.setEnabled(False)

    def _stop_infer_thread(self):
        if self.infer_worker: self.infer_worker.stop()
        if self.infer_thread:
            self.infer_thread.quit();
            self.infer_thread.wait()
        self.infer_worker = None;
        self.infer_thread = None

        # ── Capture & affichage ──

    def _on_timer_capture(self):
        now = time.time();
        dt = max(1e-6, now - self._last_disp_t);
        self._last_disp_t = now
        fps_disp = 1.0 / dt;
        self._disp_fps_ema = fps_disp if self._disp_fps_ema is None else (0.15 * fps_disp + 0.85 * self._disp_fps_ema)

        if self.playback_mode:
            ok, frame = self.cap.read()
            if not ok or frame is None: self._stop_playback(); return
            disp = frame.copy()
        else:
            self._tick += 1
            frame = None;
            ok = False
            if self.cap_backend == "lc-mjpeg" and self.lc_mjpeg is not None:
                ok, frame = self.lc_mjpeg.read()
            elif self.cap is not None:
                ok, frame = self.cap.read()
            if not ok or frame is None: self._reopen_camera_if_needed(); return

            if self.infer_worker and (self._tick % max(1, self.infer_stride) == 0):
                # keep prob threshold in worker synced with UI
                try:
                    self.infer_worker.prob_threshold = self.threshSpin.value()
                except Exception:
                    pass
                self.infer_worker.submit(frame)

            disp = frame.copy()
            if self.last_items: disp = draw_predictions_panel(disp, self.last_items, location="tr")
            if self.speedCheck.isChecked():
                speed_txt = self.last_speed_text or ""
                disp_txt = f"{self._disp_fps_ema:.1f} FPS affichage"
                if speed_txt: disp_txt = speed_txt + f"  |  {disp_txt}"
                disp = draw_bottom_right(disp, disp_txt, alpha=0.55)

        if self.recording_video:
            h, w = disp.shape[:2]
            if self.video_writer is None:
                save_name = self.sessionEdit.text().strip() or self._build_default_session_name(
                    self.meta.get("model_path", "session"))
                out_path = self.output_dir / f"{save_name}.avi"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.video_writer = cv2.VideoWriter(str(out_path), fourcc, float(self.target_fps), (w, h))
            if self.video_writer: self.video_writer.write(disp)

        self._set_pixmap(self.videoLabel, disp)

    @Slot(np.ndarray, dict, float)
    def _on_infer_result(self, frame_bgr: np.ndarray, outputs: dict, elapsed: float):
        items = [];
        per_task = {}
        for task in self.tasks_order:
            if task not in outputs: continue
            logits = outputs[task]
            logits_np = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
            vec = logits_np[0] if logits_np.ndim > 1 else logits_np
            probs = softmax_np(vec)
            idx = int(np.argmax(probs));
            score = float(probs[idx])
            labels = self.tasks.get(task, [f"{task}_{i}" for i in range(len(probs))])
            label = labels[idx] if (idx < len(labels) and score >= self.threshSpin.value()) else "Unknown"
            items.append((task, label, score));
            per_task[task] = {"label": label, "score": score}
        self.last_items = items

        self.lat_ema_s = elapsed if self.lat_ema_s is None else (0.2 * elapsed + 0.8 * self.lat_ema_s)
        inst_fps = 1.0 / max(elapsed, 1e-6);
        self.fps_hist.append(inst_fps)
        avg_fps = sum(self.fps_hist) / len(self.fps_hist)
        self.last_speed_text = f"{self.lat_ema_s * 1000:.0f} ms IA  |  {avg_fps:.1f} FPS IA"

        now = datetime.datetime.now();
        ts_ms = int(now.timestamp() * 1000)
        row = {"timestamp_ms": ts_ms, "iso_time": now.isoformat(timespec="milliseconds"),
               "latency_s": round(elapsed, 4), "camera": self.cameraCombo.currentText(),
               "model": Path(self.meta.get("model_path", "")).name}
        for t, info in per_task.items():
            row[f"{t}_label"] = info["label"];
            row[f"{t}_score"] = round(float(info["score"]), 4)
        self.summary_rows.append(row)

    # ── Vidéo on/off ──
    def toggle_video_record(self, checked: bool):
        if self.playback_mode and checked:
            QMessageBox.information(self, "Lecture", "Enregistrement indisponible pendant la lecture d’un fichier.")
            if self.recBtn.isChecked(): self.recBtn.setChecked(False)
            return
        self.recording_video = checked
        if not checked and self.video_writer is not None:
            self.video_writer.release();
            self.video_writer = None

    # ── Lecture d’enregistrements ──
    def open_recording_file(self):
        if self.cap or self.infer_thread: self.stop_session()
        vid, _ = QFileDialog.getOpenFileName(self, "Ouvrir un enregistrement vidéo", str(self.output_dir),
                                             "Vidéos (*.avi *.mp4 *.mkv *.mov)")
        if not vid: return
        self.play_source_path = Path(vid)
        self.cap = cv2.VideoCapture(str(self.play_source_path))
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Lecture", f"Impossible d’ouvrir {vid}")
            self.cap = None;
            self.play_source_path = None;
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS);
        self.play_fps = fps if fps and fps > 1e-3 else 25.0
        self.playback_mode = True
        self.timer.start(int(1000 / self.play_fps))
        self.startBtn.setEnabled(False);
        self.stopBtn.setEnabled(True);
        self.recBtn.setEnabled(False)
        self.startBtnDesk.setEnabled(False);
        self.stopBtnDesk.setEnabled(True);
        self.recBtnDesk.setEnabled(False)

    def _stop_playback(self):
        self.playback_mode = False
        if self.cap: self.cap.release(); self.cap = None
        self.timer.stop();
        self.play_source_path = None
        self.recBtn.setEnabled(True);
        self.startBtn.setEnabled(True);
        self.stopBtn.setEnabled(False)
        self.recBtnDesk.setEnabled(True);
        self.startBtnDesk.setEnabled(True);
        self.stopBtnDesk.setEnabled(False)

    # ── Transfert & E-mail ──
    def transfer_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choisir des fichiers à transférer",
                                                str(self.output_dir), "Tous (*.*)")
        if not files: return
        dest = QFileDialog.getExistingDirectory(self, "Choisir le dossier de destination")
        if not dest: return
        ok_count = 0
        for f in files:
            try:
                shutil.copy2(f, dest); ok_count += 1
            except Exception as e:
                print(f"Transfert échoué pour {f}: {e}")
        QMessageBox.information(self, "Transfert", f"{ok_count} fichier(s) copié(s) vers\n{dest}")

    def _collect_latest_session_files(self) -> List[Path]:
        base = self.sessionEdit.text().strip()
        if not base: return []
        outs: List[Path] = []
        meta = self.output_dir / f"{base}.meta.json"
        if meta.exists(): outs.append(meta)
        ext = self.formatCombo.currentText()
        path = self.output_dir / f"{base}.{'xlsx' if ext == 'xlsx' else ext}"
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
            if ok:
                QMessageBox.information(self, "E-mail", msg)
            else:
                QMessageBox.critical(self, "E-mail", msg)

    # ── Helpers ──
    def _set_pixmap(self, label: QLabel, frame_bgr: np.ndarray):
        h, w, ch = frame_bgr.shape
        if hasattr(QImage.Format, "Format_BGR888"):
            qimg = QImage(frame_bgr.data, w, h, ch * w, QImage.Format.Format_BGR888)
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        fast = os.getenv("FAST_SCALE", "1") == "1"
        transform_mode = Qt.FastTransformation if fast else Qt.SmoothTransformation
        pix = QPixmap.fromImage(qimg).scaled(label.size(), Qt.KeepAspectRatio, transform_mode)
        label.setPixmap(pix)

    def closeEvent(self, e):
        try:
            if self.playback_mode:
                self._stop_playback()
            else:
                self.stop_session()
        except Exception:
            pass
        super().closeEvent(e)

    def _write_summary_file(self):
        end_time = datetime.datetime.now()
        self.meta["end_time"] = end_time.isoformat(timespec="seconds")
        start_dt = datetime.datetime.fromisoformat(self.meta["start_time"])
        self.meta["duration_s"] = (end_time - start_dt).total_seconds()
        name = self.sessionEdit.text().strip() or self._build_default_session_name(
            self.meta.get("model_path", "session"))
        fmt = self.formatCombo.currentText()
        base = self.output_dir / f"{name}"
        meta_path = base.with_suffix(".meta.json")
        meta_obj = {"meta": self.meta, "summary_count": len(self.summary_rows)}
        meta_path.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")
        if fmt == "json":
            out = {"meta": self.meta, "frames": self.summary_rows}
            (base.with_suffix(".json")).write_text(json.dumps(out, indent=2), encoding="utf-8")
        elif fmt == "csv":
            if pd is not None:
                df = pd.DataFrame(self.summary_rows);
                df.to_csv(base.with_suffix(".csv"), index=False)
            else:
                write_csv_fallback(base.with_suffix(".csv"), self.summary_rows)
        elif fmt == "xlsx":
            if pd is not None:
                df = pd.DataFrame(self.summary_rows)
                try:
                    df.to_excel(base.with_suffix(".xlsx"), index=False)
                except Exception:
                    df.to_csv(base.with_suffix(".csv"), index=False)
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


# ───────── Utils overlay & I/O ─────────
def softmax_np(x):
    x = np.asarray(x, dtype=np.float32);
    x = x - np.max(x);
    e = np.exp(x);
    s = e.sum()
    return e / (s if s > 0 else 1.0)


def _auto_font_scale_for_h(h: int, base: float = 0.58) -> float:
    """
    Échelle de police un peu plus grande qu'avant.
    - base ↑ (0.58 vs 0.50)
    - plafond ↑ (1.05 vs 0.80)
    """
    sc = base * math.sqrt(max(h, 240) / 720.0) + 0.06
    return float(max(0.55, min(1.05, sc)))



def draw_predictions_panel(img: np.ndarray, items: List[Tuple[str, str, float]], location: str = "tr") -> np.ndarray:
    """
    Affiche un panneau (semi-transparent) avec colonnes alignées :
      [ Tâche ]  [ Classe ]  [ █████████████  92% ]
    - Colonnes alignées par sous-colonne (largeur max par colonne)
    - 1 colonne jusqu'à 8 lignes, sinon 2 colonnes
    - Barres de proba à largeur fixe par colonne (cohérentes visuellement)
    """
    if not items: return img
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Texte un peu plus généreux qu'avant
    scale = _auto_font_scale_for_h(h, base=0.62)
    txt_th = max(1, int(round(1.0 * scale)))

    # Marges & espacements
    pad_x, pad_y = 14, 12
    inter_col = 30
    gap_task_label = 10
    gap_label_bar = 14

    n = len(items)
    cols = 1 if n <= 8 else 2
    rows = (n + cols - 1) // cols

    # Répartir les lignes par colonne
    cols_items: List[List[Tuple[str, str, float]]] = []
    for c in range(cols):
        cols_items.append(items[c * rows:(c + 1) * rows])

    # Mesures par colonne (max largeur Tâche / Classe, hauteur de ligne, largeur de barre)
    col_metrics = []
    for lst in cols_items:
        max_task_w = 0
        max_label_w = 0
        line_text_h = 0
        tmp_sizes = []
        for (task, label, score) in lst:
            (tw, th_task), _ = cv2.getTextSize(task + ":", font, scale, txt_th)
            (lw, th_label), _ = cv2.getTextSize(label, font, scale, txt_th)
            max_task_w = max(max_task_w, tw)
            max_label_w = max(max_label_w, lw)
            line_text_h = max(line_text_h, th_task, th_label)
            tmp_sizes.append((tw, lw, line_text_h))
        # Largeur de barre stable par colonne
        bar_w = int(min(max(140, w * 0.22), 260))
        col_metrics.append({
            "max_task": max_task_w,
            "max_label": max_label_w,
            "line_h": line_text_h,
            "bar_w": bar_w,
            "sizes": tmp_sizes
        })

    # Taille panneau
    line_h = max([m["line_h"] for m in col_metrics] + [int(18 * scale)]) + 10
    col_widths = []
    for m in col_metrics:
        col_w = m["max_task"] + gap_task_label + m["max_label"] + gap_label_bar + m["bar_w"]
        col_widths.append(col_w)
    panel_w = pad_x * 2 + sum(col_widths) + inter_col * (cols - 1)
    panel_h = pad_y * 2 + rows * line_h

    # Position
    margin = 10
    if location == "tr":
        x1 = w - panel_w - margin; y1 = margin
    elif location == "tl":
        x1 = margin; y1 = margin
    elif location == "br":
        x1 = w - panel_w - margin; y1 = h - panel_h - margin
    else:
        x1 = margin; y1 = h - panel_h - margin
    x2 = x1 + panel_w; y2 = y1 + panel_h

    # Fond semi-transparent
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    # Couleurs
    grey = (200, 200, 200)
    white = (255, 255, 255)
    green = (60, 200, 60)
    bar_bg = (40, 40, 40)
    bar_bd = (90, 90, 90)
    pct_col = (240, 240, 240)

    # Dessin par colonne
    cx = x1 + pad_x
    for col_i, lst in enumerate(cols_items):
        m = col_metrics[col_i]
        y = y1 + pad_y + int(m["line_h"])
        for (task, label, score) in lst:
            # 1) Tâche (alignée à la largeur max de la colonne)
            ttxt = task + ":"
            cv2.putText(out, ttxt, (cx, y), font, scale, grey, txt_th, cv2.LINE_AA)

            # 2) Classe (alignée)
            lx = cx + m["max_task"] + gap_task_label
            cv2.putText(out, label, (lx, y), font, scale, white, txt_th, cv2.LINE_AA)

            # 3) Barre de proba alignée
            bx = lx + m["max_label"] + gap_label_bar
            # Hauteur de barre proportionnelle à la taille de texte, mais pas trop fine
            bh = max(8, int(m["line_h"] * 0.55))
            by = y - int(m["line_h"] * 0.75)
            bw = m["bar_w"]
            # fond + bord
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), bar_bg, -1)
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), bar_bd, 1)
            # remplissage
            fill = int(bw * float(max(0.0, min(1.0, score))))
            if fill > 0:
                cv2.rectangle(out, (bx, by), (bx + fill, by + bh), green, -1)
            # % à droite de la barre
            ptxt = f"{score * 100:.0f}%"
            (pw, ph), _ = cv2.getTextSize(ptxt, font, scale * 0.9, txt_th)
            px = bx + bw - pw - 4
            py = by + bh - 2
            cv2.putText(out, ptxt, (px, py), font, scale * 0.9, pct_col, txt_th, cv2.LINE_AA)

            y += line_h

        cx += col_widths[col_i] + inter_col

    return out



def draw_bottom_right(img: np.ndarray, text: str, alpha: float = 0.6) -> np.ndarray:
    out = img.copy();
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX;
    scale = _auto_font_scale_for_h(h, base=0.45)
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1);
    pad = 10
    x2, y2 = w - 10, h - 10;
    x1, y1 = x2 - (tw + 2 * pad), y2 - (th + 2 * pad)
    overlay = out.copy();
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1);
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    cv2.putText(out, text, (x1 + pad, y2 - pad - 2), font, scale, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(out, text, (x1 + pad, y2 - pad - 2), font, scale, (60, 200, 60), 1, cv2.LINE_AA)
    return out


def write_csv_fallback(path: Path, rows: List[Dict[str, Any]]):
    if not rows: Path(path).write_text("", encoding="utf-8"); return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = [str(r.get(k, "")) for k in keys]
            f.write(",".join(vals) + "\n")


# ───────── Dialogue e-mail (SMTP) ─────────
class EmailDialog(QDialog):
    def __init__(self, parent=None, default_attachments: Optional[List[Path]] = None):
        super().__init__(parent)
        self.setWindowTitle("Envoyer par e-mail");
        self.resize(520, 360)
        self.attachments: List[Path] = list(default_attachments or [])
        self.toEdit = QLineEdit()
        self.subjEdit = QLineEdit("Fichiers de détection")
        self.bodyEdit = QTextEdit("Veuillez trouver ci-joint les fichiers de détection.")
        self.attLabel = QLabel(self._att_text())
        addBtn = QPushButton("Ajouter pièces jointes…");
        addBtn.clicked.connect(self._add_attachments)
        clearBtn = QPushButton("Vider la liste");
        clearBtn.clicked.connect(self._clear_attachments)
        form = QFormLayout()
        form.addRow("À (séparés par ,) :", self.toEdit)
        form.addRow("Sujet :", self.subjEdit)
        form.addRow("Message :", self.bodyEdit)
        attRow = QHBoxLayout();
        attRow.addWidget(addBtn);
        attRow.addWidget(clearBtn);
        attRow.addStretch()
        v = QVBoxLayout(self)
        v.addLayout(form);
        v.addWidget(QLabel("Pièces jointes :"));
        v.addWidget(self.attLabel);
        v.addLayout(attRow)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept);
        self.buttons.rejected.connect(self.reject);
        v.addWidget(self.buttons)

    def _att_text(self) -> str:
        return "(aucune)" if not self.attachments else "\n".join(str(p) for p in self.attachments)

    def _add_attachments(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choisir des fichiers", "", "Tous (*.*)")
        if files:
            for f in files:
                p = Path(f)
                if p.exists(): self.attachments.append(p)
        self.attLabel.setText(self._att_text())

    def _clear_attachments(self):
        self.attachments.clear();
        self.attLabel.setText(self._att_text())

    def send_email(self) -> Tuple[bool, str]:
        try:
            server = DEFAULT_SMTP_SERVER;
            port = DEFAULT_SMTP_PORT
            user = DEFAULT_EMAIL_USER;
            pwd = DEFAULT_EMAIL_PASS;
            sender = DEFAULT_EMAIL_FROM
            to_list = [t.strip() for t in self.toEdit.text().split(",") if t.strip()]
            subject = self.subjEdit.text().strip();
            body = self.bodyEdit.toPlainText()
            if not (server and port and sender and to_list and user and pwd):
                return False, "Configuration e-mail incomplète (vérifiez SMTP_USER / SMTP_PASSWORD)."
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.mime.base import MIMEBase
            from email import encoders
            msg = MIMEMultipart();
            msg["From"] = sender;
            msg["To"] = ", ".join(to_list);
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))
            for att in self.attachments:
                part = MIMEBase("application", "octet-stream")
                with open(att, "rb") as f: part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f'attachment; filename="{att.name}"')
                msg.attach(part)
            s = smtplib.SMTP_SSL(server, port, timeout=30);
            s.ehlo();
            s.login(user, pwd)
            s.sendmail(sender, to_list, msg.as_string());
            s.quit()
            return True, "E-mail envoyé."
        except Exception as e:
            return False, f"Échec envoi : {e}"


# ───────── Main ─────────
def main():
    app = QApplication(sys.argv)
    win = MainWindow();
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
