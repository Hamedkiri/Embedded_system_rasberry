#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI temps réel Raspberry Pi — Détection météo
- PySide6 (Qt) pour l'IHM (responsive, HiDPI, multi-écrans)
- OpenCV pour capture/overlay/vidéo (GStreamer si dispo)
- Alternatives caméra :
    1) GStreamer (libcamerasrc)
    2) rpicam-vid / libcamera-vid (sous-processus MJPEG pipe)
    3) V4L2 (webcam USB) avec essais MJPG puis YUYV
- ModelRegistry (PyTorch / TFLite / ONNX / PatchGAN)
- Lecture d’enregistrements, transfert, e-mail

Dépendances conseillées (OS) :
  sudo apt install -y libcamera-apps gstreamer1.0-tools gstreamer1.0-libav \
     gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
     gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libcamera v4l-utils

Dépendances Python (exemples) :
  pip install PySide6 opencv-python torch torchvision pandas openpyxl onnxruntime
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

# ─────────────────────────────────────────────────────────────
# Helpers logs horodatés
# ─────────────────────────────────────────────────────────────
def _ts():
    return datetime.datetime.now().strftime("[%H:%M:%S.%f]")[:-3]
def log(msg):
    print(_ts(), msg, flush=True)

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
if videos:
    log(f"Available /dev/video*: {videos}")

# ─────────────────────────────────────────────────────────────
#                 Config e-mail (Gmail unique)
# ─────────────────────────────────────────────────────────────
DEFAULT_SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
DEFAULT_SMTP_PORT   = int(os.getenv("SMTP_PORT", "465"))
DEFAULT_EMAIL_USER  = os.getenv("SMTP_USER",  "votre@gmail.com")
DEFAULT_EMAIL_PASS  = os.getenv("SMTP_PASSWORD", "mot_de_passe_application")
DEFAULT_EMAIL_FROM  = os.getenv("SMTP_FROM", DEFAULT_EMAIL_USER)

# ─────────────────────────────────────────────────────────────
#                     Imports optionnels
# ─────────────────────────────────────────────────────────────
try:
    import pandas as pd
except Exception:
    pd = None

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

from PySide6.QtCore import Qt, QTimer, QObject, Signal, Slot, QThread, QSize
from PySide6.QtGui  import QImage, QPixmap, QFont, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QLineEdit,
    QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QMessageBox, QSizePolicy, QDialog, QFormLayout,
    QDialogButtonBox, QTextEdit
)

# ─────────────────────────────────────────────────────────────
#                   Pré-traitements images
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
#               Adapters / Registry (chargement modèle)
# ─────────────────────────────────────────────────────────────
class BaseAdapter:
    def __init__(self, tasks: Dict[str, List[str]], device: str = "cpu", input_size: int = 224, preprocess: str = "imagenet"):
        self.tasks = tasks
        self.device = device
        self.input_size = int(input_size)
        self.preprocess = preprocess

    def _prep(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.preprocess == "tanh":
            return preprocess_bgr_tanh(frame_bgr, self.input_size)
        return preprocess_bgr_imagenet(frame_bgr, self.input_size)

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        raise NotImplementedError

class PyTorchMultiTaskAdapter(BaseAdapter):
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224, extra: Optional[Dict[str, Any]]=None, preprocess: str="imagenet"):
        super().__init__(tasks, device, input_size, preprocess)
        self._jit = None
        self._model = None
        try:
            self._jit = torch.jit.load(weights_path, map_location=device)
            self._jit.eval()
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
                    base_encoder=base_encoder,
                    truncate_after_layer=truncate,
                    tasks=tasks,
                    device=device,
                    use_attention=use_attention,
                    attn_token_dim=attn_token_dim,
                    cls_hidden_dims=cls_hidden_dims,
                    cls_num_layers=cls_num_layers
                )
                load_best_model(self._model, weights_path, strict_backbone=False, verbose=False)
                self._model.eval()
                log("[Model] Loaded PyTorch state dict")
            except Exception as e:
                raise RuntimeError(f"Impossible de charger le modèle PyTorch: {e}")

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = self._prep(frame_bgr)
        nchw = np.transpose(nhwc, (2,0,1))[np.newaxis, ...].astype(np.float32)
        inp  = torch.from_numpy(nchw)
        with torch.no_grad():
            if self._jit is not None:
                return self._jit(inp)
            else:
                return self._model(inp)

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
        log("[Model] Loaded TFLite")

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = self._prep(frame_bgr)
        arr = nhwc[np.newaxis, ...].astype(np.float32)
        inp = self.input_details[0]
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
        log("[Model] Loaded ONNX")

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
        log("[Model] Loaded PatchGAN")

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = self._prep(frame_bgr)
        nchw = np.transpose(nhwc, (2,0,1))[np.newaxis, ...].astype(np.float32)
        x = torch.from_numpy(nchw)
        with torch.no_grad():
            out = self.model(x)
        return out

class ModelRegistry:
    _MAP = {"pytorch":PyTorchMultiTaskAdapter,"tflite":TFLiteMultiHeadAdapter,"onnx":ONNXMultiHeadAdapter,"patchgan":PatchGANAdapter}
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
        for k in ("patch_size","attn_tau","attn_use_se","ndf","discriminator"):
            if k in man or k in man.get("extra", {}): return True
        return False
    @staticmethod
    def _read_config(model_path: str) -> Dict[str, Any]:
        p = Path(model_path); base = p.parent
        cfg_path = base / "config.json"
        if cfg_path.exists():
            try:
                return json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                pass
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
                if cfg["tasks"] and isinstance(list(cfg["tasks"].values())[0], int):
                    cfg["tasks"] = ModelRegistry._labels_from_tasks(cfg["tasks"])
                return cfg
            except Exception:
                pass
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
        if adapter == "patchgan":
            return PatchGANAdapter(model_path, device=device)
        if mtype not in cls._MAP:
            raise ValueError(f"Type inconnu pour {model_path}. Ajoutez 'type' dans config.json (pytorch|tflite|onnx).")
        if not tasks:
            tasks = cfg.get("tasks") or {"Weather": ["No", "Yes"]}
        return cls._MAP[mtype](weights_path=model_path,tasks=tasks,device=device,input_size=input_size,extra=merged_extra,preprocess=preprocess)

# ─────────────────────────────────────────────────────────────
#         Backend alternatif : rpicam-vid / libcamera-vid
# ─────────────────────────────────────────────────────────────
class LibcameraMJPEGCapture:
    """
    Lance rpicam-vid/libcamera-vid en MJPEG sur stdout :
      <bin> -t 0 --nopreview --codec mjpeg --width WxH --height H --framerate F -o -
    Décode les JPEG en BGR via cv2.imdecode.
    """
    def __init__(self, width=1280, height=720, fps=30):
        self.width = int(width); self.height = int(height); self.fps = int(fps)
        self.proc: Optional[subprocess.Popen] = None
        self._buf = bytearray()
        self._running = False
        self._th: Optional[threading.Thread] = None
        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self.bin = None

    def _pick_bin(self) -> Optional[str]:
        env = os.getenv("PI_CAM_BIN")
        cand = [env] if env else ["rpicam-vid", "libcamera-vid"]
        for b in cand:
            if b and _shutil.which(b):
                return b
        return None

    def start(self) -> bool:
        self.bin = self._pick_bin()
        if not self.bin:
            log("[LCMJPEG] Neither rpicam-vid nor libcamera-vid was found in PATH")
            return False
        cmd = [
            self.bin, "-t", "0",
            "--nopreview",            # compatible rpicam-vid
            "--codec", "mjpeg",
            "--width", str(self.width), "--height", str(self.height),
            "--framerate", str(self.fps),
            "-o", "-"
        ]
        try:
            log(f"[LCMJPEG] Start: {' '.join(cmd)}")
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        except Exception as e:
            log(f"[LCMJPEG] Launch failed: {e}")
            self.proc = None
            return False
        self._running = True
        self._th = threading.Thread(target=self._reader_loop, daemon=True)
        self._th.start()
        return True

    def _reader_loop(self):
        SOI = b"\xff\xd8"; EOI = b"\xff\xd9"
        stdout = self.proc.stdout if self.proc else None
        while self._running and self.proc and stdout:
            try:
                chunk = stdout.read(4096)
                if not chunk:
                    time.sleep(0.005)
                    if self.proc.poll() is not None:
                        log("[LCMJPEG] Process ended")
                        break
                    continue
                self._buf.extend(chunk)
                while True:
                    soi = self._buf.find(SOI)
                    if soi < 0:
                        if len(self._buf) > 1_000_000:
                            self._buf[:] = self._buf[-200_000:]
                        break
                    eoi = self._buf.find(EOI, soi+2)
                    if eoi < 0:
                        if soi > 0:
                            del self._buf[:soi]
                        break
                    eoi += 2
                    jpg = self._buf[soi:eoi]
                    del self._buf[:eoi]
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        try:
                            if self.q.full():
                                _ = self.q.get_nowait()
                        except Exception:
                            pass
                        self.q.put_nowait(frame)
            except Exception as e:
                log(f"[LCMJPEG] Reader error: {e}")
                time.sleep(0.02)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frame = self.q.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def release(self):
        self._running = False
        if self._th:
            self._th.join(timeout=0.5)
        if self.proc:
            try:
                if self.proc.poll() is None:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=0.3)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                if self.proc.stdout: self.proc.stdout.close()
                if self.proc.stderr: self.proc.stderr.close()
            except Exception:
                pass
        self.proc = None
        self._buf.clear()
        with self.q.mutex:
            self.q.queue.clear()
        log("[LCMJPEG] Released")

# ─────────────────────────────────────────────────────────────
#                  Backends caméra: GST / LCMJPEG / V4L2
# ─────────────────────────────────────────────────────────────
def cv_has_gstreamer() -> bool:
    try:
        bi = cv2.getBuildInformation()
    except Exception:
        return False
    for line in bi.splitlines():
        if "GStreamer" in line and "YES" in line:
            return True
    return False

def default_gst_pipeline() -> str:
    w = int(os.getenv("PI_CAM_WIDTH", "1280"))
    h = int(os.getenv("PI_CAM_HEIGHT", "720"))
    fps = int(os.getenv("PI_CAM_FPS", "30"))
    pipe_env = os.getenv("PI_GST_PIPELINE")
    if pipe_env:
        return pipe_env
    return (
        f"libcamerasrc ! video/x-raw,width={w},height={h},framerate={fps}/1,format=BGRx "
        f"! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
    )
# ─────────────────────────────────────────────────────────────
#                 Thread léger d’inférence (non bloquant)
# ─────────────────────────────────────────────────────────────
class InferWorker(QObject):
    # frame_bgr (np.ndarray), outputs (dict[str, torch.Tensor|np.ndarray]), elapsed_sec (float)
    resultReady = Signal(np.ndarray, dict, float)

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
                _ = self.queue.get_nowait()  # drop frame si on est en retard
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
        self.setMinimumSize(1100, 640)

        # État caméra/backends
        self.cap = None
        self.cap_backend = None  # "gst" | "lc-mjpeg" | "v4l2"
        self.lc_mjpeg: Optional[LibcameraMJPEGCapture] = None
        self._read_fail = 0
        self._tick = 0

        # Timer capture
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_capture)
        self.target_fps = 20

        # Inference
        self.infer_thread = None
        self.infer_worker = None
        self.current_adapter: Optional[BaseAdapter] = None
        self.tasks: Dict[str, List[str]] = {}
        self.tasks_order: List[str] = []
        self.device = "cpu"

        # Session
        self.recording_video = False
        self.video_writer = None
        self.output_dir = Path("runs"); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_rows: List[Dict[str, Any]] = []
        self.meta = {}
        self.session_started_at: Optional[datetime.datetime] = None

        # Lecture
        self.playback_mode = False
        self.play_source_path: Optional[Path] = None
        self.play_fps = 25.0

        # Mesures vitesse
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
        self.cameraCombo = QComboBox(); self.cameraCombo.addItems([f"Caméra {i}" for i in range(6)])
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
        log(f"[UI] Screen changed. DPI: {dpi} scale: {scale}")
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
        self.modelCombo.clear()
        models_dir = Path("models").resolve(); models_dir.mkdir(exist_ok=True)
        patterns = ("**/*.pt", "**/*.pth", "**/*.tflite", "**/*.onnx")
        files: List[str] = []
        for pat in patterns:
            files += glob.glob(str(models_dir / pat), recursive=True)
        files = sorted(files)
        log(f"[Models] Found: {len(files)}")
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

    # ---------- V4L2 helper ----------
    def _try_open_v4l2(self, cam_index: int, w: int, h: int, fps: int):
        trials = [("MJPG", cv2.VideoWriter_fourcc(*"MJPG")), ("YUYV", cv2.VideoWriter_fourcc(*"YUYV")), ("AUTO", None)]
        for label, fourcc in trials:
            cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
            if not cap or not cap.isOpened():
                continue
            if fourcc is not None:
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ok, frame = cap.read()
            if ok and frame is not None:
                log(f"[Camera] V4L2 OK with fourcc={label}")
                return cap
            else:
                log(f"[Camera] V4L2 read failed with fourcc={label}; retry next mode")
                cap.release()
        return None

    # ---------- Camera open/close ----------
    def _open_camera(self, cam_index: int) -> bool:
        """Essaie GST → rpicam/libcamera-vid (MJPEG) → V4L2. Retourne True si OK."""
        self._release_camera()

        # Choix forcé optionnel
        force = (os.getenv("PI_FORCE_BACKEND") or "").lower()
        order = ["gst", "lc-mjpeg", "v4l2"]
        if force in order:
            order = [force] + [b for b in order if b != force]
            log(f"[Camera] Forcing backend preference: {order}")

        w = int(os.getenv("PI_CAM_WIDTH", "1280"))
        h = int(os.getenv("PI_CAM_HEIGHT", "720"))
        fps = int(os.getenv("PI_CAM_FPS", "30"))

        for backend in order:
            if backend == "gst":
                if cv_has_gstreamer():
                    pipe = default_gst_pipeline()
                    log(f"[Camera] Try GStreamer:\n{pipe}")
                    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                    if cap is not None and cap.isOpened():
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            self.cap = cap; self.cap_backend = "gst"
                            log("[Camera] GStreamer OK")
                            return True
                        log("[Camera] GStreamer opened but first read failed")
                        cap.release()
                else:
                    log("[Camera] OpenCV built without GStreamer")

            elif backend == "lc-mjpeg":
                self.lc_mjpeg = LibcameraMJPEGCapture(w, h, fps)
                if self.lc_mjpeg.start():
                    t0 = time.time()
                    ok, frame = False, None
                    while time.time() - t0 < 1.2 and not ok:
                        ok, frame = self.lc_mjpeg.read()
                        if not ok:
                            time.sleep(0.02)
                    if ok and frame is not None:
                        self.cap_backend = "lc-mjpeg"
                        log("[Camera] rpicam-vid/libcamera-vid MJPEG OK")
                        return True
                    else:
                        log("[Camera] libcamera-vid started but no frame")
                        self.lc_mjpeg.release(); self.lc_mjpeg = None
                else:
                    log("[Camera] rpicam-vid/libcamera-vid backend not available")

            elif backend == "v4l2":
                log(f"[Camera] Try V4L2 index {cam_index}")
                cap = self._try_open_v4l2(cam_index, w, h, fps)
                if cap:
                    self.cap = cap; self.cap_backend = "v4l2"
                    return True
                else:
                    log("[Camera] V4L2 failed on all tried FOURCCs")

        log("[Camera] All backends failed")
        return False

    def _release_camera(self):
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        if self.lc_mjpeg is not None:
            try: self.lc_mjpeg.release()
            except Exception: pass
            self.lc_mjpeg = None
        self.cap_backend = None
        self._read_fail = 0

    def _reopen_camera_if_needed(self):
        self._read_fail += 1
        if self._read_fail % 20 == 0:
            log(f"[Capture] READ FAIL x{self._read_fail} (tick {self._tick})")
        if self._read_fail >= 60:
            log("[Capture] Too many read fails → reopen camera")
            idx = self.cameraCombo.currentIndex()
            self._open_camera(idx)
            self._read_fail = 0

    # ---------- Start / Stop ----------
    def start_session(self):
        if self.playback_mode: self._stop_playback()

        model_path = self.modelCombo.currentData() or self.modelCombo.currentText()
        model_path = str(model_path).strip()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "Modèle", "Aucun modèle valide sélectionné."); return

        self.tasks = self._load_tasks(model_path)
        self.device = "cuda" if hasattr(torch, "cuda") and getattr(torch.cuda, "is_available", lambda: False)() else "cpu"
        try:
            self.current_adapter = ModelRegistry.create(model_path=model_path,tasks=self.tasks,device=self.device,extra=None)
            if isinstance(self.current_adapter, PatchGANAdapter):
                self.tasks = self.current_adapter.tasks
        except Exception as e:
            QMessageBox.critical(self, "Chargement modèle", str(e)); return

        self.tasks_order = list(self.tasks.keys())
        self.deviceLabel.setText(f"Device: {'GPU' if self.device=='cuda' else 'CPU'}")

        cam_index = self.cameraCombo.currentIndex()
        log(f"[Start] Opening camera index: {cam_index}")
        ok = self._open_camera(cam_index)
        if not ok:
            QMessageBox.critical(self, "Caméra", "Impossible d’ouvrir une source vidéo (GStreamer/rpicam-vid/V4L2).")
            return

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
        self.meta = {"model_path": model_path, "tasks": self.tasks_order.copy(), "start_time": self.session_started_at.isoformat(timespec="seconds")}
        self.lat_ema_s = None; self.fps_hist.clear()

        self.target_fps = self.fpsSpin.value()
        self.timer.start(int(1000 / max(1, self.target_fps)))
        self.startBtn.setEnabled(False); self.stopBtn.setEnabled(True)
        log(f"[Start] Capture timer @ {self.target_fps} FPS | backend={self.cap_backend}")

    def stop_session(self):
        log("[Stop] Stopping session")
        self.timer.stop()
        self._release_camera()
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
        if self.playback_mode:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._stop_playback(); return
            disp = draw_bottom_right(frame.copy(), "Lecture enregistrement", alpha=0.6)
            self._set_pixmap(self.videoLabel, disp); return

        self._tick += 1

        frame = None; ok = False
        if self.cap_backend == "lc-mjpeg" and self.lc_mjpeg is not None:
            ok, frame = self.lc_mjpeg.read()
        elif self.cap is not None:
            ok, frame = self.cap.read()

        if not ok or frame is None:
            self._reopen_camera_if_needed()
            return

        self._read_fail = 0
        if self.infer_worker:
            self.infer_worker.prob_threshold = self.threshSpin.value()
            self.infer_worker.submit(frame)

    @Slot(np.ndarray, dict, float)
    def _on_infer_result(self, frame_bgr: np.ndarray, outputs: dict, elapsed: float):
        items = []
        per_task = {}
        for task in self.tasks_order:
            if task not in outputs:
                continue
            logits = outputs[task]
            logits_np = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
            vec = logits_np[0] if logits_np.ndim > 1 else logits_np
            probs = softmax_np(vec)
            idx = int(np.argmax(probs))
            score = float(probs[idx])
            labels = self.tasks.get(task, [f"{task}_{i}" for i in range(len(probs))])
            label = labels[idx] if (idx < len(labels) and score >= self.threshSpin.value()) else "Unknown"
            items.append((task, label, score))
            per_task[task] = {"label": label, "score": score}

        self.lat_ema_s = elapsed if self.lat_ema_s is None else (0.1*elapsed + 0.9*self.lat_ema_s)
        inst_fps = 1.0 / max(elapsed, 1e-6); self.fps_hist.append(inst_fps)
        avg_fps = sum(self.fps_hist)/len(self.fps_hist)

        disp = draw_predictions_panel(frame_bgr.copy(), items, location="tr")
        if self.speedCheck.isChecked():
            disp = draw_bottom_right(disp, f"{self.lat_ema_s*1000:.1f} ms  |  {avg_fps:.1f} FPS", alpha=0.55)

        if self.recording_video:
            h, w = disp.shape[:2]
            if self.video_writer is None:
                save_name = self.sessionEdit.text().strip() or self._build_default_session_name(self.meta.get("model_path","session"))
                out_path = self.output_dir / f"{save_name}.avi"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.video_writer = cv2.VideoWriter(str(out_path), fourcc, float(self.target_fps), (w, h))
            if self.video_writer: self.video_writer.write(disp)

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

def _auto_font_scale_for_h(h: int, base: float = 0.50) -> float:
    sc = base * math.sqrt(max(h, 240) / 720.0) + 0.05
    return float(max(0.45, min(0.80, sc)))

def draw_predictions_panel(img: np.ndarray, items: List[Tuple[str, str, float]], location: str = "tr") -> np.ndarray:
    if not items:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = _auto_font_scale_for_h(h)
    thick = 1
    lines = [f"{t}: {lbl} ({s:.2f})" for (t, lbl, s) in items]
    n = len(lines)
    cols = 1 if n <= 6 else 2
    rows = int(math.ceil(n / cols))
    pad_x, pad_y = 10, 8
    inter_col = 24
    cols_text: List[List[str]] = []
    for c in range(cols):
        start = c * rows
        cols_text.append(lines[start:start+rows])
    col_widths = []
    text_sizes = []
    for c in range(cols):
        c_sizes = []
        m = 0
        for line in cols_text[c]:
            (tw, th), _ = cv2.getTextSize(line, font, scale, thick)
            c_sizes.append((tw, th))
            m = max(m, tw)
        col_widths.append(m); text_sizes.append(c_sizes)
    panel_w = sum(col_widths) + pad_x * 2 + (cols - 1) * inter_col
    line_h = (text_sizes[0][0][1] if text_sizes[0] else int(18 * scale)) + 6
    panel_h = rows * line_h + pad_y * 2
    margin = 10
    if location == "tr":
        x1 = w - panel_w - margin; y1 = margin
    elif location == "tl":
        x1 = margin; y1 = margin
    elif location == "br":
        x1 = w - panel_w - margin; y1 = h - panel_h - margin
    else:
        x1 = margin; y1 = h - panel_h - margin
    x2 = x1 + int(panel_w); y2 = y1 + int(panel_h)
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    col_x = x1 + pad_x
    green = (60, 200, 60); grey = (190, 190, 190)
    for c in range(cols):
        col_lines = cols_text[c]
        y = y1 + pad_y + int(text_sizes[c][0][1] if text_sizes[c] else 16)
        for line in col_lines:
            if ": " in line: task_txt, rest = line.split(": ", 1)
            else: task_txt, rest = line, ""
            cv2.putText(out, task_txt + ":", (col_x, y), font, scale, grey, 1, cv2.LINE_AA)
            (tw_task, _), _ = cv2.getTextSize(task_txt + ":", font, scale, 1)
            cv2.putText(out, " " + rest, (col_x + tw_task, y), font, scale, green, 1, cv2.LINE_AA)
            y += line_h
        col_x += col_widths[c] + inter_col
    return out

def draw_bottom_right(img: np.ndarray, text: str, alpha: float=0.6) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = _auto_font_scale_for_h(h, base=0.45)
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    pad = 10
    x2, y2 = w - 10, h - 10
    x1, y1 = x2 - (tw + 2*pad), y2 - (th + 2*pad)
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,0,0), -1)
    cv2.addWeighted(overlay, alpha, out, 1-alpha, 0, out)
    cv2.putText(out, text, (x1 + pad, y2 - pad - 2), font, scale, (240,240,240), 2, cv2.LINE_AA)
    cv2.putText(out, text, (x1 + pad, y2 - pad - 2), font, scale, (60,200,60), 1, cv2.LINE_AA)
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
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
