# core_app.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORE — Caméra, modèles, inférence, overlay, export, email SMTP
- Ne dépend PAS de la mise en page UI.
- Peut dépendre de Qt (QThread/QImage) pour conversion image et worker.
"""

from __future__ import annotations

import os, json, time, glob, queue, datetime, shutil, ssl, smtplib, math, subprocess, threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

import numpy as np
import cv2

# Limiter threads OpenCV
try:
    cv2.setNumThreads(int(os.getenv("OPENCV_THREADS", "1")))
except Exception:
    pass

# (Optionnel) pandas
try:
    import pandas as pd
except Exception:
    pd = None

# (Optionnel) torch
try:
    import torch
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(int(os.getenv("TORCH_THREADS", "1")))
    import torch.nn as nn
    if not hasattr(nn.ModuleDict, "get"):
        def _md_get(self, key, default=None):
            return self[key] if key in self else default
        nn.ModuleDict.get = _md_get
except Exception:
    class _DummyTorch:
        def no_grad(self): return self
        def __enter__(self): return self
        def __exit__(self, *a): pass
    torch = _DummyTorch()

from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt
from PySide6.QtGui import QImage, QPixmap

# SMTP config (Gmail)
DEFAULT_SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
DEFAULT_SMTP_PORT   = int(os.getenv("SMTP_PORT", "465"))
DEFAULT_EMAIL_USER  = os.getenv("SMTP_USER",  "votre@gmail.com")
DEFAULT_EMAIL_PASS  = os.getenv("SMTP_PASSWORD", "mot_de_passe_application")
DEFAULT_EMAIL_FROM  = os.getenv("SMTP_FROM", DEFAULT_EMAIL_USER)

# ──────────────────────────────────────────────────────────────────────────────
# Preprocess
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# Adapters / ModelRegistry (inchangé sur le fond, déplacé dans le core)
# ──────────────────────────────────────────────────────────────────────────────

class BaseAdapter:
    def __init__(self, tasks: Dict[str, List[str]], device: str = "cpu", input_size: int = 224, preprocess: str = "imagenet"):
        self.tasks = tasks
        self.device = device
        self.input_size = int(input_size)
        self.preprocess = preprocess

    def _prep(self, frame_bgr: np.ndarray) -> np.ndarray:
        return preprocess_bgr_tanh(frame_bgr, self.input_size) if self.preprocess == "tanh" else preprocess_bgr_imagenet(frame_bgr, self.input_size)

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        raise NotImplementedError


class PyTorchMultiTaskAdapter(BaseAdapter):
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224,
                 extra: Optional[Dict[str, Any]] = None, preprocess: str = "imagenet"):
        super().__init__(tasks, device, input_size, preprocess)
        self._jit = None
        self._model = None
        try:
            self._jit = torch.jit.load(weights_path, map_location=device)
            self._jit.eval()
        except Exception:
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

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = self._prep(frame_bgr)
        nchw = np.transpose(nhwc, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        inp = torch.from_numpy(nchw)
        with torch.no_grad():
            return self._jit(inp) if self._jit is not None else self._model(inp)


class TFLiteMultiHeadAdapter(BaseAdapter):
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224,
                 extra: Optional[Dict[str, Any]] = None, preprocess: str = "imagenet"):
        super().__init__(tasks, device, input_size, preprocess)
        extra = extra or {}
        try:
            import tflite_runtime.interpreter as tflite
        except Exception:
            import tensorflow.lite as tflite

        self.interp = tflite.Interpreter(model_path=weights_path, num_threads=int(extra.get("threads", 2)))
        self.interp.allocate_tensors()
        self.input_details = self.interp.get_input_details()
        self.output_details = self.interp.get_output_details()
        self.output_task_order = extra.get("output_task_order", list(tasks.keys()))

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
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224,
                 extra: Optional[Dict[str, Any]] = None, preprocess: str = "imagenet"):
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
        nchw = nhwc.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
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
    def __init__(self, model_path: str, device="cpu", selected_tasks: Optional[List[str]] = None):
        p = Path(model_path)
        manifest = p.parent / "manifest.json"
        if not manifest.exists():
            raise RuntimeError(f"manifest.json introuvable à côté de {model_path}")

        from loader_patchgan_from_manifest import create_patchgan_from_manifest
        model, tasks = create_patchgan_from_manifest(str(manifest), device=device, selected_tasks=selected_tasks)

        man = json.loads(manifest.read_text(encoding="utf-8"))
        extra = man.get("extra", {})
        input_size = int(man.get("input_size", extra.get("input_size", 224)))
        preprocess = str(extra.get("preprocess", "tanh")).lower()

        super().__init__(tasks=tasks, device=device, input_size=input_size, preprocess=preprocess)
        self.model = model.eval()

    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, "torch.Tensor"]:
        nhwc = self._prep(frame_bgr)
        nchw = np.transpose(nhwc, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        x = torch.from_numpy(nchw)
        with torch.no_grad():
            return self.model(x)


class ModelRegistry:
    _MAP = {"pytorch": PyTorchMultiTaskAdapter, "tflite": TFLiteMultiHeadAdapter, "onnx": ONNXMultiHeadAdapter}

    @staticmethod
    def _labels_from_tasks(tasks_dict: Optional[Dict[str, int]]) -> Dict[str, List[str]]:
        if not isinstance(tasks_dict, dict):
            return {"Weather": ["No", "Yes"]}
        out: Dict[str, List[str]] = {}
        for task, n in tasks_dict.items():
            try:
                n = int(n)
            except Exception:
                n = 2
            out[task] = [f"{task}_{i}" for i in range(n)]
        return out

    @staticmethod
    def _looks_like_patchgan(man: Dict[str, Any], path_hint: str) -> bool:
        txt = (json.dumps(man, ensure_ascii=False) + " " + path_hint).lower()
        if "patchgan" in txt:
            return True
        for k in ("patch_size", "attn_tau", "attn_use_se", "ndf", "discriminator"):
            if k in man or k in man.get("extra", {}):
                return True
        return False

    @staticmethod
    def read_config(model_path: str) -> Dict[str, Any]:
        p = Path(model_path)
        base = p.parent

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
    def infer_type_from_ext(model_path: str) -> Optional[str]:
        ext = Path(model_path).suffix.lower()
        return {".pt": "pytorch", ".pth": "pytorch", ".tflite": "tflite", ".onnx": "onnx"}.get(ext)

    @classmethod
    def create(cls, model_path: str, tasks: Optional[Dict[str, List[str]]], device="cpu",
               extra: Optional[Dict[str, Any]] = None, selected_tasks: Optional[List[str]] = None) -> BaseAdapter:
        cfg = cls.read_config(model_path)
        adapter = cfg.get("adapter")
        mtype = cfg.get("type") or cls.infer_type_from_ext(model_path)
        input_size = int(cfg.get("input_size", 224))

        merged_extra = dict(cfg.get("extra", {}))
        if extra:
            merged_extra.update(extra)

        preprocess = merged_extra.get("preprocess", "imagenet")

        # PatchGAN : prune via loader
        if adapter == "patchgan":
            return PatchGANAdapter(model_path, device=device, selected_tasks=selected_tasks)

        # Autres : prune logique via sous-dict des tâches
        if not tasks:
            tasks = cfg.get("tasks") or {"Weather": ["No", "Yes"]}
        if selected_tasks:
            tasks = {k: v for k, v in tasks.items() if k in selected_tasks}

        if mtype not in cls._MAP:
            raise ValueError(f"Type inconnu pour {model_path} (pytorch|tflite|onnx).")
        return cls._MAP[mtype](weights_path=model_path, tasks=tasks, device=device, input_size=input_size, extra=merged_extra, preprocess=preprocess)


# ──────────────────────────────────────────────────────────────────────────────
# Camera backends
# ──────────────────────────────────────────────────────────────────────────────

class LibcameraMJPEGCapture:
    """Backend rpicam-vid/libcamera-vid MJPEG via stdout → parse JPEG."""
    def __init__(self, width=1280, height=720, fps=30):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
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
            if b and shutil.which(b):
                return b
        return None

    def start(self) -> bool:
        self.bin = self._pick_bin()
        if not self.bin:
            return False

        cmd = [
            self.bin, "-t", "0", "--nopreview", "--codec", "mjpeg",
            "--width", str(self.width), "--height", str(self.height),
            "--framerate", str(self.fps), "-o", "-"
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        except Exception:
            self.proc = None
            return False

        self._running = True
        self._th = threading.Thread(target=self._reader_loop, daemon=True)
        self._th.start()
        return True

    def _reader_loop(self):
        SOI = b"\xff\xd8"
        EOI = b"\xff\xd9"
        stdout = self.proc.stdout if self.proc else None

        while self._running and self.proc and stdout:
            try:
                chunk = stdout.read(4096)
                if not chunk:
                    time.sleep(0.005)
                    if self.proc.poll() is not None:
                        break
                    continue

                self._buf.extend(chunk)

                while True:
                    soi = self._buf.find(SOI)
                    if soi < 0:
                        if len(self._buf) > 1_000_000:
                            self._buf[:] = self._buf[-200_000:]
                        break
                    eoi = self._buf.find(EOI, soi + 2)
                    if eoi < 0:
                        if soi > 0:
                            del self._buf[:soi]
                        break

                    eoi += 2
                    jpg = self._buf[soi:eoi]
                    del self._buf[:eoi]

                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    # queue maxsize=1 : garder la frame la plus récente
                    try:
                        if self.q.full():
                            _ = self.q.get_nowait()
                    except Exception:
                        pass
                    try:
                        self.q.put_nowait(frame)
                    except Exception:
                        pass

            except Exception:
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
                if self.proc.stdout:
                    self.proc.stdout.close()
                if self.proc.stderr:
                    self.proc.stderr.close()
            except Exception:
                pass
        self.proc = None
        self._buf.clear()
        with self.q.mutex:
            self.q.queue.clear()


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

def list_v4l2_devices() -> List[Tuple[str, int, str]]:
    devs = sorted(glob.glob("/dev/video*"))
    out: List[Tuple[str, int, str]] = []
    name_by_dev = {}

    if shutil.which("v4l2-ctl"):
        try:
            txt = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True)
            cur = None
            for line in txt.splitlines():
                if not line.strip():
                    continue
                if not line.startswith("\t"):
                    cur = line.strip()
                else:
                    d = line.strip()
                    if d.startswith("/dev/video"):
                        name_by_dev[d] = cur or d
        except Exception:
            pass

    for d in devs:
        try:
            idx = int(Path(d).name.replace("video", ""))
        except Exception:
            continue
        label = f"{name_by_dev.get(d, 'V4L2')} ({d})"
        out.append((label, idx, d))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Inference worker (thread)
# ──────────────────────────────────────────────────────────────────────────────

class InferWorker(QObject):
    """
    Worker d’inférence :
    - queue maxsize=1 → pas de backlog
    - émet resultReady(outputs, elapsed)
    """
    resultReady = Signal(dict, float)
    error = Signal(str)

    def __init__(self, infer_adapter: BaseAdapter):
        super().__init__()
        self.infer = infer_adapter
        self.queue = queue.Queue(maxsize=1)
        self._running = True

    @Slot()
    def run(self):
        while self._running:
            try:
                frame = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                t0 = time.time()
                outputs = self.infer.predict_bgr(frame)
                elapsed = time.time() - t0
                self.resultReady.emit(outputs, elapsed)
            except Exception as e:
                self.error.emit(str(e))

    def submit(self, frame: np.ndarray):
        try:
            if self.queue.full():
                _ = self.queue.get_nowait()
            self.queue.put_nowait(frame)
        except Exception:
            pass

    def stop(self):
        self._running = False


# ──────────────────────────────────────────────────────────────────────────────
# Overlay utils
# ──────────────────────────────────────────────────────────────────────────────

def softmax_np(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / (s if s > 0 else 1.0)

def _auto_font_scale_for_h(h: int, base: float = 0.58) -> float:
    sc = base * math.sqrt(max(h, 240) / 720.0) + 0.06
    return float(max(0.55, min(1.05, sc)))

def draw_predictions_panel(img: np.ndarray, items: List[Tuple[str, str, float]], location: str = "tr") -> np.ndarray:
    if not items:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = _auto_font_scale_for_h(h, base=0.62)
    txt_th = max(1, int(round(1.0 * scale)))

    pad_x, pad_y = 14, 12
    inter_col = 30
    gap_task_label = 10
    gap_label_bar = 14

    n = len(items)
    cols = 1 if n <= 8 else 2
    rows = (n + cols - 1) // cols

    cols_items: List[List[Tuple[str, str, float]]] = []
    for c in range(cols):
        cols_items.append(items[c * rows:(c + 1) * rows])

    col_metrics = []
    for lst in cols_items:
        max_task_w, max_label_w, line_text_h = 0, 0, 0
        for (task, label, score) in lst:
            (tw, th_task), _ = cv2.getTextSize(task + ":", font, scale, txt_th)
            (lw, th_label), _ = cv2.getTextSize(label, font, scale, txt_th)
            max_task_w = max(max_task_w, tw)
            max_label_w = max(max_label_w, lw)
            line_text_h = max(line_text_h, th_task, th_label)
        bar_w = int(min(max(140, w * 0.22), 260))
        col_metrics.append({"max_task": max_task_w, "max_label": max_label_w, "line_h": line_text_h, "bar_w": bar_w})

    line_h = max([m["line_h"] for m in col_metrics] + [int(18 * scale)]) + 10
    col_widths = []
    for m in col_metrics:
        col_w = m["max_task"] + gap_task_label + m["max_label"] + gap_label_bar + m["bar_w"]
        col_widths.append(col_w)

    panel_w = pad_x * 2 + sum(col_widths) + inter_col * (cols - 1)
    panel_h = pad_y * 2 + rows * line_h

    margin = 10
    if location == "tr":
        x1 = w - panel_w - margin
        y1 = margin
    elif location == "tl":
        x1 = margin
        y1 = margin
    elif location == "br":
        x1 = w - panel_w - margin
        y1 = h - panel_h - margin
    else:
        x1 = margin
        y1 = h - panel_h - margin

    x2, y2 = x1 + panel_w, y1 + panel_h

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    grey = (200, 200, 200)
    white = (255, 255, 255)
    green = (60, 200, 60)
    bar_bg = (40, 40, 40)
    bar_bd = (90, 90, 90)
    pct_col = (240, 240, 240)

    cx = x1 + pad_x
    for col_i, lst in enumerate(cols_items):
        m = col_metrics[col_i]
        y = y1 + pad_y + int(m["line_h"])
        for (task, label, score) in lst:
            cv2.putText(out, task + ":", (cx, y), font, scale, grey, txt_th, cv2.LINE_AA)
            lx = cx + m["max_task"] + gap_task_label
            cv2.putText(out, label, (lx, y), font, scale, white, txt_th, cv2.LINE_AA)

            bx = lx + m["max_label"] + gap_label_bar
            bh = max(8, int(m["line_h"] * 0.55))
            by = y - int(m["line_h"] * 0.75)
            bw = m["bar_w"]

            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), bar_bg, -1)
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), bar_bd, 1)

            fill = int(bw * float(max(0.0, min(1.0, score))))
            if fill > 0:
                cv2.rectangle(out, (bx, by), (bx + fill, by + bh), green, -1)

            ptxt = f"{score * 100:.0f}%"
            (pw, _), _ = cv2.getTextSize(ptxt, font, scale * 0.9, txt_th)
            px = bx + bw - pw - 4
            py = by + bh - 2
            cv2.putText(out, ptxt, (px, py), font, scale * 0.9, pct_col, txt_th, cv2.LINE_AA)

            y += line_h

        cx += col_widths[col_i] + inter_col

    return out

def draw_bottom_right(img: np.ndarray, text: str, alpha: float = 0.6) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = _auto_font_scale_for_h(h, base=0.45)
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    pad = 10
    x2, y2 = w - 10, h - 10
    x1, y1 = x2 - (tw + 2 * pad), y2 - (th + 2 * pad)

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    cv2.putText(out, text, (x1 + pad, y2 - pad - 2), font, scale, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(out, text, (x1 + pad, y2 - pad - 2), font, scale, (60, 200, 60), 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Export (csv/json/xlsx/txt)
# ──────────────────────────────────────────────────────────────────────────────

def write_csv_fallback(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = [str(r.get(k, "")) for k in keys]
            f.write(",".join(vals) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Email sender (SMTP)
# ──────────────────────────────────────────────────────────────────────────────

def send_email_smtp(to_list: List[str], subject: str, body: str, attachments: List[Path]) -> Tuple[bool, str]:
    try:
        server = DEFAULT_SMTP_SERVER
        port = DEFAULT_SMTP_PORT
        user = DEFAULT_EMAIL_USER
        pwd = DEFAULT_EMAIL_PASS
        sender = DEFAULT_EMAIL_FROM

        if not (server and port and sender and to_list and user and pwd):
            return False, "Configuration e-mail incomplète (SMTP_USER / SMTP_PASSWORD)."

        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = ", ".join(to_list)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        for att in attachments:
            att = Path(att)
            if not att.exists():
                continue
            part = MIMEBase("application", "octet-stream")
            with open(att, "rb") as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{att.name}"')
            msg.attach(part)

        s = smtplib.SMTP_SSL(server, port, timeout=30)
        s.ehlo()
        s.login(user, pwd)
        s.sendmail(sender, to_list, msg.as_string())
        s.quit()
        return True, "E-mail envoyé."
    except Exception as e:
        return False, f"Échec envoi : {e}"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers Qt image (BGR → QPixmap)
# ──────────────────────────────────────────────────────────────────────────────

def bgr_to_qpixmap_scaled(frame_bgr: np.ndarray, target_size, fast: bool = True) -> QPixmap:
    """
    Convertit une frame BGR (OpenCV) → QPixmap scaled pour un QLabel.
    - target_size : QSize ou size() du QLabel
    - fast=True : Qt.FastTransformation (plus rapide sur Pi)
    """
    h, w, ch = frame_bgr.shape
    if hasattr(QImage.Format, "Format_BGR888"):
        qimg = QImage(frame_bgr.data, w, h, ch * w, QImage.Format.Format_BGR888)
    else:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)

    transform_mode = Qt.FastTransformation if fast else Qt.SmoothTransformation
    pix = QPixmap.fromImage(qimg).scaled(target_size, Qt.KeepAspectRatio, transform_mode)
    return pix


# ──────────────────────────────────────────────────────────────────────────────
# SessionManager : la “machine” runtime (capture → infer → overlay → record/export)
# ──────────────────────────────────────────────────────────────────────────────

class SessionManager(QObject):
    """
    Orchestrateur core :
    - Chargement modèle + tâches
    - Ouverture caméra
    - Thread d’inférence
    - Tick capture (appelé depuis UI via timer/Controller)
    - Recording + export

    Correctifs principaux vs version précédente :
    - Démarrage/arrêt robustes (pas de QThread zombie, stop idempotent)
    - Gestion sûre des frames (copie avant submit au worker, drop si backlog)
    - Re-open caméra plus stable (retente le backend sélectionné puis auto)
    - Noms de session cohérents (base_name mémorisé, collect_latest_session_files fiable)
    - Export robuste (start_time absent → pas de crash)
    - tick() ne plante jamais : erreurs → signal error/info, et retour silencieux
    """

    # Signals → Controller/UI
    tasksLoaded = Signal(dict, list)        # tasks_dict, tasks_order
    deviceChanged = Signal(str)             # "CPU" / "GPU"
    frameReady = Signal(object)             # QPixmap
    speedText = Signal(str)
    error = Signal(str)
    info = Signal(str)

    def __init__(self, root_dir: Path):
        super().__init__()
        self.root_dir = Path(root_dir).resolve()

        # Runtime state
        self.cap = None
        self.cap_backend = None
        self.cap_index = None
        self.lc_mjpeg: Optional[LibcameraMJPEGCapture] = None

        self.current_adapter: Optional[BaseAdapter] = None
        self.all_tasks: Dict[str, List[str]] = {}
        self.tasks: Dict[str, List[str]] = {}
        self.tasks_order: List[str] = []
        self.selected_tasks: List[str] = []

        self.device = "cpu"
        self._tick = 0
        self._read_fail = 0

        # Inference thread
        self.infer_thread: Optional[QThread] = None
        self.infer_worker: Optional[InferWorker] = None

        # Latest predictions
        self.last_items: List[Tuple[str, str, float]] = []
        self.lat_ema_s = None
        self.fps_hist = deque(maxlen=30)
        self.last_speed_text = ""

        # Display FPS
        self._last_disp_t = time.time()
        self._disp_fps_ema = None

        # Recording / export
        self.output_dir = self.root_dir / "runs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recording_video = False
        self.video_writer = None

        self.summary_rows: List[Dict[str, Any]] = []
        self.meta: Dict[str, Any] = {}
        self.session_started_at: Optional[datetime.datetime] = None

        # Session naming (important for export + collect_latest_session_files)
        self.session_name = ""
        self._last_session_base_name: str = ""   # mémorise le dernier nom réellement utilisé

        # Playback (si tu en as besoin ensuite)
        self.playback_mode = False
        self.play_fps = 25.0

        # Settings
        self.target_fps = 20
        self.infer_stride = 1
        self.prob_threshold = 0.5
        self.show_speed = False
        self.export_format = "json"
        self.classes_json_path = ""

        # Tick pacing (évite surchauffe UI si timer trop rapide)
        self._last_tick_t = 0.0

        # Partage thread-safe minimal (évite incohérences UI pendant update)
        self._lock = threading.Lock()

    # ───────────────────────── Models / tasks ─────────────────────────

    def list_models(self) -> List[Tuple[str, str]]:
        models_dir = (self.root_dir / "models").resolve()
        models_dir.mkdir(exist_ok=True)

        patterns = ("**/*.pt", "**/*.pth", "**/*.tflite", "**/*.onnx")
        files: List[str] = []
        for pat in patterns:
            files += glob.glob(str(models_dir / pat), recursive=True)
        files = sorted(files)

        out = []
        for f in files:
            fpath = Path(f).resolve()
            try:
                rel = str(fpath.relative_to(models_dir))
            except Exception:
                rel = str(fpath)
            out.append((rel, str(fpath)))
        return out

    def load_tasks_for_model(self, model_path: str) -> Dict[str, List[str]]:
        model_path = str(model_path).strip()
        if not model_path or not Path(model_path).exists():
            return {}

        # 1) classes.json si fourni
        if self.classes_json_path and Path(self.classes_json_path).exists():
            try:
                data = json.loads(Path(self.classes_json_path).read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {str(k): list(v) for k, v in data.items()}
            except Exception as e:
                self.error.emit(f"classes_json_path invalide: {e}")

        # 2) config/manifest
        try:
            cfg = ModelRegistry.read_config(model_path)
            if isinstance(cfg, dict) and "tasks" in cfg and isinstance(cfg["tasks"], dict):
                return {k: list(v) for k, v in cfg["tasks"].items()}
        except Exception as e:
            self.error.emit(f"Impossible de lire config modèle: {e}")

        # 3) fallback
        return {"Weather": ["No", "Yes"]}

    # ───────────────────────── Cameras ─────────────────────────

    def list_cameras(self) -> List[Tuple[str, object]]:
        entries: List[Tuple[str, object]] = []
        entries.append(("Auto (GStreamer ▶ libcamera-vid ▶ V4L2)", ("auto", None)))
        entries.append(("GStreamer (libcamerasrc)", ("gst", None)))
        entries.append(("rpicam-vid / libcamera-vid (MJPEG pipe)", ("lc-mjpeg", None)))
        for label, idx, dev in list_v4l2_devices():
            entries.append((f"V4L2: {label}", ("v4l2", idx)))
        return entries

    def _try_open_v4l2(self, cam_index: int, w: int, h: int, fps: int):
        trials = [
            ("MJPG", cv2.VideoWriter_fourcc(*"MJPG")),
            ("YUYV", cv2.VideoWriter_fourcc(*"YUYV")),
            ("AUTO", None),
        ]
        for label, fourcc in trials:
            cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
            if not cap or not cap.isOpened():
                continue
            try:
                if fourcc is not None:
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, fps)

                ok, frame = cap.read()
                if ok and frame is not None:
                    self.info.emit(f"Caméra V4L2 ouverte (idx={cam_index}, {label})")
                    return cap
            except Exception:
                pass
            cap.release()
        return None

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

        self.cap_backend = None
        self.cap_index = None
        self._read_fail = 0

    def _open_camera_backend(self, backend: str, index: Optional[int]) -> bool:
        """
        Ouvre une source. Mémorise cap_backend/cap_index en cas de succès.
        """
        self._release_camera()

        w = int(os.getenv("PI_CAM_WIDTH", "1280"))
        h = int(os.getenv("PI_CAM_HEIGHT", "720"))
        fps = int(os.getenv("PI_CAM_FPS", "30"))

        backend = (backend or "auto").strip()

        try:
            if backend == "gst":
                if cv_has_gstreamer():
                    pipe = default_gst_pipeline()
                    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                    if cap is not None and cap.isOpened():
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            self.cap = cap
                            self.cap_backend = "gst"
                            self.cap_index = None
                            self.info.emit("Caméra ouverte via GStreamer.")
                            return True
                        cap.release()
                return False

            if backend == "lc-mjpeg":
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
                        self.cap_index = None
                        self.info.emit("Caméra ouverte via libcamera MJPEG pipe.")
                        return True
                try:
                    self.lc_mjpeg.release()
                except Exception:
                    pass
                self.lc_mjpeg = None
                return False

            if backend == "v4l2":
                idx = 0 if index is None else int(index)
                cap = self._try_open_v4l2(idx, w, h, fps)
                if cap:
                    self.cap = cap
                    self.cap_backend = "v4l2"
                    self.cap_index = idx
                    return True
                return False

            if backend == "auto":
                # ordre : gst -> lc-mjpeg -> v4l2
                for pref_backend, pref_idx in [("gst", None), ("lc-mjpeg", None), ("v4l2", 0)]:
                    if self._open_camera_backend(pref_backend, pref_idx):
                        return True
                return False

        except Exception as e:
            self.error.emit(f"Erreur ouverture caméra ({backend}): {e}")
            self._release_camera()
            return False

        return False

    def _reopen_camera_safely(self):
        """
        Tente de rouvrir le backend courant (si connu), sinon auto.
        """
        b = self.cap_backend or "auto"
        idx = self.cap_index
        if b != "auto":
            if self._open_camera_backend(b, idx):
                return
        self._open_camera_backend("auto", None)

    # ───────────────────────── Start/Stop session ─────────────────────────

    def start_session(self, model_path: str, camera_data: object):
        """
        Démarre :
        - charge tasks
        - crée adapter (avec prune)
        - ouvre caméra
        - lance thread infer
        """
        model_path = str(model_path).strip()
        if not model_path or not Path(model_path).exists():
            self.error.emit("Aucun modèle valide sélectionné.")
            return

        # Stop propre si session déjà active
        try:
            self.stop_session()
        except Exception:
            pass

        # Tasks
        self.all_tasks = self.load_tasks_for_model(model_path)
        if not self.all_tasks:
            self.error.emit("Impossible de charger les tâches du modèle.")
            return

        if not self.selected_tasks:
            self.selected_tasks = list(self.all_tasks.keys())

        # Device
        try:
            self.device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"
        self.deviceChanged.emit("GPU" if self.device == "cuda" else "CPU")

        # Adapter + prune
        try:
            self.current_adapter = ModelRegistry.create(
                model_path=model_path,
                tasks=self.all_tasks,
                device=self.device,
                extra=None,
                selected_tasks=self.selected_tasks,
            )
        except Exception as e:
            self.current_adapter = None
            self.error.emit(f"Impossible de créer l'adapter du modèle: {e}")
            return

        self.tasks = getattr(self.current_adapter, "tasks", {}) or {}
        self.tasks_order = list(self.tasks.keys())
        if not self.tasks_order:
            self.error.emit("Le modèle n'a aucune tâche exploitable.")
            return

        self.tasksLoaded.emit(self.tasks, self.tasks_order)

        # Caméra
        backend, idx = camera_data if camera_data else ("auto", None)
        ok = self._open_camera_backend(backend, idx)
        if not ok and backend != "auto":
            ok = self._open_camera_backend("auto", None)
        if not ok:
            self.error.emit("Impossible d’ouvrir une source vidéo (GStreamer/rpicam-vid/V4L2).")
            return

        # Thread inference
        self._start_infer_thread()

        # Session meta
        self.summary_rows.clear()
        self.session_started_at = datetime.datetime.now()
        self.meta = {
            "model_path": model_path,
            "tasks": self.tasks_order.copy(),
            "start_time": self.session_started_at.isoformat(timespec="seconds"),
            "camera_backend": str(self.cap_backend),
        }

        with self._lock:
            self.last_items = []
            self.lat_ema_s = None
            self.fps_hist.clear()
            self.last_speed_text = ""
            self._tick = 0
            self._read_fail = 0

        self.info.emit("Session démarrée.")

    def _start_infer_thread(self):
        self.stop_infer_thread()

        if self.current_adapter is None:
            return

        self.infer_thread = QThread()
        self.infer_worker = InferWorker(self.current_adapter)
        self.infer_worker.moveToThread(self.infer_thread)

        self.infer_thread.started.connect(self.infer_worker.run)
        self.infer_worker.resultReady.connect(self._on_infer_result)
        self.infer_worker.error.connect(lambda msg: self.error.emit(f"Infer error: {msg}"))

        # Garantit stop si thread se termine
        self.infer_thread.finished.connect(lambda: self.info.emit("Thread inference terminé."))

        self.infer_thread.start()

    def stop_session(self):
        # Arrêt dans l'ordre : worker -> caméra -> recording -> export
        self.stop_infer_thread()
        self._release_camera()
        self.stop_recording()

        if self.session_started_at is not None:
            try:
                self.write_summary_file()
            except Exception as e:
                self.error.emit(f"Erreur export: {e}")
            self.session_started_at = None

        self.info.emit("Session arrêtée.")

    def stop_infer_thread(self):
        # Idempotent
        if self.infer_worker is not None:
            try:
                self.infer_worker.stop()
            except Exception:
                pass

        if self.infer_thread is not None:
            try:
                self.infer_thread.requestInterruption()
            except Exception:
                pass
            try:
                self.infer_thread.quit()
                self.infer_thread.wait(1200)
            except Exception:
                pass

        self.infer_worker = None
        self.infer_thread = None

    # ───────────────────────── Tick capture (appelé par le contrôleur) ─────────────────────────

    def tick(self, target_qsize, fast_scale: bool = True):
        """
        Un tick = lit une frame, soumet au worker (1/N), applique overlay, renvoie QPixmap.
        target_qsize : QSize du QLabel (ou widget) pour scaler côté Qt.
        """
        # pacing (évite que l’UI appelle 200 FPS si timer mal réglé)
        now = time.time()
        if self.target_fps and self.target_fps > 0:
            min_dt = 1.0 / float(self.target_fps)
            if (now - self._last_tick_t) < (0.65 * min_dt):
                return
        self._last_tick_t = now

        # FPS affichage (EMA)
        dt = max(1e-6, now - self._last_disp_t)
        self._last_disp_t = now
        fps_disp = 1.0 / dt
        self._disp_fps_ema = fps_disp if self._disp_fps_ema is None else (0.15 * fps_disp + 0.85 * self._disp_fps_ema)

        self._tick += 1

        frame = None
        ok = False

        try:
            if self.cap_backend == "lc-mjpeg" and self.lc_mjpeg is not None:
                ok, frame = self.lc_mjpeg.read()
            elif self.cap is not None:
                ok, frame = self.cap.read()
        except Exception as e:
            ok, frame = False, None
            self.error.emit(f"Erreur lecture caméra: {e}")

        if not ok or frame is None:
            self._read_fail += 1
            if self._read_fail >= 60:
                self.info.emit("Lecture caméra instable → tentative de réouverture.")
                self._reopen_camera_safely()
                self._read_fail = 0
            return

        self._read_fail = 0

        # Submit to inference worker 1/N (copie pour éviter data-race OpenCV)
        if self.infer_worker and (self._tick % max(1, int(self.infer_stride)) == 0):
            try:
                # drop si backlog (si le worker expose une API optionnelle)
                if hasattr(self.infer_worker, "queue_size") and callable(getattr(self.infer_worker, "queue_size")):
                    if self.infer_worker.queue_size() >= 2:
                        pass
                    else:
                        self.infer_worker.submit(frame.copy())
                else:
                    self.infer_worker.submit(frame.copy())
            except Exception as e:
                self.error.emit(f"Submit inference failed: {e}")

        # Overlay
        disp = frame  # pas besoin de copy si on ne modifie pas frame ; on va dessiner → copy
        try:
            disp = frame.copy()
            with self._lock:
                items = list(self.last_items)
                speed_txt = self.last_speed_text

            if items:
                disp = draw_predictions_panel(disp, items, location="tr")

            if self.show_speed:
                disp_txt = f"{self._disp_fps_ema:.1f} FPS affichage"
                if speed_txt:
                    disp_txt = speed_txt + f"  |  {disp_txt}"
                disp = draw_bottom_right(disp, disp_txt, alpha=0.55)

            # Recording
            if self.recording_video:
                self._record_frame(disp)

            # Convert to pixmap and emit
            pix = bgr_to_qpixmap_scaled(disp, target_qsize, fast=fast_scale)
            self.frameReady.emit(pix)

        except Exception as e:
            self.error.emit(f"Erreur overlay/affichage: {e}")

    # ───────────────────────── Inference result ─────────────────────────

    @Slot(dict, float)
    def _on_infer_result(self, outputs: dict, elapsed: float):
        try:
            items: List[Tuple[str, str, float]] = []
            per_task: Dict[str, Dict[str, Any]] = {}

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
                label = labels[idx] if (idx < len(labels) and score >= float(self.prob_threshold)) else "Unknown"

                items.append((task, label, score))
                per_task[task] = {"label": label, "score": score}

            # Update shared state
            with self._lock:
                self.last_items = items

                self.lat_ema_s = elapsed if self.lat_ema_s is None else (0.2 * elapsed + 0.8 * self.lat_ema_s)
                inst_fps = 1.0 / max(elapsed, 1e-6)
                self.fps_hist.append(inst_fps)
                avg_fps = sum(self.fps_hist) / max(1, len(self.fps_hist))
                self.last_speed_text = f"{self.lat_ema_s * 1000:.0f} ms IA  |  {avg_fps:.1f} FPS IA"
                speed_txt = self.last_speed_text

            self.speedText.emit(speed_txt)

            # Summary row
            now = datetime.datetime.now()
            ts_ms = int(now.timestamp() * 1000)

            model_name = ""
            try:
                model_name = Path(self.meta.get("model_path", "")).name
            except Exception:
                model_name = str(self.meta.get("model_path", ""))

            row = {
                "timestamp_ms": ts_ms,
                "iso_time": now.isoformat(timespec="milliseconds"),
                "latency_s": round(float(elapsed), 4),
                "camera": str(self.cap_backend),
                "model": model_name,
            }
            for t, info in per_task.items():
                row[f"{t}_label"] = info["label"]
                row[f"{t}_score"] = round(float(info["score"]), 4)

            self.summary_rows.append(row)

        except Exception as e:
            self.error.emit(f"Erreur traitement résultat inference: {e}")

    # ───────────────────────── Recording ─────────────────────────

    def start_recording(self):
        self.recording_video = True
        self.info.emit("Recording ON")

    def stop_recording(self):
        self.recording_video = False
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None
        self.info.emit("Recording OFF")

    def _record_frame(self, disp_bgr: np.ndarray):
        try:
            h, w = disp_bgr.shape[:2]
            if self.video_writer is None:
                base = self._resolve_session_base_name(self.meta.get("model_path", "session"))
                out_path = self.output_dir / f"{base}.avi"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.video_writer = cv2.VideoWriter(str(out_path), fourcc, float(self.target_fps or 20), (w, h))
            if self.video_writer:
                self.video_writer.write(disp_bgr)
        except Exception as e:
            self.error.emit(f"Erreur recording: {e}")
            self.stop_recording()

    def _build_default_session_name(self, model_path: str) -> str:
        base = Path(model_path).stem if model_path else "session"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{ts}"

    def _resolve_session_base_name(self, model_path: str) -> str:
        """
        Renvoie le nom de session réellement utilisé (et le mémorise).
        - Si self.session_name est fixé : on le respecte
        - Sinon : on génère un nom et on le mémorise pour collect_latest_session_files()
        """
        name = (self.session_name or "").strip()
        if not name:
            # si déjà calculé pour cette session, réutilise
            if self._last_session_base_name:
                return self._last_session_base_name
            name = self._build_default_session_name(model_path)
        self._last_session_base_name = name
        return name

    # ───────────────────────── Export ─────────────────────────

    def write_summary_file(self):
        end_time = datetime.datetime.now()

        # start_time robuste
        start_iso = self.meta.get("start_time")
        if start_iso:
            try:
                start_dt = datetime.datetime.fromisoformat(start_iso)
            except Exception:
                start_dt = self.session_started_at or end_time
        else:
            start_dt = self.session_started_at or end_time
            self.meta["start_time"] = start_dt.isoformat(timespec="seconds")

        self.meta["end_time"] = end_time.isoformat(timespec="seconds")
        self.meta["duration_s"] = (end_time - start_dt).total_seconds()

        base = self._resolve_session_base_name(self.meta.get("model_path", "session"))
        fmt = (self.export_format or "json").strip().lower()
        out_base = self.output_dir / f"{base}"

        # meta
        meta_path = out_base.with_suffix(".meta.json")
        meta_obj = {"meta": self.meta, "summary_count": len(self.summary_rows)}
        meta_path.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")

        # frames
        if fmt == "json":
            out = {"meta": self.meta, "frames": self.summary_rows}
            out_base.with_suffix(".json").write_text(json.dumps(out, indent=2), encoding="utf-8")

        elif fmt == "csv":
            if pd is not None:
                pd.DataFrame(self.summary_rows).to_csv(out_base.with_suffix(".csv"), index=False)
            else:
                write_csv_fallback(out_base.with_suffix(".csv"), self.summary_rows)

        elif fmt == "xlsx":
            if pd is not None:
                df = pd.DataFrame(self.summary_rows)
                try:
                    df.to_excel(out_base.with_suffix(".xlsx"), index=False)
                except Exception:
                    df.to_csv(out_base.with_suffix(".csv"), index=False)
            else:
                write_csv_fallback(out_base.with_suffix(".csv"), self.summary_rows)

        elif fmt == "txt":
            with open(out_base.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write("# META\n")
                for k, v in self.meta.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n# FRAMES\n")
                if self.summary_rows:
                    keys = list(self.summary_rows[0].keys())
                    f.write("\t".join(keys) + "\n")
                    for r in self.summary_rows:
                        f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")

        else:
            # fallback
            out = {"meta": self.meta, "frames": self.summary_rows}
            out_base.with_suffix(".json").write_text(json.dumps(out, indent=2), encoding="utf-8")

        self.info.emit(f"Export écrit: {out_base.name}.*")

    # ───────────────────────── Convenience core utilities ─────────────────────────

    def copy_files(self, files: List[str], dest: str) -> int:
        dest = str(dest)
        Path(dest).mkdir(parents=True, exist_ok=True)
        ok_count = 0
        for f in files:
            try:
                shutil.copy2(f, dest)
                ok_count += 1
            except Exception:
                pass
        return ok_count

    def collect_latest_session_files(self) -> List[Path]:
        """
        Retourne les fichiers du dernier export (meta + summary + video).
        Fiable même si session_name était vide (grâce à _last_session_base_name).
        """
        base = (self.session_name or "").strip() or (self._last_session_base_name or "").strip()
        if not base:
            return []

        outs: List[Path] = []
        meta = self.output_dir / f"{base}.meta.json"
        if meta.exists():
            outs.append(meta)

        ext = (self.export_format or "json").strip().lower()
        if ext == "xlsx":
            path = self.output_dir / f"{base}.xlsx"
            if not path.exists():
                alt = self.output_dir / f"{base}.csv"
                if alt.exists():
                    path = alt
        else:
            path = self.output_dir / f"{base}.{ext}"
        if path.exists():
            outs.append(path)

        vid = self.output_dir / f"{base}.avi"
        if vid.exists():
            outs.append(vid)

        return outs

