# model_registry.py
import os, json, importlib
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

# ─── Préproc partagé ───
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_bgr_imagenet(frame_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    import cv2
    img = cv2.resize(frame_bgr, (size, size))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return img

# ─── BaseAdapter ───
class BaseAdapter:
    def __init__(self, tasks: Dict[str, List[str]], device: str = "cpu", input_size: int = 224):
        self.tasks = tasks
        self.device = device
        self.input_size = int(input_size)
    def predict_bgr(self, frame_bgr: np.ndarray):
        raise NotImplementedError

# ─── Registry custom (par id) ───
_CUSTOM = {}
def register_adapter(model_id: str):
    def deco(cls):
        _CUSTOM[model_id] = cls
        return cls
    return deco

# ─── Adapters génériques par backend (fallback) ───
class _GenericTorchAdapter(BaseAdapter):
    """Fallback PyTorch: attend un torchscript qui renvoie dict task->logits."""
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224, extra=None):
        super().__init__(tasks, device, input_size)
        import torch
        self.m = torch.jit.load(weights_path, map_location=device)
        self.m.eval()
        self.torch = torch
    def predict_bgr(self, frame_bgr: np.ndarray):
        x = preprocess_bgr_imagenet(frame_bgr, self.input_size)
        x = self.torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).to(self.device)
        return self.m(x)

class _GenericONNXAdapter(BaseAdapter):
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224, extra=None):
        super().__init__(tasks, device, input_size)
        import onnxruntime as ort
        self.sess = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.order = (extra or {}).get("output_task_order", list(tasks.keys()))
    def predict_bgr(self, frame_bgr: np.ndarray):
        x = preprocess_bgr_imagenet(frame_bgr, self.input_size).transpose(2,0,1)[None].astype(np.float32)
        outs = self.sess.run(self.output_names, {self.input_name: x})
        out = {}
        if len(outs) == len(self.order):
            for t, arr in zip(self.order, outs):
                out[t] = arr
        else:
            total = sum(len(v) for v in self.tasks.values())
            flat = outs[0]
            off = 0
            for t, cls in self.tasks.items():
                n = len(cls); out[t] = flat[:, off:off+n]; off += n
        import torch
        return {k: torch.from_numpy(v) for k, v in out.items()}

class _GenericTFLiteAdapter(BaseAdapter):
    def __init__(self, weights_path: str, tasks: Dict[str, List[str]], device="cpu", input_size=224, extra=None):
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
        self.order = extra.get("output_task_order", list(tasks.keys()))
        import torch; self.torch = torch
    def predict_bgr(self, frame_bgr: np.ndarray):
        x = preprocess_bgr_imagenet(frame_bgr, self.input_size)[None].astype(np.float32)  # [1,H,W,3]
        inp = self.input_details[0]
        if inp["dtype"].__name__ == "uint8":
            x = (np.clip((x * _IMAGENET_STD + _IMAGENET_MEAN), 0, 1) * 255).astype(np.uint8)
        self.interp.set_tensor(inp["index"], x)
        self.interp.invoke()
        outs = [self.interp.get_tensor(od["index"]) for od in self.output_details]
        out = {}
        if len(outs) == len(self.order):
            for t, arr in zip(self.order, outs):
                out[t] = arr
        else:
            total = sum(len(v) for v in self.tasks.values())
            flat = outs[0]; off = 0
            for t, cls in self.tasks.items():
                n = len(cls); out[t] = flat[:, off:off+n]; off += n
        return {k: self.torch.from_numpy(v) for k, v in out.items()}

# ─── Factory à partir d’un manifest ───
class ModelRegistry:
    @staticmethod
    def load_manifest(path: str) -> Dict[str, Any]:
        p = Path(path)
        if p.is_dir(): p = p / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8"))

    @classmethod
    def create_from_manifest(cls, manifest_path: str, device="cpu"):
        man = cls.load_manifest(manifest_path)
        model_id    = man["id"]
        interface   = man.get("interface", "classifier")
        backend     = man.get("backend", "pytorch")
        input_size  = int(man.get("input_size", 224))
        weights_rel = man.get("weights", "weights.pth")
        tasks       = man.get("tasks", {})
        extra       = man.get("extra", {})

        root = Path(manifest_path).parent if manifest_path.endswith(".json") else Path(manifest_path)
        weights_path = str((root / weights_rel).resolve())

        # 1) Adapter personnalisé si dispo
        if model_id in _CUSTOM:
            Adapter = _CUSTOM[model_id]
            inst = Adapter(weights_path=weights_path, tasks=tasks, device=device, input_size=input_size, extra=extra)
            return inst, interface, man

        # 2) Fallback générique par backend
        if backend == "pytorch":
            inst = _GenericTorchAdapter(weights_path, tasks, device=device, input_size=input_size, extra=extra)
        elif backend == "onnx":
            inst = _GenericONNXAdapter(weights_path, tasks, device=device, input_size=input_size, extra=extra)
        elif backend == "tflite":
            inst = _GenericTFLiteAdapter(weights_path, tasks, device=device, input_size=input_size, extra=extra)
        else:
            raise ValueError(f"Backend inconnu: {backend}")
        return inst, interface, man
