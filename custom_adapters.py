# custom_adapters.py
import os
import json
import numpy as np
import cv2
import torch
from typing import Dict, List, Any, Optional

# Importe la base/registry de ton projet
from model_registry import BaseAdapter, register_adapter

# ⚠️ ADAPTE ces imports au nom du module/fichier où tu as défini ces fonctions/classes
# ex: from patchgan_impl import MultiTaskPatchGAN, load_model_weights, checkpoint_has_se
from your_patchgan_module import MultiTaskPatchGAN, load_model_weights, checkpoint_has_se


def _preprocess_bgr_imagenet(frame_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """BGR uint8 -> resize/crop -> RGB float32 normalisé ImageNet, shape [H,W,3]."""
    img = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    # center crop 224
    off = (256 - size) // 2
    img = img[off:off+size, off:off+size, :]
    img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return img


@register_adapter("patchgan_mtm")
class PatchGANMTMAdapter(BaseAdapter):
    """
    Adapter pour le Multi-Task PatchGAN (classifier multi-tâches).
    - Construit l'archi d'après manifest.extra (patch_size, attention...)
    - Détecte SE ('attn_use_se') automatiquement si demandé
    - Charge les poids
    - Expose predict_bgr(frame_bgr) -> dict{task: logits Tensor[1,C]}
    """
    def __init__(
        self,
        weights_path: str,
        tasks: Dict[str, List[str]],
        device: str = "cpu",
        input_size: int = 224,
        extra: Optional[Dict[str, Any]] = None
    ):
        super().__init__(tasks, device, input_size)
        extra = extra or {}

        # --- Hyper/archi depuis manifest ---
        patch_size = int(extra.get("patch_size", 31))
        attn_tau   = float(extra.get("attn_tau", 0.7))
        attn_softmax_spatial = bool(extra.get("attn_softmax_spatial", True))

        # Détection auto de SE si 'auto'
        attn_use_se_cfg = extra.get("attn_use_se", "auto")
        if attn_use_se_cfg == "auto":
            has_se = bool(checkpoint_has_se(weights_path, device=device))
            attn_use_se = has_se
        else:
            attn_use_se = bool(attn_use_se_cfg)

        # Tâches -> nb classes
        tasks_dict = {t: len(cls) for t, cls in tasks.items()}

        # --- Construire le modèle ---
        self.model = MultiTaskPatchGAN(
            tasks_dict=tasks_dict,
            input_nc=3,
            ndf=64,
            norm="instance",
            patch_size=patch_size,
            device=device,
            attn_tau=attn_tau,
            attn_use_se=attn_use_se,
            attn_softmax_spatial=attn_softmax_spatial,
            ablate_attention=False
        ).to(device).eval()

        # --- Charger les poids ---
        load_model_weights(self.model, weights_path, torch.device(device), strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict_bgr(self, frame_bgr: np.ndarray) -> Dict[str, torch.Tensor]:
        img = _preprocess_bgr_imagenet(frame_bgr, self.input_size)
        x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        outs = self.model(x)  # dict: task -> logits [1, C]
        return outs
