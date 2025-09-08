# loader_patchgan_from_manifest.py — loader minimal et robuste
import os, json, torch
from typing import Tuple, Dict, List, Any
from patchGAN_module import checkpoint_has_se, MultiTaskPatchGAN, load_model_weights
# ⚠️ On utilise les classes/fonctions DÉJÀ DÉFINIES plus haut dans CE fichier :
# - MultiTaskPatchGAN
# - load_model_weights
# - checkpoint_has_se
# (Donc pas d'import externe patchGAN_module ici.)

def create_patchgan_from_manifest(manifest_path: str, device: str = "cpu") -> Tuple[torch.nn.Module, Dict[str, List[str]]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)

    # 1) labels par tâche attendus par l'UI
    tasks: Dict[str, List[str]] = man["tasks"]

    # 2) chemins
    extra: Dict[str, Any] = man.get("extra", {})
    weights_rel = man.get("weights", "weights.pth")
    root = os.path.dirname(manifest_path)
    weights_path = os.path.normpath(os.path.join(root, weights_rel))

    # 3) hyperparams PatchGAN
    patch_size = int(extra.get("patch_size", 31))
    attn_tau   = float(extra.get("attn_tau", 0.7))
    attn_softmax_spatial = bool(extra.get("attn_softmax_spatial", True))

    attn_use_se_cfg = extra.get("attn_use_se", "auto")
    if attn_use_se_cfg == "auto":
        attn_use_se = bool(checkpoint_has_se(weights_path, device=device))
    else:
        attn_use_se = bool(attn_use_se_cfg)

    # 4) construction modèle (nb de classes par tâche)
    tasks_dict = {t: len(cls) for t, cls in tasks.items()}
    model = MultiTaskPatchGAN(
        tasks_dict=tasks_dict,
        input_nc=3, ndf=64, norm="instance",
        patch_size=patch_size, device=device,
        attn_tau=attn_tau,
        attn_use_se=attn_use_se,
        attn_softmax_spatial=attn_softmax_spatial,
        ablate_attention=False
    ).to(device).eval()

    # 5) chargement des poids
    load_model_weights(model, weights_path, torch.device(device), strict=True)

    # 6) renvoyer le modèle prêt + les labels
    return model, tasks
