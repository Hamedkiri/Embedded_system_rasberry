# loader_patchgan_from_manifest.py — loader minimal et robuste
import os, json, torch
from typing import Tuple, Dict, List, Any
from patchGAN_module import checkpoint_has_se, MultiTaskPatchGAN, load_model_weights
# loader_patchgan_from_manifest.py — loader minimal et robuste (avec PRUNE des tâches)
import os, json, torch
from typing import Tuple, Dict, List, Any, Optional

# ⚠️ On utilise les classes/fonctions DÉJÀ DÉFINIES plus haut dans CE fichier :
# - MultiTaskPatchGAN
# - load_model_weights
# - checkpoint_has_se
# (Donc pas d'import externe patchGAN_module ici.)

def create_patchgan_from_manifest(
    manifest_path: str,
    device: str = "cpu",
    selected_tasks: Optional[List[str]] = None,
) -> Tuple[torch.nn.Module, Dict[str, List[str]]]:
    """
    Charge un MultiTaskPatchGAN à partir d'un manifest.json.
    - Supporte la PRUNE des têtes via `selected_tasks` (liste de noms de tâches à garder).
    - Détecte automatiquement l'usage de Squeeze&Excitation (attn_use_se="auto") si possible.
    - Charge les poids en 'strict=True' quand aucune PRUNE n'est demandée, sinon 'strict=False'.

    Retourne:
      (model, tasks_labels)
      - model: nn.Module prêt à l'inférence (.eval())
      - tasks_labels: mapping {task_name -> [labels...]} (déjà filtré selon selected_tasks)
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)

    # 1) labels par tâche attendus par l'UI (supporte "tasks" ou "labels")
    tasks: Dict[str, List[str]] = man.get("tasks") or man.get("labels") or {}
    if not isinstance(tasks, dict) or not tasks:
        raise RuntimeError("Manifest: champ 'tasks' (ou 'labels') manquant ou invalide.")

    # 2) chemins
    extra: Dict[str, Any] = man.get("extra", {})
    weights_rel = man.get("weights", "weights.pth")
    root = os.path.dirname(manifest_path)
    weights_path = os.path.normpath(os.path.join(root, weights_rel))

    # 3) hyperparams PatchGAN (valeurs par défaut compatibles)
    patch_size = int(extra.get("patch_size", 31))
    attn_tau   = float(extra.get("attn_tau", 0.7))
    attn_softmax_spatial = bool(extra.get("attn_softmax_spatial", True))
    ablate_attention = bool(extra.get("ablate_attention", False))

    attn_use_se_cfg = extra.get("attn_use_se", "auto")
    if attn_use_se_cfg == "auto":
        try:
            attn_use_se = bool(checkpoint_has_se(weights_path, device=device))
        except Exception:
            attn_use_se = False
    else:
        attn_use_se = bool(attn_use_se_cfg)

    # 4) PRUNE des tâches si demandé (en conservant l'ordre du manifest)
    if selected_tasks:
        sel_set = set(selected_tasks)
        pruned_tasks = {t: lbls for t, lbls in tasks.items() if t in sel_set}
        missing = [t for t in selected_tasks if t not in tasks]
        if missing:
            print(f"[PatchGAN][warn] Tâches demandées absentes du manifest: {missing}")
        if not pruned_tasks:
            print("[PatchGAN][warn] PRUNE vide → on retombe sur toutes les tâches du manifest.")
            pruned_tasks = tasks
    else:
        pruned_tasks = tasks

    # 5) construction modèle (nb de classes par tâche)
    tasks_dict = {t: len(lbls) for t, lbls in pruned_tasks.items()}
    model = MultiTaskPatchGAN(
        tasks_dict=tasks_dict,
        input_nc=3,
        ndf=64,
        norm="instance",
        patch_size=patch_size,
        device=device,
        attn_tau=attn_tau,
        attn_use_se=attn_use_se,
        attn_softmax_spatial=attn_softmax_spatial,
        ablate_attention=ablate_attention,
    ).to(device).eval()

    # 6) chargement des poids
    #    - strict=True si aucune PRUNE (mêmes têtes que le checkpoint)
    #    - strict=False si PRUNE (on autorise missing/unexpected)
    strict_load = (len(pruned_tasks) == len(tasks))
    try:
        load_model_weights(model, weights_path, torch.device(device), strict=strict_load)
    except Exception as e:
        # tentative de fallback: recharger en non strict si l'utilisateur avait imposé la PRUNE
        if strict_load:
            raise
        print(f"[PatchGAN][warn] Chargement partiel (non strict) suite à PRUNE: {e}")
        load_model_weights(model, weights_path, torch.device(device), strict=False)

    # 7) renvoyer le modèle prêt + les labels (déjà filtrés)
    return model, pruned_tasks
