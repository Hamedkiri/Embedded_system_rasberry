# loader_patchgan_from_manifest.py
# -*- coding: utf-8 -*-
"""
Loader PatchGAN depuis manifest.json (robuste / tolérant aux ablations).

Exigence (selon ton message) :
- NE PAS analyser les poids pour deviner SE/attention sauf si attn_use_se == "auto".
- Sinon: on se FIe au manifest.
- Retirer en amont (AVANT load_state_dict) les parties absentes selon le manifest:
  * si attn_use_se=false  -> retirer toutes les clés ".se."
  * si ablate_attention=true -> retirer toutes les clés liées à l'attention (patterns usuels)
- PRUNE des tâches:
  * on construit le modèle seulement avec les tâches retenues
  * on retire du state_dict les têtes des tâches non retenues pour éviter les "unexpected"

Ce fichier ne charge le checkpoint qu'une seule fois (sauf si attn_use_se="auto" et
que tu utilises un helper externe qui relit — ici on évite, on déduit depuis le state_dict déjà chargé).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

import torch

# À ADAPTER à ton projet
from patchGAN_module import MultiTaskPatchGAN


# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

def _as_bool(v: Any, default: bool = False) -> bool:
    """Convertit proprement un champ du manifest qui peut être bool/str/int."""
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
    return default


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state_dict(weights_path: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Charge un checkpoint et retourne un state_dict (mapping param_name -> tensor).
    Supporte plusieurs formats de sauvegarde.
    """
    ckpt = torch.load(weights_path, map_location=torch.device(device))
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # parfois le dict est déjà le state_dict
        return ckpt
    raise RuntimeError(f"Checkpoint illisible: {weights_path}")


def _has_key_substring(sd: Dict[str, Any], needle: str) -> bool:
    return any(needle in k for k in sd.keys())


def _filter_state_dict(
    sd: Dict[str, Any],
    drop_substrings: Optional[List[str]] = None,
    drop_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Filtre un state_dict en supprimant des clés."""
    drop_substrings = drop_substrings or []
    drop_prefixes = drop_prefixes or []
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        if any(ss in k for ss in drop_substrings):
            continue
        if any(k.startswith(p) for p in drop_prefixes):
            continue
        out[k] = v
    return out


def _task_head_prefix(task_name: str) -> str:
    # Convention standard dans ton erreur: task_heads.<Task Name>.se.mlp...
    return f"task_heads.{task_name}."


def _prune_state_dict_for_tasks(
    sd: Dict[str, Any],
    pruned_tasks: Dict[str, List[str]],
    all_tasks: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Supprime du checkpoint les poids des têtes des tâches non conservées."""
    keep = set(pruned_tasks.keys())
    drop_prefixes: List[str] = []
    for t in all_tasks.keys():
        if t not in keep:
            drop_prefixes.append(_task_head_prefix(t))
    if not drop_prefixes:
        return sd
    return _filter_state_dict(sd, drop_prefixes=drop_prefixes)


def _drop_attention_keys(sd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retire des clés typiques d'un bloc attention.
    Comme on ne connait pas tes noms exacts, on couvre les patterns les plus communs
    (et ça n'affecte pas strict=False si jamais il en reste).
    """
    patterns = [
        ".attn.", "attn.", ".attention.", "attention.",
        ".q_proj.", ".k_proj.", ".v_proj.", ".out_proj.",
        ".to_q.", ".to_k.", ".to_v.", ".to_out.",
        ".query.", ".key.", ".value.",
        ".mhsa.", "mhsa.", ".msa.", "msa.",
    ]
    return _filter_state_dict(sd, drop_substrings=patterns)


def _load_into_model(model: torch.nn.Module, sd: Dict[str, Any], strict: bool) -> Tuple[List[str], List[str]]:
    """
    Wrapper cross-version: renvoie (missing_keys, unexpected_keys)
    """
    res = model.load_state_dict(sd, strict=strict)
    # res est un IncompatibleKeys dans les versions récentes
    mk = list(getattr(res, "missing_keys", []))
    uk = list(getattr(res, "unexpected_keys", []))
    # si ancienne version renvoie tuple
    if not mk and not uk and isinstance(res, tuple) and len(res) == 2:
        mk, uk = list(res[0]), list(res[1])
    return mk, uk


# ──────────────────────────────────────────────────────────────────────────────
# API principale
# ──────────────────────────────────────────────────────────────────────────────

def create_patchgan_from_manifest(
    manifest_path: str,
    device: str = "cpu",
    selected_tasks: Optional[List[str]] = None,
) -> Tuple[torch.nn.Module, Dict[str, List[str]]]:
    """
    Charge MultiTaskPatchGAN depuis manifest.json + weights.

    Règle:
    - si extra["attn_use_se"] != "auto": on fait CONFIANCE au manifest (pas d'analyse du ckpt pour deviner)
    - si "auto": on déduit depuis le state_dict (déjà chargé) via présence de ".se."

    Retour:
      (model.eval(), pruned_tasks)
    """
    mp = Path(str(manifest_path))
    if not mp.exists():
        raise RuntimeError(f"manifest introuvable: {mp}")

    man = _read_json(mp)

    # 1) tasks/labels
    tasks: Dict[str, List[str]] = man.get("tasks") or man.get("labels") or {}
    if not isinstance(tasks, dict) or not tasks:
        raise RuntimeError("Manifest: champ 'tasks' (ou 'labels') manquant ou invalide.")

    # 2) weights path
    root = mp.parent
    weights_rel = man.get("weights", "weights.pth")
    weights_path = (root / weights_rel).resolve()
    if not weights_path.exists():
        raise RuntimeError(f"weights introuvables: {weights_path}")

    # 3) extra
    extra: Dict[str, Any] = man.get("extra", {}) or {}
    patch_size = int(extra.get("patch_size", 31))
    attn_tau = float(extra.get("attn_tau", 0.7))
    attn_softmax_spatial = _as_bool(extra.get("attn_softmax_spatial", True), default=True)

    # IMPORTANT: ici on respecte le manifest (ablation)
    ablate_attention = _as_bool(extra.get("ablate_attention", False), default=False)

    # 4) PRUNE tasks (ordre du manifest conservé)
    if selected_tasks:
        sel = set(selected_tasks)
        pruned_tasks = {t: lbls for t, lbls in tasks.items() if t in sel}
        if not pruned_tasks:
            pruned_tasks = tasks
    else:
        pruned_tasks = tasks

    tasks_dict = {t: len(lbls) for t, lbls in pruned_tasks.items()}

    # 5) Charger UNE SEULE FOIS le state_dict (on s'en sert aussi pour "auto")
    sd = _load_state_dict(str(weights_path), device=device)

    # 6) Déterminer attn_use_se (uniquement en mode auto on inspecte sd)
    attn_use_se_cfg = extra.get("attn_use_se", "auto")

    if isinstance(attn_use_se_cfg, str) and attn_use_se_cfg.strip().lower() == "auto":
        # auto: inspection du sd déjà chargé (PAS de second chargement)
        attn_use_se = _has_key_substring(sd, ".se.")
    else:
        # on fait confiance au manifest (pas d'analyse des poids)
        attn_use_se = _as_bool(attn_use_se_cfg, default=False)

    # 7) Construire le modèle selon le manifest (SE/attention)
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

    # 8) Adapter le state_dict EN AMONT selon le manifest
    # 8.a) Si SE désactivé: retirer les poids SE du checkpoint
    if not attn_use_se:
        sd = _filter_state_dict(sd, drop_substrings=[".se."])

    # 8.b) Si attention ablatée: retirer les poids attention du checkpoint
    #       (ça évite d'avoir des unexpected si ton modèle n'a pas ces modules)
    if ablate_attention:
        sd = _drop_attention_keys(sd)

    # 8.c) PRUNE: retirer les têtes des tâches non conservées
    sd = _prune_state_dict_for_tasks(sd, pruned_tasks=pruned_tasks, all_tasks=tasks)

    # 9) Charger les poids
    #    - strict=True seulement si aucune prune ET qu'on ne s'attend pas à des surprises
    #    - sinon strict=False (recommandé pour robustesse)
    strict_load = (len(pruned_tasks) == len(tasks))

    try:
        mk, uk = _load_into_model(model, sd, strict=strict_load)
        if (mk or uk) and not strict_load:
            print(f"[PatchGAN][warn] load_state_dict non strict: missing={len(mk)} unexpected={len(uk)}")
    except RuntimeError as e:
        # fallback non-strict + filtrage par clés existantes dans le modèle
        print(f"[PatchGAN][warn] strict load a échoué: {e}")
        model_keys = set(model.state_dict().keys())
        sd2 = {k: v for k, v in sd.items() if k in model_keys}
        mk, uk = _load_into_model(model, sd2, strict=False)
        if mk:
            print(f"[PatchGAN][warn] missing_keys (non strict) = {len(mk)}")
        if uk:
            print(f"[PatchGAN][warn] unexpected_keys (non strict) = {len(uk)}")

    return model, pruned_tasks


# ──────────────────────────────────────────────────────────────────────────────
# Debug CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tasks", default="", help="comma-separated task names to keep")
    args = ap.parse_args()

    sel = [t.strip() for t in args.tasks.split(",") if t.strip()] or None
    m, t = create_patchgan_from_manifest(args.manifest, device=args.device, selected_tasks=sel)
    print("Loaded model OK")
    print("Tasks:", list(t.keys()))
