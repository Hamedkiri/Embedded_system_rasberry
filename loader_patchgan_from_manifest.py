# loader_patchgan_from_manifest.py
# -*- coding: utf-8 -*-
"""
Loader PatchGAN depuis manifest.json (robuste / tolérant aux ablations).

Objectif:
- Lire manifest.json
- Construire MultiTaskPatchGAN avec les bons hyperparams (patch_size, attn_tau, ...)
- Respecter attn_use_se du manifest:
    * si False -> construire le modèle SANS SE
    * si True  -> construire le modèle AVEC SE
    * si "auto" -> déduire depuis le checkpoint (fallback False si doute)
- Charger les poids sans erreur même si:
    * le checkpoint contient/omet des clés SE
    * PRUNE des tâches activée
    * certaines clés existent dans le ckpt mais pas dans le modèle (unexpected)
    * certaines clés manquent dans le ckpt (missing)
- "Se servir du manifest pour retirer en amont les parties absentes":
    -> si attn_use_se est False, on filtre du state_dict toutes les clés ".se."
       AVANT load_state_dict, et on construit le modèle sans se.

NOTE:
- Ce fichier suppose que MultiTaskPatchGAN et ses dépendances sont accessibles
  via import patchGAN_module (ou à adapter selon ton projet).
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

import torch

# Adapte ces imports à ton arborescence:
# - MultiTaskPatchGAN: ton modèle
# - load_model_weights: ta fonction de chargement si tu veux, sinon on fait direct torch.load + load_state_dict
# - checkpoint_has_se: optionnel, mais pratique si "auto"
from patchGAN_module import MultiTaskPatchGAN

# Optionnels: si tu les as déjà, on les utilise, sinon on fallback proprement.
try:
    from patchGAN_module import checkpoint_has_se  # doit retourner bool
except Exception:
    checkpoint_has_se = None


# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

def _as_bool(v: Any, default: bool = False) -> bool:
    """Convertit proprement un champ de manifest qui peut être bool, str, int..."""
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


def _load_checkpoint(weights_path: str, device: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(weights_path, map_location=torch.device(device))

    # Plusieurs formats possibles :
    # - state_dict direct
    # - {"state_dict": ...}
    # - {"model": ...}
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Checkpoint illisible: {weights_path}")


def _state_dict_has_any_key(sd: Dict[str, Any], needle: str) -> bool:
    for k in sd.keys():
        if needle in k:
            return True
    return False


def _filter_state_dict(sd: Dict[str, Any],
                      drop_substrings: Optional[List[str]] = None,
                      keep_prefixes: Optional[List[str]] = None,
                      drop_prefixes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Filtre un state_dict:
    - drop_substrings : enlève si substring dans la clé
    - drop_prefixes   : enlève si k.startswith(prefix)
    - keep_prefixes   : si fourni, garde seulement les clés commençant par un des prefixes
    """
    drop_substrings = drop_substrings or []
    drop_prefixes = drop_prefixes or []

    out = {}
    for k, v in sd.items():
        if any(ss in k for ss in drop_substrings):
            continue
        if any(k.startswith(p) for p in drop_prefixes):
            continue
        if keep_prefixes is not None and len(keep_prefixes) > 0:
            if not any(k.startswith(p) for p in keep_prefixes):
                continue
        out[k] = v
    return out


def _task_head_prefix(task_name: str) -> str:
    # Convention fréquente : task_heads.<task_name>.
    # Si tu utilises une normalisation des noms dans MultiTaskPatchGAN, adapte ici.
    return f"task_heads.{task_name}."


def _prune_state_dict_for_tasks(sd: Dict[str, Any],
                               pruned_tasks: Dict[str, List[str]],
                               all_tasks: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Si on prune des tâches côté modèle, on supprime du checkpoint:
    - toutes les clés task_heads.<task>.* où <task> n'est pas conservée.
    On évite ainsi d'avoir énormément de "unexpected keys".
    """
    keep = set(pruned_tasks.keys())
    drop_prefixes = []
    for t in all_tasks.keys():
        if t not in keep:
            drop_prefixes.append(_task_head_prefix(t))
    if not drop_prefixes:
        return sd
    return _filter_state_dict(sd, drop_prefixes=drop_prefixes)


def _load_into_model(model: torch.nn.Module, sd: Dict[str, Any], strict: bool):
    """
    Wrapper pour obtenir un message exploitable si mismatch.
    """
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    # PyTorch >= 2 renvoie IncompatibleKeys object; mais il est iterable
    # On normalise :
    if hasattr(missing, "missing_keys") and hasattr(missing, "unexpected_keys"):
        # cas IncompatibleKeys (rare selon version)
        mk = list(missing.missing_keys)
        uk = list(missing.unexpected_keys)
        return mk, uk
    # cas tuple (missing_keys, unexpected_keys)
    return list(missing), list(unexpected)


# ──────────────────────────────────────────────────────────────────────────────
# API principale
# ──────────────────────────────────────────────────────────────────────────────

def create_patchgan_from_manifest(
    manifest_path: str,
    device: str = "cpu",
    selected_tasks: Optional[List[str]] = None,
) -> Tuple[torch.nn.Module, Dict[str, List[str]]]:
    """
    Charge un MultiTaskPatchGAN à partir d'un manifest.json.

    - selected_tasks: si fourni, ne garde que ces tâches (PRUNE sur le modèle et sur le state_dict)
    - attn_use_se: lu du manifest (bool/str). Si False -> modèle construit sans SE et clés SE filtrées du ckpt.

    Retour:
      (model.eval(), tasks_labels_pruned)
    """
    manifest_path = str(manifest_path)
    mp = Path(manifest_path)
    if not mp.exists():
        raise RuntimeError(f"manifest introuvable: {manifest_path}")

    man = _read_json(mp)

    # 1) Tasks labels (UI)
    tasks: Dict[str, List[str]] = man.get("tasks") or man.get("labels") or {}
    if not isinstance(tasks, dict) or not tasks:
        raise RuntimeError("Manifest: champ 'tasks' (ou 'labels') manquant ou invalide.")

    # 2) paths
    root = mp.parent
    weights_rel = man.get("weights", "weights.pth")
    weights_path = str((root / weights_rel).resolve())
    if not Path(weights_path).exists():
        raise RuntimeError(f"weights introuvables: {weights_path}")

    # 3) extra / hyperparams
    extra: Dict[str, Any] = man.get("extra", {}) or {}
    patch_size = int(extra.get("patch_size", 31))
    attn_tau = float(extra.get("attn_tau", 0.7))
    attn_softmax_spatial = _as_bool(extra.get("attn_softmax_spatial", True), default=True)
    ablate_attention = _as_bool(extra.get("ablate_attention", False), default=False)

    # preprocess (utilisé par ton adapter, pas forcément ici)
    # preprocess = str(extra.get("preprocess", "tanh")).lower()

    # 4) PRUNE tasks (en conservant l'ordre du manifest)
    if selected_tasks:
        sel_set = set(selected_tasks)
        pruned_tasks = {t: lbls for t, lbls in tasks.items() if t in sel_set}
        if not pruned_tasks:
            # fallback: si on a tout filtré par erreur
            pruned_tasks = tasks
    else:
        pruned_tasks = tasks

    # 5) attn_use_se (du manifest) :
    #    - accepte bool, "false"/"true", "auto"
    attn_use_se_cfg = extra.get("attn_use_se", "auto")
    attn_use_se: bool
    if isinstance(attn_use_se_cfg, str) and attn_use_se_cfg.strip().lower() == "auto":
        # auto: tente de détecter depuis le checkpoint
        try:
            if checkpoint_has_se is not None:
                attn_use_se = bool(checkpoint_has_se(weights_path, device=device))
            else:
                # fallback simple: regarde s'il existe ".se." dans le state_dict
                sd_tmp = _load_checkpoint(weights_path, device=device)
                attn_use_se = _state_dict_has_any_key(sd_tmp, ".se.")
        except Exception:
            attn_use_se = False
    else:
        attn_use_se = _as_bool(attn_use_se_cfg, default=False)

    # 6) construire modèle selon tâches prunées + SE activé/désactivé
    tasks_dict = {t: len(lbls) for t, lbls in pruned_tasks.items()}

    model = MultiTaskPatchGAN(
        tasks_dict=tasks_dict,
        input_nc=3,
        ndf=64,
        norm="instance",
        patch_size=patch_size,
        device=device,
        attn_tau=attn_tau,
        attn_use_se=attn_use_se,  # IMPORTANT
        attn_softmax_spatial=attn_softmax_spatial,
        ablate_attention=ablate_attention,
    ).to(device).eval()

    # 7) charger checkpoint + filtrer en amont ce qui est absent selon manifest
    sd = _load_checkpoint(weights_path, device=device)

    # 7.a) si SE est désactivé dans le manifest -> on retire toutes les clés ".se."
    #      (ça évite le cas "missing keys ... se.mlp.*" si on construisait avec SE,
    #       et évite "unexpected" si ckpt contenait SE alors que modèle n'en a pas)
    if not attn_use_se:
        sd = _filter_state_dict(sd, drop_substrings=[".se."])

    # 7.b) PRUNE: retirer les task_heads des tâches non conservées
    sd = _prune_state_dict_for_tasks(sd, pruned_tasks=pruned_tasks, all_tasks=tasks)

    # 7.c) charge strictness :
    # - si pas de prune ET on a aligné SE via manifest -> strict=True est possible
    # - sinon strict=False
    strict_load = (len(pruned_tasks) == len(tasks))

    # Attention:
    # même sans PRUNE, certains checkpoints peuvent inclure d'autres clés (optimizer, ema, etc.)
    # donc on peut préférer strict=False pour être tolérant.
    # Ici: strict=True uniquement si pas de PRUNE ET (attn_use_se) cohérent.
    try:
        mk, uk = _load_into_model(model, sd, strict=strict_load)
        if mk or uk:
            # Si strict=True et mismatch -> PyTorch lève déjà une exception.
            # Si strict=False, on informe sans planter.
            if not strict_load:
                print(f"[PatchGAN][warn] load_state_dict non strict: missing={len(mk)} unexpected={len(uk)}")
    except RuntimeError as e:
        # Fallback: re-tenter en non strict + filtrage supplémentaire
        print(f"[PatchGAN][warn] strict load a échoué: {e}")

        # On retente en non strict en filtrant aussi tout ce qui n'existe pas dans model.state_dict
        model_keys = set(model.state_dict().keys())
        sd2 = {k: v for k, v in sd.items() if k in model_keys}

        mk, uk = _load_into_model(model, sd2, strict=False)
        if mk:
            print(f"[PatchGAN][warn] missing_keys (non strict) = {len(mk)}")
        if uk:
            print(f"[PatchGAN][warn] unexpected_keys (non strict) = {len(uk)}")

    return model, pruned_tasks


# ──────────────────────────────────────────────────────────────────────────────
# Petit helper si tu veux debug rapidement
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
