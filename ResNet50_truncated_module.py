# ResNet50_truncated_module.py — sélection/élagage de tâches + chargement filtré
import math
import re
from typing import Dict, List, Optional, Union, Iterable

import torch
import torch.nn as nn


def load_best_model(model,
                    filepath,
                    strict_backbone: bool = True,
                    verbose: bool = True,
                    keep_tasks: Optional[Iterable[str]] = None):
    """
    Chargement partiel robuste avec remap + filtrage par tâches:
      - supporte 'module.' ; 'backbone.'/ 'truncated_encoder.' / ResNet brut
      - remap des classifieurs: 'classifiers.classifier_X.weight/bias'
        → dernière couche linéaire du MLP: 'classifiers.classifier_X.{last_idx}.weight/bias'
      - si shapes diffèrent, copie partielle (jusqu'à min sur chaque axe)
      - keep_tasks: liste des tâches à charger (les autres têtes sont ignorées)
    """
    ckpt = torch.load(filepath, map_location=model.device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # 0) retire 'module.'
    ckpt = { (k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items() }

    # 1) détecte préfixe features dans le ckpt
    has_backbone   = any(k.startswith("backbone.")          for k in ckpt)
    has_truncated  = any(k.startswith("truncated_encoder.") for k in ckpt)
    if has_backbone:
        feat_prefix_ckpt = "backbone."
    elif has_truncated:
        feat_prefix_ckpt = "truncated_encoder."
    else:
        feat_prefix_ckpt = None  # ResNet brut

    # préfixe attendu ici
    feat_prefix_model = ("truncated_encoder."
                         if any(k.startswith("truncated_encoder.") for k in model.state_dict())
                         else "backbone.")

    remapped = {}

    # 2) remap des features (backbone/truncated_encoder/ResNet brut)
    if feat_prefix_ckpt is None:
        # ckpt au format ResNet brut (conv1, bn1, layer1..)
        root2idx = {
            "conv1": 0, "bn1": 1, "relu": 2, "maxpool": 3,
            "layer1": 4, "layer2": 5, "layer3": 6, "layer4": 7,
        }
        cur_children = list(model.truncated_encoder.children())
        for k, v in ckpt.items():
            root = k.split('.')[0]
            if root not in root2idx:
                continue  # ignore avgpool/fc
            idx = root2idx[root]
            if idx >= len(cur_children):
                continue
            new_k = f"{feat_prefix_model}{idx}{k[len(root):]}"
            remapped[new_k] = v
    else:
        if feat_prefix_ckpt == feat_prefix_model:
            remapped = dict(ckpt)
        else:
            cut = len(feat_prefix_ckpt)
            for k, v in ckpt.items():
                if k.startswith(feat_prefix_ckpt):
                    new_k = f"{feat_prefix_model}{k[cut:]}"
                    remapped[new_k] = v
                else:
                    remapped[k] = v  # têtes etc.

    # 3) REMAP spécial des classifieurs (Linear → MLP[...Linear])
    #    - si ckpt a 'classifiers.classifier_X.weight/bias' (Linear pur)
    #      on redirige vers la DERNIÈRE couche Linear du Sequential.
    #    - si le modèle a des têtes linéaires pures, ce remap ne change rien.
    cls_last_linear_idx = {}
    for cls_name, mod in model.classifiers.items():
        last_idx = None
        if isinstance(mod, nn.Sequential):
            for i, m in enumerate(mod):
                if isinstance(m, nn.Linear):
                    last_idx = i
        elif isinstance(mod, nn.Linear):
            last_idx = None
        cls_last_linear_idx[cls_name] = last_idx

    converted = dict(remapped)  # copy
    pat_simple = re.compile(r'^classifiers\.(classifier_[^\.]+)\.(weight|bias)$')
    for k, v in list(remapped.items()):
        m = pat_simple.match(k)
        if m:
            cls_name, wb = m.group(1), m.group(2)  # ex: classifier_Weather_Type, weight
            last_idx = cls_last_linear_idx.get(cls_name, None)
            if last_idx is None:
                new_k = f"classifiers.{cls_name}.{wb}"
            else:
                new_k = f"classifiers.{cls_name}.{last_idx}.{wb}"
            converted[new_k] = v
            if verbose and new_k != k:
                print(f"[remap] {k}  →  {new_k}")

    remapped = converted

    # 3bis) FILTRAGE par tâches (attentions.* et classifiers.*) si keep_tasks fourni
    if keep_tasks is not None:
        keep_set = { _task_to_key(t) for t in keep_tasks }  # noms avec underscores
        filtered = {}
        pat_attn = re.compile(r'^attentions\.attention_([^\.]+)\.')
        pat_cls  = re.compile(r'^classifiers\.classifier_([^\.]+)\.')
        for k, v in remapped.items():
            if k.startswith("attentions.attention_"):
                m = pat_attn.match(k)
                if m and m.group(1) in keep_set:
                    filtered[k] = v
                else:
                    # on drop
                    pass
            elif k.startswith("classifiers.classifier_"):
                m = pat_cls.match(k)
                if m and m.group(1) in keep_set:
                    filtered[k] = v
            else:
                filtered[k] = v  # backbone/truncated_encoder/others
        remapped = filtered

    # 4) filtrage/redimensionnement éventuel (copie partielle sur mismatch)
    new_state, to_load = model.state_dict(), {}
    for k, v in remapped.items():
        if k not in new_state:
            if verbose and (k.startswith("classifiers.") or k.startswith("attentions.")):
                print(f"[skip] {k} absent du modèle courant")
            continue
        if v.shape == new_state[k].shape:
            to_load[k] = v
        else:
            tgt = new_state[k].clone()
            slices = tuple(slice(0, min(a, b)) for a, b in zip(v.shape, tgt.shape))
            tgt[slices] = v[slices]
            to_load[k] = tgt
            if verbose:
                print(f"[resize] {k}: {v.shape} → {tgt.shape}")

    # 5) vérification stricte du backbone si demandé
    if strict_backbone:
        missing = [k for k in new_state
                   if k.startswith(feat_prefix_model) and k not in to_load]
        if missing:
            raise RuntimeError(f"Backbone keys manquantes ({len(missing)}). Ex: {missing[:10]}")

    msg = model.load_state_dict(to_load, strict=False)  # strict False pour tolérer l'élagage
    if verbose:
        print(f"✔ {len(to_load)} tenseurs chargés (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")
    model.to(model.device)


def _task_to_key(task: str) -> str:
    """'Weather Type' → 'Weather_Type' (clé utilisée dans les ModuleDicts)."""
    return task.replace(' ', '_')


# --- Nouvelle attention : query par tâche sur tokens spatiaux [B,HW,C] ---
class TaskAttentionHead(nn.Module):
    """Attention 'query par tâche' sur tokens spatiaux. Entrée: [B,HW,C] → Sortie: [B,C]."""
    def __init__(self, dim: int, token_dim: Optional[int] = None):
        super().__init__()
        d = token_dim or dim
        self.q = nn.Parameter(torch.randn(1, 1, d))   # requête apprise
        self.proj = nn.Linear(dim, d, bias=False)     # projette tokens -> d
        self.out  = nn.Linear(d, dim, bias=False)     # d -> C

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, HW, C]
        T = self.proj(tokens)                                    # [B, HW, d]
        q = self.q.expand(T.size(0), -1, -1)                     # [B, 1, d]
        attn = torch.softmax((q @ T.transpose(1, 2)) / math.sqrt(T.size(-1)), dim=-1)  # [B,1,HW]
        h = (attn @ T).squeeze(1)                                # [B, d]
        return self.out(h)                                       # [B, C]


class MultiHeadAttentionPerTaskModel(nn.Module):
    """
    ResNet tronqué (sans avgpool/fc) + (optionnel) attention par tâche.
    - use_attention=True  : tokens [B,HW,C] -> TaskAttentionHead -> MLP par tâche
    - use_attention=False : ablation, GAP [B,C] -> MLP par tâche
    Supporte le retour des embeddings (par tâche et/ou partagés) pour t-SNE.

    Nouveautés:
      - forward(..., only=[...]) : n'inférer que sur un sous-ensemble de tâches
      - prune_to_tasks([...])    : supprimer définitivement les têtes non gardées
    """
    def __init__(self,
                 base_encoder: nn.Module,
                 truncate_after_layer: int,
                 tasks: Dict[str, Union[List, int]],
                 device: Union[str, torch.device] = "cpu",
                 use_attention: bool = True,
                 attn_token_dim: Optional[int] = None,
                 cls_hidden_dims: Optional[List[int]] = None,
                 cls_num_layers: int = 0):
        super().__init__()
        self.device = torch.device(device)
        # Normalize tasks -> counts
        self.tasks = {t: (len(v) if isinstance(v, (list, tuple)) else int(v)) for t, v in tasks.items()}
        self.use_attention = use_attention

        # 1) Tronque le backbone sans avgpool/fc pour conserver HxW>1
        enc_layers = list(base_encoder.children())[:-2]  # conv1..layer4
        truncate_after_layer = max(1, min(truncate_after_layer, len(enc_layers)))
        self.truncated_encoder = nn.Sequential(*enc_layers[:truncate_after_layer]).to(self.device)

        # 2) Infère C (nb de canaux)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            feat  = self.truncated_encoder(dummy)  # [1,C,H,W]
            C = feat.shape[1]
        self.num_features = C

        # 3) Heads attentionnelles (optionnelles) + classifieurs MLP par tâche
        self.attentions  = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        cls_hidden_dims = cls_hidden_dims or []

        for task, n_cls in self.tasks.items():
            key = _task_to_key(task)
            if self.use_attention:
                self.attentions[f"attention_{key}"] = TaskAttentionHead(C, attn_token_dim)
            # MLP: C -> hidden_dims[:cls_num_layers] -> n_cls
            hds = cls_hidden_dims[:cls_num_layers]
            dims = [C] + hds
            layers = []
            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(inplace=True)]
            layers.append(nn.Linear(dims[-1], n_cls))
            self.classifiers[f"classifier_{key}"] = nn.Sequential(*layers)

    # ---- Gestion des sous-ensembles de tâches ----
    def _tasks_to_run(self, only: Optional[Iterable[str]]) -> List[str]:
        if only is None:
            return list(self.tasks.keys())
        # on garde l'ordre fourni par 'only' mais on filtre
        avail = set(self.tasks.keys())
        return [t for t in only if t in avail]

    def prune_to_tasks(self, keep_tasks: Iterable[str]):
        """
        Supprime définitivement les têtes non gardées et met à jour self.tasks.
        """
        keep = set(keep_tasks or [])
        # supprimer des ModuleDicts
        for task in list(self.tasks.keys()):
            if task not in keep:
                key = _task_to_key(task)
                cls_key = f"classifier_{key}"
                att_key = f"attention_{key}"
                if cls_key in self.classifiers:
                    del self.classifiers[cls_key]
                if att_key in self.attentions:
                    del self.attentions[att_key]
                del self.tasks[task]

    def forward(self,
                x: torch.Tensor,
                *,
                return_task_embeddings: bool = False,
                return_shared_embedding: bool = False,
                only: Optional[List[str]] = None):
        """
        Retourne:
          - logits_dict: dict{task -> logits}
          - (+) task_embeds: dict{task -> [B,C]} si return_task_embeddings
          - (+) shared: Tensor [B,C] si return_shared_embedding
        Paramètre:
          - only: liste des tâches à inférer (sous-ensemble)
        """
        x = x.to(self.device)
        feat = self.truncated_encoder(x)                # [B,C,H,W]
        B, C, H, W = feat.shape

        # Embedding 'partagé' utilisé pour t-SNE global: on prend GAP
        shared = feat.mean(dim=(2, 3))                  # [B,C]

        logits_dict, task_embeds = {}, {}
        tasks_to_run = self._tasks_to_run(only)

        if self.use_attention:
            tokens = feat.flatten(2).transpose(1, 2)    # [B,HW,C]
            for task in tasks_to_run:
                key = _task_to_key(task)
                attn = self.attentions.get(f"attention_{key}", None)
                cls  = self.classifiers.get(f"classifier_{key}", None)
                if attn is None or cls is None:
                    continue
                h = attn(tokens)                        # [B,C] embedding par tâche
                logits_dict[task] = cls(h)
                task_embeds[task] = h
        else:
            # Ablation: pas d'attention, GAP → MLP par tâche
            for task in tasks_to_run:
                key = _task_to_key(task)
                cls = self.classifiers.get(f"classifier_{key}", None)
                if cls is None:
                    continue
                logits_dict[task] = cls(shared)
                task_embeds[task] = shared

        if return_task_embeddings and return_shared_embedding:
            return logits_dict, task_embeds, shared
        if return_task_embeddings:
            return logits_dict, task_embeds
        if return_shared_embedding:
            return logits_dict, shared
        return logits_dict


class TaskSpecificModel(nn.Module):
    """Wrapper pour n’inférer que sur une tâche donnée (conserve le backbone partagé)."""
    def __init__(self, model, task_name):
        super().__init__()
        self.model = model
        self.task_name = task_name

    def forward(self, x):
        outputs = self.model(x, only=[self.task_name])
        return outputs[self.task_name]
