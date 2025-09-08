#loader_patchgan_from_manifest.py

import argparse

from torch.utils.data import Subset

import os
import json
import numpy as np
import itertools
from PIL import Image
import torch
import cv2
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
# -------------------------------------------------------------------
# Le dictionnaire de colormaps OpenCV pour Grad-CAM
# -------------------------------------------------------------------
colormap_dict = {
    'autumn': cv2.COLORMAP_AUTUMN,
    'bone': cv2.COLORMAP_BONE,
    'hot': cv2.COLORMAP_HOT,
    'afmhot': cv2.COLORMAP_TURBO,
    'inferno': cv2.COLORMAP_INFERNO,
    'jet': cv2.COLORMAP_JET,
    'turbo': cv2.COLORMAP_TURBO,
    'viridis': cv2.COLORMAP_VIRIDIS,
    'magma': cv2.COLORMAP_MAGMA,
    # Vous pouvez en ajouter d'autres si nécessaire
}

# -------------------------------------------------------------------
# 1) DATASET MULTI-TÂCHES
#    (basé sur vos JSON data_json / classes_json, comme dans vos scripts)
# -------------------------------------------------------------------
# =============================================================================
# Dataset multi-tâches
# =============================================================================
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, data_json, classes_json, transform=None, search_folder=None, find_images_by_sub_folder=None):
        with open(data_json, 'r') as f:
            self.data = json.load(f)
        with open(classes_json, 'r') as f:
            self.classes = json.load(f)
        self.transform = transform
        self.search_folder = search_folder
        self.find_images_by_sub_folder = find_images_by_sub_folder
        self.samples = []
        self.class_to_idx = {}
        self.task_classes = {}

        # Construire la correspondance des classes
        for task, class_list in self.classes.items():
            self.task_classes[task] = class_list
            self.class_to_idx[task] = {cls.lower(): idx for idx, cls in enumerate(class_list)}

        # Construire la liste des échantillons
        for folder, images in self.data.items():
            for img_name, img_info in images.items():
                orig_path = img_info['image_path']
                if self.search_folder:
                    image_identifier = os.path.join(self.search_folder, os.path.basename(orig_path))
                elif self.find_images_by_sub_folder:
                    # Extraire le sous-dossier juste avant le nom de l'image dans le chemin d'origine
                    # ex: .../training_13052025/sun/2025xxx.jpg -> subfolder = 'sun'
                    subfolder = os.path.basename(os.path.dirname(orig_path))
                    image_identifier = os.path.join(
                        self.find_images_by_sub_folder,
                        subfolder,
                        os.path.basename(orig_path)
                    )
                else:
                    image_identifier = orig_path

                labels = {}
                for task in self.classes:
                    label_val = img_info.get(task)
                    if label_val is not None:
                        lbl = label_val.lower()
                        labels[task] = self.class_to_idx[task].get(lbl)
                        if labels[task] is None:
                            print(f"Warning: label '{lbl}' for task '{task}' not found")
                    else:
                        labels[task] = None
                self.samples.append((image_identifier, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, labels = self.samples[idx]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, labels



# -------------------------------------------------------------------
# 2) MODULES D'ATTENTION & TÊTES MULTI-TÂCHES (PatchGAN) — VERSION ADAPTÉE
# -------------------------------------------------------------------


# ---------- Channel Attention (léger) ----------
class SE(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        hid = max(c // r, 1)
        self.mlp = nn.Sequential(
            nn.Linear(c, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, c), nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        wgt = self.mlp(x.mean((2, 3))).view(n, c, 1, 1)
        return x * wgt

# ---------- Head avec attention utile + ablation ----------
class TaskHeadImproved(nn.Module):
    """
    - (option) SE (channel attention)
    - Conv1x1 -> logits d'attention -> Softmax spatial(τ) -> A
    - Conv1x1 classes -> GWAP normalisé par A  => logits [N, K]
    - Ablation: A uniforme (GAP), pas d'attention apprise
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_se: bool = True,
                 tau: float = 0.7,
                 use_softmax_spatial: bool = True,
                 ablate_attention: bool = False):
        super().__init__()
        self.use_se = use_se
        self.tau = tau
        self.use_softmax_spatial = use_softmax_spatial
        self.ablate_attention = ablate_attention

        self.se = SE(in_channels) if use_se else nn.Identity()
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.cls_conv  = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, feat: torch.Tensor):
        N, C, H, W = feat.shape
        x = self.se(feat) if self.use_se and not self.ablate_attention else feat

        if self.ablate_attention:
            A = torch.ones((N, 1, H, W), device=feat.device, dtype=feat.dtype) / float(H * W)
            M = self.cls_conv(x)                 # [N, K, H, W]
            logits = (M * A).sum(dim=(2, 3))     # GAP équivalent
            return logits, A

        a = self.attn_conv(x).view(N, 1, H * W)
        if self.use_softmax_spatial:
            A = torch.softmax(a / self.tau, dim=-1).view(N, 1, H, W)
        else:
            A = torch.sigmoid(a).view(N, 1, H, W)
            A = A / (A.sum(dim=(2, 3), keepdim=True) + 1e-6)

        M = self.cls_conv(x)                     # [N, K, H, W]
        num = (M * A).sum(dim=(2, 3))            # [N, K]
        den = (A.sum(dim=(2, 3)) + 1e-6)         # [N, 1]
        logits = num / den
        return logits, A

# -------------------------------------------------------------------
# 3) PATCHGAN TRONQUÉ MULTI-TÂCHES — VERSION ADAPTÉE (compat arrière)
# -------------------------------------------------------------------
class MultiTaskPatchGAN(nn.Module):
    """
    - Tronc PatchGAN (InstanceNorm affine=True recommandé pour style)
    - Une tête par tâche avec attention améliorée
    - Compat arrière:
        * model(x) -> {task: logits Tensor} (comme avant)
        * model(x, return_full=True) -> {task: {'logits': Tensor, 'attn': Tensor}}
        * return_embeddings/return_task_embeddings disponibles (comme avant)
    """
    def __init__(self, tasks_dict, input_nc=3, ndf=64, norm="instance", patch_size=70, device='cpu',
                 attn_tau=0.7, attn_use_se=True, attn_softmax_spatial=True, ablate_attention=False):
        super().__init__()
        self.device = device
        self.tasks_dict = tasks_dict
        self.ablate_attention = ablate_attention

        if norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
        else:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)

        layers = []
        num_filters = ndf
        kernel_size = 4
        padding = 1
        stride = 2
        receptive_field_size = float(patch_size)
        in_nc = input_nc

        while receptive_field_size > 4 and num_filters <= 512:
            layers += [
                nn.Conv2d(in_nc, num_filters, kernel_size, stride, padding),
                norm_layer(num_filters),
                nn.LeakyReLU(0.2, True)
            ]
            in_nc = num_filters
            num_filters *= 2
            receptive_field_size /= stride

        final_c = num_filters
        layers += [
            nn.Conv2d(in_nc, final_c, kernel_size, 1, padding),
            norm_layer(final_c),
            nn.LeakyReLU(0.2, True)
        ]

        self.trunk = nn.Sequential(*layers).to(self.device)

        self.task_heads = nn.ModuleDict()
        for task_name, nb_cls in tasks_dict.items():
            self.task_heads[task_name] = TaskHeadImproved(
                in_channels=final_c, out_channels=nb_cls,
                use_se=attn_use_se,
                tau=attn_tau,
                use_softmax_spatial=attn_softmax_spatial,
                ablate_attention=ablate_attention
            ).to(self.device)

    @torch.no_grad()
    def _embeddings_from_feats(self, feats, flatten=True):
        if flatten:
            N = feats.size(0)
            return feats.view(N, -1).cpu()
        return feats.mean(dim=[2, 3]).cpu()  # global feature

    def forward(self, x, return_embeddings=False, return_task_embeddings=False, return_full=False):
        x = x.to(self.device)
        feats = self.trunk(x)  # [N, C, H, W]

        if return_task_embeddings:
            outputs = {}
            task_embeddings = {}
            for task_name, head in self.task_heads.items():
                logits, _A = head(feats)
                outputs[task_name] = logits
                task_embeddings[task_name] = feats.mean(dim=[2,3]).cpu()  # embedding partagé
            return outputs, task_embeddings

        if return_embeddings:
            N, C, H, W = feats.shape
            feats_flat = feats.view(N, -1).cpu()
            return feats_flat

        # sortie normale
        if return_full:
            outputs = {}
            for t, head in self.task_heads.items():
                logits, A = head(feats)
                outputs[t] = {'logits': logits, 'attn': A}
            return outputs
        else:
            outputs = {}
            for t, head in self.task_heads.items():
                logits, _A = head(feats)
                outputs[t] = logits
            return outputs

# -------------------------------------------------------------------
# 4) MODELE POUR GRADCAM (sélectionner la tâche) — ADAPTÉ
# -------------------------------------------------------------------
class TaskSpecificModel(nn.Module):
    """
    Wrap pour extraire le logits d'une tâche spécifique (compat nouvelles sorties).
    """
    def __init__(self, model, task_name):
        super().__init__()
        self.model = model
        self.task_name = task_name

    def forward(self, x):
        outs = self.model(x, return_full=False)  # compat: dict[str]->Tensor logits
        return outs[self.task_name]  # [N, nb_classes]

# -------------------------------------------------------------------
# 4bis) UTILITAIRES TEST (chargement & params)
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# 6) CHARGEMENT DU MODELE
# -------------------------------------------------------------------
def load_model_weights(model: nn.Module, path: str, device: torch.device, strict: bool = True):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith('module.') else k  # retire DataParallel si présent
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    if missing:
        print(f"[load] Missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[load] Unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")
    print(f"[load] strict={strict} -> OK (si pas d'exception)")


def checkpoint_has_se(path, device='cpu'):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict):
        sd = sd.get('model', sd.get('state_dict', sd))
    return any('.se.mlp.' in k for k in sd.keys())










@torch.no_grad()
def compute_attn_embeddings_per_task_with_paths(model, loader, device, tasks_json):
    """
    Extrait des embeddings 'par tâche' pondérés par l'attention:
      e_t = sum_{h,w} (feat * A_t) / sum_{h,w} A_t  -> vecteur [C]
    Renvoie:
      - embeddings_data: dict(task -> np.array [N_t, C])
      - labels_data:     dict(task -> np.array [N_t])
      - img_paths_data:  dict(task -> List[str]) (aligné avec embeddings/labels de cette tâche)
    Seuls les échantillons avec un label défini pour la tâche sont inclus.
    """
    model.eval()
    embeddings_data = {t: [] for t in tasks_json.keys()}
    labels_data     = {t: [] for t in tasks_json.keys()}
    paths_data      = {t: [] for t in tasks_json.keys()}

    # hook pour récupérer la sortie du trunk
    feats_cache = []
    def hook_fn(_m, _inp, out):
        feats_cache.append(out.detach())
    h = model.trunk.register_forward_hook(hook_fn)

    # chemins d'images dans l'ordre du loader (shuffle=False)
    all_paths = _get_loader_paths(loader)
    ptr = 0

    for imgs, labels in loader:
        bsz = imgs.size(0)
        imgs = imgs.to(device, non_blocking=True)

        # forward complet pour récupérer les cartes d'attention
        outs = model(imgs, return_full=True)  # dict[task] -> {'logits':..., 'attn':...}
        feats = feats_cache.pop(0)            # [N,C,H,W] sortie du trunk
        N, C, H, W = feats.shape

        # par tâche: projeter via A_t
        for task in tasks_json.keys():
            A = outs[task]['attn']            # [N,1,H,W]
            lbl_t = labels[task]              # liste de labels (tenseurs ou None)
            # on ne garde que les échantillons avec label défini
            for i in range(N):
                # indice global → chemin image
                img_path = all_paths[ptr + i]
                lab = lbl_t[i]
                if lab is None:
                    continue
                lab = int(lab) if not torch.is_tensor(lab) else int(lab.item())
                # embedding attention-pondéré
                Ai = A[i]                     # [1,H,W]
                Fi = feats[i]                 # [C,H,W]
                num = (Fi * Ai).sum(dim=(1,2))         # [C]
                den = Ai.sum(dim=(1,2)).clamp_min(1e-6)  # [1]
                emb = (num / den).cpu().numpy()        # [C]
                embeddings_data[task].append(emb)
                labels_data[task].append(lab)
                paths_data[task].append(img_path)

        ptr += bsz

    # convert lists -> arrays
    for t in tasks_json.keys():
        if len(embeddings_data[t]) > 0:
            embeddings_data[t] = np.stack(embeddings_data[t], axis=0)
            labels_data[t]     = np.array(labels_data[t])
        else:
            embeddings_data[t] = np.empty((0, 0), dtype=np.float32)
            labels_data[t]     = np.array([], dtype=np.int64)

    h.remove()
    return embeddings_data, labels_data, paths_data

# ------------------------------ MAIN (tous les modes, t-SNE maj) ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test d'un PatchGAN Multi-tâches avec divers modes")
    # Base / chemins
    parser.add_argument('--data', type=str, help='Chemin vers le JSON du dataset (obligatoire pour classifier/clustering/tsne/inference)')
    parser.add_argument('--build_classifier', type=str, required=True, help='Chemin vers le JSON de description des tâches/classes')
    parser.add_argument('--config_path', type=str, required=True, help="Chemin vers le JSON d'hyperparamètres du modèle")
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers le fichier .pth du modèle entraîné')
    parser.add_argument('--save_dir', default='results', type=str, help='Répertoire de sortie')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--tensorboard', action='store_true')

    # Modes
    parser.add_argument('--mode',
                        choices=['classifier', 'tsne', 'tsne_interactive', 'camera', 'clustering',
                                 'folder', 'benchmark_patchGAN_Gram', 'watch_folder', 'inference'],
                        default='classifier')

    # Explainability / viz
    parser.add_argument('--visualize_gradcam', action='store_true')
    parser.add_argument('--save_gradcam_images', action='store_true')
    parser.add_argument('--gradcam_task', type=str, default=None)
    parser.add_argument('--colormap', type=str, default='hot')
    parser.add_argument('--integrated_gradients', action='store_true')
    parser.add_argument('--integrated_gradients_task', type=str, default=None)

    # Inference / mesure
    parser.add_argument('--prob_threshold', default=0.5, type=float)
    parser.add_argument('--measure_time', action='store_true')
    parser.add_argument('--save_test_images', action='store_true')
    parser.add_argument('--count_params', action='store_true')

    # Données
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--test_images_folder', type=str)
    parser.add_argument('--test_following_task', type=str, default=None)
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--search_folder', type=str, default=None)
    parser.add_argument('--find_images_by_sub_folder', type=str, default=None)

    # t-SNE / clustering
    parser.add_argument('--colors', nargs='+', default=None, metavar='COLORS')
    parser.add_argument('--per_task_tsne', action='store_true', help='(legacy) t-SNE par tâche avec embeddings classiques')
    parser.add_argument('--per_task', action='store_true', help='t-SNE par tâche avec embeddings pondérés par ATTENTION')
    parser.add_argument('--clustering_class', type=str)
    parser.add_argument('--min_cluster_size', type=int, nargs='+', default=[10, 15, 20])
    parser.add_argument('--min_samples', type=int, nargs='+', default=[5, 10])

    # Caméra
    parser.add_argument('--kalman_filter', action='store_true')
    parser.add_argument('--camera_index', type=int, default=0)
    parser.add_argument('--save_camera_video', action='store_true')

    # Benchmark
    parser.add_argument('--benchmark_folder', type=str)
    parser.add_argument('--benchmark_mapping', type=str)
    parser.add_argument('--roc_output', type=str, default='roc_curves')
    parser.add_argument('--auto_mapping', action='store_true')

    # Watch folders
    parser.add_argument('--watch_folders', type=str, default=None)
    parser.add_argument('--poll_intervals', type=str, default=None)
    parser.add_argument('--save_dir_to_canon', default=None, type=str)
    parser.add_argument('--eval_annotations', action='store_true')
    parser.add_argument('--annotations_folders', type=str, default=None)
    parser.add_argument('--truth_mapping', type=str, default=None)
    parser.add_argument('--metry_every', default=50, type=int)

    # Attention (construction/ablation)
    parser.add_argument('--ablate_attention', action='store_true')
    parser.add_argument('--attn_use_se', action='store_true')
    parser.add_argument('--attn_tau', type=float, default=0.7)
    parser.add_argument('--attn_no_softmax', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    writer = (SummaryWriter(log_dir=os.path.join(args.save_dir, 'TensorBoard'))
              if args.tensorboard else None)

    # Hyperparams & tâches
    with open(args.config_path, 'r') as f:
        best_config = json.load(f)
    with open(args.build_classifier, 'r') as f:
        tasks_json = json.load(f)
    tasks_dict = {task_name: len(class_list) for task_name, class_list in tasks_json.items()}
    print(f"Nombre de tâches: {len(tasks_dict)}")
    for t, n in tasks_dict.items():
        print(f"  Tâche '{t}': {n} classes")

    # Construction du modèle compatible avec le ckpt
    patch_size = best_config.get('patch_size', 70)
    attn_tau = float(best_config.get('attn_tau', args.attn_tau))
    attn_softmax_spatial = bool(best_config.get('attn_softmax_spatial', not args.attn_no_softmax))
    ckpt_has_se = checkpoint_has_se(args.model_path, device)
    attn_use_se = True if ckpt_has_se else bool(best_config.get('attn_use_se', args.attn_use_se))
    print(f"[build] ckpt_has_se={ckpt_has_se} | attn_use_se(model)={attn_use_se} | ablate={args.ablate_attention}")

    model = MultiTaskPatchGAN(
        tasks_dict=tasks_dict,
        input_nc=3, ndf=64, norm="instance",
        patch_size=patch_size, device=device,
        attn_tau=attn_tau,
        attn_use_se=attn_use_se,
        attn_softmax_spatial=attn_softmax_spatial,
        ablate_attention=args.ablate_attention
    ).to(device)

    load_model_weights(model, args.model_path, device, strict=True)








# -------------------------------------------------------------------
# 7) CALCUL EMBEDDINGS (pour t-SNE, clustering, etc.)
# -------------------------------------------------------------------
def compute_embeddings_with_paths(model, loader, device, tasks_json, per_task_tsne=False):
    model.eval()
    if per_task_tsne:
        embeddings_dict = {tname: [] for tname in tasks_json.keys()}
        labels_dict = {tname: [] for tname in tasks_json.keys()}
        img_paths_dict = {tname: [] for tname in tasks_json.keys()}
    else:
        all_embeddings = []
        all_labels = []
        img_paths = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            outputs, task_emb = model(inputs, return_task_embeddings=True)
            batch_size = inputs.size(0)
            if isinstance(loader.dataset, Subset):
                indices = loader.dataset.indices[batch_idx * loader.batch_size : batch_idx * loader.batch_size + batch_size]
                batch_img_paths = [loader.dataset.dataset.samples[idx][0] for idx in indices]
            else:
                batch_img_paths = [loader.dataset.samples[idx][0] for idx in range(batch_idx * loader.batch_size,
                                        batch_idx * loader.batch_size + batch_size)]
            if per_task_tsne:
                for tname in tasks_json.keys():
                    emb_batch = task_emb[tname].cpu().numpy()
                    label_batch = labels[tname].clone()
                    label_batch[label_batch < 0] = -1
                    for i in range(batch_size):
                        embeddings_dict[tname].append(emb_batch[i])
                        labels_dict[tname].append(int(label_batch[i].item()) if label_batch[i].item() >= 0 else -1)
                        img_paths_dict[tname].append(batch_img_paths[i])
            else:
                first_task = list(tasks_json.keys())[0]
                emb_batch = task_emb[first_task].cpu().numpy()
                label_batch = labels[first_task].clone()
                label_batch[label_batch < 0] = -1
                all_embeddings.append(emb_batch)
                all_labels.append(label_batch.cpu().numpy())
                img_paths.extend(batch_img_paths)
    if per_task_tsne:
        for tname in embeddings_dict.keys():
            embeddings_dict[tname] = np.array(embeddings_dict[tname])
            labels_dict[tname] = np.array(labels_dict[tname])
        return embeddings_dict, labels_dict, img_paths_dict
    else:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_embeddings, all_labels, img_paths

