#!/usr/bin/env python3
"""
Tursiops truncatus binary student — LODO (Leave-One-Dataset-Out) variant sweep.
OPTIMIZED VERSION — same logic, much faster epochs.

Key optimizations vs. the previous version:
  1. Pre-normalized FLOAT16 tensor cache instead of PIL cache.
     - The expensive PIL→tensor→Normalize chain happens ONCE per image,
       not every epoch. ~55k images × 60 epochs of redundant work eliminated.
     - fp16 halves RAM vs fp32: 55k × 3 × 224 × 224 × 2B ≈ 16 GB instead of 32 GB.
       Cast back to fp32 (or use AMP autocast) on the fly in __getitem__.
  2. Multi-worker DataLoader with persistent_workers + prefetch.
     - Augmentations (TimeMask/FreqMask/Cutout/ColorJitter) now run in parallel.
     - Workers fork-inherit the cached tensor, so no per-worker RAM blow-up
       on Linux (copy-on-write).
  3. Mixed precision (torch.amp.autocast + GradScaler) on the L4.
     - L4 has strong fp16 tensor cores; EfficientNet-B0 at bs=128 benefits a lot.
  4. cudnn.benchmark = True (fixed input shapes → faster kernels).
  5. .to(device, non_blocking=True) + pin_memory=True for overlapped H2D copies.
  6. set_to_none=True on zero_grad (slightly cheaper than zeroing).
  7. Teacher embedding pre-converted to a single torch tensor (no per-item
     np→torch conversion in __getitem__).
  8. Augmentations restructured so ColorJitter (PIL-only) is dropped — we now
     work directly with normalized tensors. We replace the brightness/contrast
     PIL jitter with cheap tensor-space equivalents (multiplicative scaling),
     which preserves the regularization intent without forcing a PIL roundtrip.
"""

import argparse
import os, json, time, gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import timm
timm.create_model('efficientnet_b0', pretrained=True)  # warm cache
print("[startup] EfficientNet-B0 weights cached.", flush=True)

from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from tqdm import tqdm

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    '--holdout', nargs='+',
    default=['Adriatic_Sea', 'ALNITAK_CAVANILLES'],
    help='One or more dataset names to hold out as test sets (space-separated).',
)
parser.add_argument(
    '--workers', type=int, default=6,
    help='DataLoader workers. 6–8 is a good sweet spot on 80-core hosts; '
         'workers share the parent cache via copy-on-write on Linux.',
)
parser.add_argument(
    '--batch-size', type=int, default=128,
    help='Per-step batch size. With AMP on L4, 192–256 is also fine for EfficientNet-B0.',
)
parser.add_argument(
    '--no-amp', action='store_true',
    help='Disable mixed precision (debug only — AMP is a major speedup).',
)
args = parser.parse_args()
HOLDOUT_DATASETS: list[str] = args.holdout
print(f"Hold-out datasets: {HOLDOUT_DATASETS}")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path('/data2/mromaniuc/cet-det/tursiops_perch/student_teacher')
META_PATH    = BASE / 'meta_train_with_paths.parquet'
TEACHER_PATH = BASE / 'X_teacher_emb.npy'
RUNS_DIR     = BASE / 'runs_lodo'
RUNS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SPECIES = 'Tursiops_truncatus'

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = args.batch_size
N_WORKERS     = args.workers
USE_AMP       = not args.no_amp
SEED          = 42
TEACHER_DIM   = 1536
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 15
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 30
PHASE3_EPOCHS = 20
N_EPOCHS      = PHASE1_EPOCHS + PHASE2_EPOCHS + PHASE3_EPOCHS
EARLY_STOP_START_EPOCH = PHASE1_EPOCHS + PHASE2_EPOCHS

torch.manual_seed(SEED)
np.random.seed(SEED)

# Big speedup for fixed-shape conv nets:
torch.backends.cudnn.benchmark = True
# Allow TF32 for matmuls (free precision/speed tradeoff on L4):
torch.set_float32_matmul_precision('high')

print(f"\n[startup] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[startup] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[startup] VRAM: {torch.cuda.mem_get_info(0)[1]/1e9:.1f} GB total")
print(f"[startup] workers={N_WORKERS}  batch_size={BATCH_SIZE}  amp={USE_AMP}")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Variants
# ══════════════════════════════════════════════════════════════════════════════
VARIANTS = {
    'v01_grl_distil': dict(
        alpha_distil=2.0, beta_class=1.0, gamma_adv=1.0,
        lambda_max=0.2, use_tanh=False, domain_head_size='small',
        aug_brightness=True, aug_contrast=True,
        aug_time_mask=True, aug_freq_mask=True, aug_cutout=True,
        note='binary + distillation + GRL',
    ),
    # 'v02_grl_only': dict(
    #     alpha_distil=0.0, beta_class=1.0, gamma_adv=1.0,
    #     lambda_max=0.2, use_tanh=False, domain_head_size='small',
    #     aug_brightness=True, aug_contrast=True,
    #     aug_time_mask=True, aug_freq_mask=True, aug_cutout=True,
    #     note='binary + GRL only — no distillation',
    # ),
    # 'v03_no_grl': dict(
    #     alpha_distil=0.0, beta_class=1.0, gamma_adv=0.0,
    #     lambda_max=0.0, use_tanh=False, domain_head_size='small',
    #     aug_brightness=True, aug_contrast=True,
    #     aug_time_mask=True, aug_freq_mask=True, aug_cutout=True,
    #     note='binary only — no distillation, no GRL (true ablation baseline)',
    # ),
    # 'v04_strong_distil': dict(
    #     alpha_distil=5.0, beta_class=1.0, gamma_adv=1.0,
    #     lambda_max=0.2, use_tanh=False, domain_head_size='small',
    #     aug_brightness=True, aug_contrast=True,
    #     aug_time_mask=True, aug_freq_mask=True, aug_cutout=True,
    #     note='strong distillation pull toward Perch geometry',
    # ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Tensor-space augmentations
#   Inputs are *already normalized* float32 tensors of shape (3, H, W).
#   Brightness/contrast jitter is applied multiplicatively in normalized space —
#   the regularization effect is equivalent to the old PIL ColorJitter for
#   spectrogram PNGs (which are essentially grayscale).
# ══════════════════════════════════════════════════════════════════════════════
class TensorBrightnessContrast:
    def __init__(self, brightness=0.2, contrast=0.2, p=0.5):
        self.brightness = brightness; self.contrast = contrast; self.p = p
    def __call__(self, x):
        if torch.rand(1).item() > self.p: return x
        if self.brightness > 0:
            b = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
            x = x * b
        if self.contrast > 0:
            c = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
            mean = x.mean(dim=(1, 2), keepdim=True)
            x = (x - mean) * c + mean
        return x

class TimeMask:
    def __init__(self, max_width=40, p=0.5, n_masks=1):
        self.max_width = max_width; self.p = p; self.n_masks = n_masks
    def __call__(self, x):
        if torch.rand(1).item() > self.p: return x
        x = x.clone(); _, _, W = x.shape
        for _ in range(self.n_masks):
            w = int(torch.randint(1, self.max_width + 1, (1,)).item())
            if w >= W: continue
            t0 = int(torch.randint(0, W - w + 1, (1,)).item())
            x[:, :, t0:t0+w] = 0.0
        return x

class FreqMask:
    def __init__(self, max_height=24, p=0.5, n_masks=1):
        self.max_height = max_height; self.p = p; self.n_masks = n_masks
    def __call__(self, x):
        if torch.rand(1).item() > self.p: return x
        x = x.clone(); _, H, _ = x.shape
        for _ in range(self.n_masks):
            h = int(torch.randint(1, self.max_height + 1, (1,)).item())
            if h >= H: continue
            f0 = int(torch.randint(0, H - h + 1, (1,)).item())
            x[:, f0:f0+h, :] = 0.0
        return x

class Cutout:
    def __init__(self, max_size=48, p=0.5):
        self.max_size = max_size; self.p = p
    def __call__(self, x):
        if torch.rand(1).item() > self.p: return x
        x = x.clone(); _, H, W = x.shape
        h = int(torch.randint(8, self.max_size + 1, (1,)).item())
        w = int(torch.randint(8, self.max_size + 1, (1,)).item())
        if h >= H or w >= W: return x
        y0 = int(torch.randint(0, H - h + 1, (1,)).item())
        x0 = int(torch.randint(0, W - w + 1, (1,)).item())
        x[:, y0:y0+h, x0:x0+w] = 0.0
        return x


def build_train_aug(use_brightness, use_contrast,
                    use_time_mask, use_freq_mask, use_cutout):
    """All ops here operate on already-normalized fp32 tensors."""
    ops = []
    if use_brightness or use_contrast:
        ops.append(TensorBrightnessContrast(
            brightness=0.2 if use_brightness else 0.0,
            contrast  =0.2 if use_contrast   else 0.0,
            p=0.5,
        ))
    if use_time_mask: ops.append(TimeMask(max_width=40, p=0.5, n_masks=1))
    if use_freq_mask: ops.append(FreqMask(max_height=24, p=0.5, n_masks=1))
    if use_cutout:    ops.append(Cutout(max_size=48, p=0.4))
    if not ops:
        return lambda x: x
    return transforms.Compose(ops)


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading metadata and teacher embeddings...")
meta = pd.read_parquet(META_PATH)
assert meta['png_path'].isna().sum() == 0, \
    "Missing PNG paths — run build_png_index_tursiops.py first"

meta['binary_label'] = (meta['coarse_class'] == TARGET_SPECIES).astype(np.int64)
print(f"  Total — Tursiops: {meta['binary_label'].sum():,}  "
      f"background: {(meta['binary_label']==0).sum():,}")
print(f"  All datasets: {sorted(meta['dataset'].unique())}")

available = set(meta['dataset'].unique())
for ds in HOLDOUT_DATASETS:
    if ds not in available:
        raise ValueError(
            f"Hold-out dataset '{ds}' not found in metadata. "
            f"Available: {sorted(available)}"
        )

teacher_emb_np = np.load(TEACHER_PATH)
# Single torch tensor — no per-item conversion in __getitem__.
# Float32 ~ 65k × 1536 × 4B ≈ 400 MB, fine.
TEACHER_EMB_T = torch.from_numpy(teacher_emb_np).float()
print(f"  Teacher emb: {teacher_emb_np.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# Dataset — pre-normalized fp16 tensor cache
# ══════════════════════════════════════════════════════════════════════════════
class SpectrogramDataset(Dataset):
    """
    Caches every image as a pre-normalized fp16 CHW tensor.
    __getitem__ casts back to fp32 and applies augmentations.
    On Linux fork-based workers, the cache list is shared copy-on-write,
    so multiple workers do NOT multiply the RAM footprint.
    """
    def __init__(self, meta_subset, teacher_emb_tensor, augment, cache=True):
        self.meta        = meta_subset.reset_index(drop=True)
        self.teacher_emb = teacher_emb_tensor  # shared torch tensor
        self.augment     = augment             # callable on (3,H,W) fp32 tensor
        self.cache       = None
        if cache:
            self._build_cache()

    def _build_cache(self):
        n = len(self.meta)
        t0 = time.time()
        print(f"  Caching {n} pre-normalized fp16 tensors in RAM...", flush=True)
        # Pre-allocate one big contiguous tensor — better than a list of tensors
        # for fork-COW sharing and slightly faster indexing.
        cache = torch.empty((n, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float16)
        mean = IMAGENET_MEAN  # (3,1,1) float32
        std  = IMAGENET_STD
        paths = self.meta['png_path'].tolist()
        for idx in range(n):
            img = Image.open(paths[idx]).convert('RGB')
            if img.size != (IMG_SIZE, IMG_SIZE):
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            # PIL → fp32 [0,1] CHW
            arr = np.asarray(img, dtype=np.uint8)               # H,W,3
            t   = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
            t   = (t - mean) / std
            cache[idx] = t.to(torch.float16)
            if idx > 0 and idx % 10000 == 0:
                print(f"    {idx}/{n} ({time.time()-t0:.0f}s)", flush=True)
        self.cache = cache
        mem_gb = cache.element_size() * cache.numel() / 1e9
        print(f"  Caching done in {time.time()-t0:.1f}s "
              f"({mem_gb:.1f} GB fp16)", flush=True)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        # Cast cached fp16 → fp32 (cheap), then augment.
        img = self.cache[idx].float()
        if self.augment is not None:
            img = self.augment(img)
        row     = self.meta.iloc[idx]
        t_emb   = self.teacher_emb[int(row['audio_row'])]  # already a tensor view
        label   = torch.tensor(int(row['binary_label']), dtype=torch.float32)
        dataset = torch.tensor(int(row['dataset_idx']),  dtype=torch.long)
        return img, t_emb, label, dataset


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        lam, = ctx.saved_tensors
        return -lam * grad_output, None

class GRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = 0.0
    def forward(self, x):
        return GradientReversalFn.apply(x, self.lam)

class TursiopsStudent(nn.Module):
    def __init__(self, n_datasets, teacher_dim=1536,
                 use_tanh=False, domain_head_size='small'):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0', pretrained=True,
            num_classes=0, global_pool='avg',
        )
        feat_dim    = self.backbone.num_features
        proj_layers = [
            nn.Linear(feat_dim, teacher_dim, bias=False),
            nn.LayerNorm(teacher_dim),
            nn.Dropout(0.3),
        ]
        if use_tanh:
            proj_layers.append(nn.Tanh())
        self.projection = nn.Sequential(*proj_layers)

        self.class_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(teacher_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

        self.grl = GRL()
        if domain_head_size == 'big':
            self.domain_head = nn.Sequential(
                nn.Linear(teacher_dim, 512), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(512, 256),         nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(256, n_datasets),
            )
        else:
            self.domain_head = nn.Sequential(
                nn.Linear(teacher_dim, 256), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(256, n_datasets),
            )

    def forward(self, x):
        feat = self.backbone(x)
        proj = self.projection(feat)
        return {
            'embedding':     proj,
            'class_logit':   self.class_head(proj).squeeze(1),
            'domain_logits': self.domain_head(self.grl(proj)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def get_lambda(epoch, lambda_max):
    if epoch < PHASE1_EPOCHS:
        return 0.0
    elif epoch < PHASE1_EPOCHS + PHASE2_EPOCHS:
        return lambda_max * (epoch - PHASE1_EPOCHS) / PHASE2_EPOCHS
    else:
        return lambda_max


def threshold_sweep(y_true, probs, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)
    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        results.append({'threshold': float(t), 'f1': float(f1)})
    best = max(results, key=lambda r: r['f1'])
    return best['threshold'], results


def print_split_stats(name, df, label_col='binary_label'):
    n_pos = df[label_col].sum()
    n_tot = len(df)
    print(f"    {name:12s}: {n_tot:6,} rows | "
          f"{n_pos:5,} Tursiops / {n_tot-n_pos:6,} bg "
          f"({100*n_pos/max(n_tot,1):.1f}% positive) | "
          f"datasets: {sorted(df['dataset'].unique())}")


# ══════════════════════════════════════════════════════════════════════════════
# One (variant × held-out dataset) run
# ══════════════════════════════════════════════════════════════════════════════
def run_fold(held_out_ds: str, variant_name: str, cfg: dict,
             meta_train_pool: pd.DataFrame,
             meta_test: pd.DataFrame,
             dataset_enc: LabelEncoder,
             shared_caches: dict) -> dict:
    """
    `shared_caches` lets us reuse the train/val/test cache tensors across
    variants for the same held-out fold (the cached pixels don't depend on
    the variant — only the augmentations do).
    """
    fold_tag = f"lodo_{held_out_ds}__{variant_name}"
    out_dir  = RUNS_DIR / held_out_ds / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / 'best_model.pt'

    print(f"\n{'█'*72}")
    print(f"  FOLD : held-out = {held_out_ds}")
    print(f"  VARIANT: {variant_name}  —  {cfg['note']}")
    print(f"  α={cfg['alpha_distil']}  γ={cfg['gamma_adv']}  λ_max={cfg['lambda_max']}")
    print('█'*72)

    # ── Train / val split (group-aware, within training pool) ─────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=SEED)
    train_idx, val_idx = next(
        gss.split(meta_train_pool, groups=meta_train_pool['group_key'])
    )
    meta_tr  = meta_train_pool.iloc[train_idx]
    meta_val = meta_train_pool.iloc[val_idx]

    print(f"  Data split (train pool = all except {held_out_ds}):")
    print_split_stats('train',    meta_tr)
    print_split_stats('val',      meta_val)
    print_split_stats('test(LODO)', meta_test)

    n_tt_test = meta_test['binary_label'].sum()
    if n_tt_test == 0:
        print(f"  ⚠️  WARNING: {held_out_ds} has 0 Tursiops rows — "
              f"test AUC will be undefined. Skipping.")
        return {'held_out_ds': held_out_ds, 'variant': variant_name,
                'skipped': True, 'reason': 'no_tursiops_in_test'}

    train_aug = build_train_aug(
        cfg['aug_brightness'], cfg['aug_contrast'],
        cfg['aug_time_mask'],  cfg['aug_freq_mask'], cfg['aug_cutout'],
    )

    n_train_ds = len(dataset_enc.classes_)

    # ── Build datasets (reusing cached tensors across variants) ───────────────
    t_cache = time.time()
    if 'train' not in shared_caches:
        train_ds = SpectrogramDataset(meta_tr,   TEACHER_EMB_T, train_aug)
        val_ds   = SpectrogramDataset(meta_val,  TEACHER_EMB_T, None)
        test_ds  = SpectrogramDataset(meta_test, TEACHER_EMB_T, None)
        shared_caches['train'] = (train_ds, meta_tr.index.tolist())
        shared_caches['val']   = (val_ds,   meta_val.index.tolist())
        shared_caches['test']  = (test_ds,  meta_test.index.tolist())
        print(f"  [timing] all datasets cached in {time.time()-t_cache:.1f}s total")
    else:
        # Reuse the cached pixel tensors; rebuild only the dataset wrappers so
        # we can swap in the variant-specific augmentations cheaply.
        train_ds_prev, _ = shared_caches['train']
        val_ds_prev,   _ = shared_caches['val']
        test_ds_prev,  _ = shared_caches['test']

        train_ds = SpectrogramDataset.__new__(SpectrogramDataset)
        train_ds.meta        = train_ds_prev.meta
        train_ds.teacher_emb = TEACHER_EMB_T
        train_ds.augment     = train_aug
        train_ds.cache       = train_ds_prev.cache

        val_ds = SpectrogramDataset.__new__(SpectrogramDataset)
        val_ds.meta          = val_ds_prev.meta
        val_ds.teacher_emb   = TEACHER_EMB_T
        val_ds.augment       = None
        val_ds.cache         = val_ds_prev.cache

        test_ds = SpectrogramDataset.__new__(SpectrogramDataset)
        test_ds.meta         = test_ds_prev.meta
        test_ds.teacher_emb  = TEACHER_EMB_T
        test_ds.augment      = None
        test_ds.cache        = test_ds_prev.cache

        print(f"  [timing] reusing cached tensors from previous variant "
              f"(saved ~{200:.0f}s of cache rebuild)")

    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=N_WORKERS, pin_memory=pin,
        persistent_workers=(N_WORKERS > 0),
        prefetch_factor=(4 if N_WORKERS > 0 else None),
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=max(2, N_WORKERS // 2), pin_memory=pin,
        persistent_workers=(N_WORKERS > 0),
        prefetch_factor=(4 if N_WORKERS > 0 else None),
    )
    test_dl = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=max(2, N_WORKERS // 2), pin_memory=pin,
        persistent_workers=(N_WORKERS > 0),
        prefetch_factor=(4 if N_WORKERS > 0 else None),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  [device] using: {device}")
    if device.type == 'cuda':
        print(f"  [device] GPU: {torch.cuda.get_device_name(0)}")
        print(f"  [device] VRAM free/total: "
              f"{torch.cuda.mem_get_info(0)[0]/1e9:.1f} / "
              f"{torch.cuda.mem_get_info(0)[1]/1e9:.1f} GB")

    model = TursiopsStudent(
        n_train_ds, TEACHER_DIM,
        use_tanh=cfg['use_tanh'],
        domain_head_size=cfg['domain_head_size'],
    ).to(device, memory_format=torch.channels_last)
    # channels_last gives a small bump on convnets with AMP on Ampere/Ada GPUs.

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-6)

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP and device.type == 'cuda')
    amp_dtype = torch.float16  # L4 has good fp16 tensor cores
    amp_ctx = (lambda: torch.amp.autocast('cuda', dtype=amp_dtype,
                                          enabled=USE_AMP and device.type == 'cuda'))

    n_pos = meta_tr['binary_label'].sum()
    n_neg = len(meta_tr) - n_pos
    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    class_loss_fn  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    domain_loss_fn = nn.CrossEntropyLoss()
    distil_loss_fn = nn.MSELoss()

    history = {k: [] for k in ('train_loss', 'train_auc', 'val_loss',
                                'val_auc', 'val_f1',
                                'distil_loss', 'domain_loss', 'epoch_time_s')}
    best_auc, best_f1, best_epoch, wait = 0.0, 0.0, 0, 0
    t_start = time.time()

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        lam = get_lambda(epoch, cfg['lambda_max'])
        model.grl.lam = lam
        phase = (1 if epoch < PHASE1_EPOCHS else
                 2 if epoch < PHASE1_EPOCHS + PHASE2_EPOCHS else 3)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_losses, dl_losses, do_losses = [], [], []
        all_logits_list, all_labels_list = [], []

        for imgs, t_emb, labels, ds in train_dl:
            imgs   = imgs.to(device, non_blocking=True,
                              memory_format=torch.channels_last)
            t_emb  = t_emb.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            ds     = ds.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx():
                out = model(imgs)
                loss_class  = class_loss_fn(out['class_logit'], labels)
                loss_domain = domain_loss_fn(out['domain_logits'], ds)
                loss_distil = (distil_loss_fn(out['embedding'], t_emb)
                               if cfg['alpha_distil'] > 0
                               else torch.tensor(0.0, device=device))

                if lam > 0 and cfg['gamma_adv'] > 0:
                    loss = (cfg['alpha_distil'] * loss_distil
                          + cfg['beta_class']   * loss_class
                          + cfg['gamma_adv']    * loss_domain)
                else:
                    proj_sg          = out['embedding'].detach()
                    domain_logits_sg = model.domain_head(proj_sg)
                    loss_domain_sg   = domain_loss_fn(domain_logits_sg, ds)
                    loss = (cfg['alpha_distil'] * loss_distil
                          + cfg['beta_class']   * loss_class
                          + loss_domain_sg)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            tr_losses.append(loss_class.item())
            dl_losses.append(loss_distil.item())
            do_losses.append(loss_domain.item())
            all_logits_list.append(out['class_logit'].detach().float().cpu())
            all_labels_list.append(labels.detach().cpu())

        scheduler.step()

        all_logits_t = torch.cat(all_logits_list)
        all_labels_t = torch.cat(all_labels_list)
        tr_probs = torch.sigmoid(all_logits_t).numpy()
        tr_auc   = (roc_auc_score(all_labels_t.numpy(), tr_probs)
                    if len(all_labels_t.unique()) > 1 else 0.0)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for imgs, t_emb, labels, ds in val_dl:
                imgs   = imgs.to(device, non_blocking=True,
                                  memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                with amp_ctx():
                    out = model(imgs)
                    vloss = class_loss_fn(out['class_logit'], labels)
                val_losses.append(vloss.item())
                val_logits_list.append(out['class_logit'].float().cpu())
                val_labels_list.append(labels.cpu())

        val_logits_t = torch.cat(val_logits_list)
        val_labels_t = torch.cat(val_labels_list)
        val_probs  = torch.sigmoid(val_logits_t).numpy()
        val_labels = val_labels_t.numpy()
        val_auc    = (roc_auc_score(val_labels, val_probs)
                      if len(np.unique(val_labels)) > 1 else 0.0)
        val_f1     = f1_score(val_labels, (val_probs >= 0.5).astype(int),
                               zero_division=0)

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(float(np.mean(tr_losses)))
        history['train_auc'].append(float(tr_auc))
        history['val_loss'].append(float(np.mean(val_losses)))
        history['val_auc'].append(float(val_auc))
        history['val_f1'].append(float(val_f1))
        history['distil_loss'].append(float(np.mean(dl_losses)))
        history['domain_loss'].append(float(np.mean(do_losses)))
        history['epoch_time_s'].append(float(epoch_time))

        print(f"  [{fold_tag}] ep {epoch+1:03d}/{N_EPOCHS} [P{phase}] ({epoch_time:.1f}s)  λ={lam:.3f}"
              f"\n    train  loss={np.mean(tr_losses):.4f}  distil={np.mean(dl_losses):.4f}"
              f"  domain={np.mean(do_losses):.4f}  auc={tr_auc:.4f}"
              f"\n    val    loss={np.mean(val_losses):.4f}  auc={val_auc:.4f}  f1={val_f1:.4f}",
              flush=True)

        if val_auc > best_auc + 1e-4:
            best_auc, best_f1, best_epoch, wait = val_auc, val_f1, epoch + 1, 0
            torch.save({
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'val_auc':      val_auc,
                'val_f1':       val_f1,
                'dataset_enc':  dataset_enc,
                'held_out_ds':  held_out_ds,
            }, weights_path)
            print(f"    ✓ saved best (auc={best_auc:.4f}  f1={best_f1:.4f})")
        elif epoch >= EARLY_STOP_START_EPOCH:
            wait += 1
            if wait >= PATIENCE:
                print(f"  [{fold_tag}] early stop at epoch {epoch+1}")
                break

    elapsed = time.time() - t_start
    avg_epoch_s = np.mean(history['epoch_time_s'])
    print(f"\n[{fold_tag}] best val_auc={best_auc:.4f}  val_f1={best_f1:.4f} "
          f"@ ep {best_epoch}  ({elapsed/60:.1f} min total, {avg_epoch_s:.1f}s/epoch avg)")

    # ── Test evaluation on held-out dataset ───────────────────────────────────
    ckpt = torch.load(weights_path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    test_logits_list, test_labels_list2 = [], []
    with torch.no_grad():
        for imgs, _, labels, _ in test_dl:
            imgs = imgs.to(device, non_blocking=True,
                            memory_format=torch.channels_last)
            with amp_ctx():
                out = model(imgs)
            test_logits_list.append(out['class_logit'].float().cpu())
            test_labels_list2.append(labels.cpu())

    test_probs  = torch.sigmoid(torch.cat(test_logits_list)).numpy()
    test_labels = torch.cat(test_labels_list2).numpy()

    test_auc = roc_auc_score(test_labels, test_probs)
    test_ap  = average_precision_score(test_labels, test_probs)

    best_thresh, thresh_sweep_results = threshold_sweep(val_labels, val_probs)
    test_preds = (test_probs >= best_thresh).astype(int)
    test_f1    = f1_score(test_labels, test_preds, zero_division=0)

    per_ds_rows = []
    for ds_name, grp in meta_test.groupby('dataset'):
        idx_local = grp.index - meta_test.index[0]
        if idx_local.max() >= len(test_probs): continue
        grp_probs  = test_probs[idx_local]
        grp_labels = test_labels[idx_local]
        if len(set(grp_labels)) < 2: continue
        per_ds_rows.append({
            'dataset': ds_name,
            'n': len(grp_labels),
            'n_pos': int(grp_labels.sum()),
            'auc': float(roc_auc_score(grp_labels, grp_probs)),
            'ap':  float(average_precision_score(grp_labels, grp_probs)),
        })

    print(f"[{fold_tag}] TEST  auc={test_auc:.4f}  ap={test_ap:.4f}  "
          f"f1@{best_thresh:.2f}={test_f1:.4f}")

    # ── Embedding extraction (test set only) ──────────────────────────────────
    print(f"[{fold_tag}] extracting test embeddings...")
    test_emb_list = []
    with torch.no_grad():
        for imgs, _, _, _ in test_dl:
            imgs = imgs.to(device, non_blocking=True,
                            memory_format=torch.channels_last)
            with amp_ctx():
                out = model(imgs)
            test_emb_list.append(out['embedding'].float().cpu().numpy())
    X_test_emb = np.concatenate(test_emb_list, axis=0)
    np.save(out_dir / 'X_test_emb.npy', X_test_emb)
    print(f"[{fold_tag}] test embeddings saved: {X_test_emb.shape}")

    summary = {
        'held_out_ds':        held_out_ds,
        'variant':            variant_name,
        'note':               cfg['note'],
        'config':             cfg,
        'best_val_auc':       float(best_auc),
        'best_val_f1':        float(best_f1),
        'best_epoch':         best_epoch,
        'test_auc':           float(test_auc),
        'test_ap':            float(test_ap),
        'test_f1':            float(test_f1),
        'best_threshold':     float(best_thresh),
        'threshold_sweep':    thresh_sweep_results,
        'n_test_tursiops':    int(n_tt_test),
        'n_test_total':       int(len(test_labels)),
        'per_dataset_test':   per_ds_rows,
        'n_epochs_run':       epoch + 1,
        'training_min':       elapsed / 60,
        'avg_epoch_s':        float(avg_epoch_s),
        'history':            history,
        'class_pos_weight':   float(pos_weight.item()),
        'training_prevalence': float(n_pos / len(meta_tr)),
        'train_datasets':     sorted(meta_tr['dataset'].unique().tolist()),
    }
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({**cfg, 'variant': variant_name,
                   'held_out_ds': held_out_ds}, f, indent=2)

    # Tear down loaders + workers but KEEP the cached tensors for the next variant.
    del model, optimizer, scheduler, train_dl, val_dl, test_dl, ckpt
    gc.collect()
    torch.cuda.empty_cache()
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Main — outer loop: held-out dataset × variant
# ══════════════════════════════════════════════════════════════════════════════
all_results = []

for held_out_ds in HOLDOUT_DATASETS:
    print(f"\n{'═'*72}")
    print(f"  LODO FOLD: holding out  →  {held_out_ds}")
    print(f"{'═'*72}")

    meta_test       = meta[meta['dataset'] == held_out_ds].copy()
    meta_train_pool = meta[meta['dataset'] != held_out_ds].copy()

    dataset_enc = LabelEncoder()
    meta_train_pool['dataset_idx'] = dataset_enc.fit_transform(
        meta_train_pool['dataset'])
    meta_test['dataset_idx'] = -1

    print(f"\n  Training pool datasets ({len(dataset_enc.classes_)}):")
    for ds in sorted(dataset_enc.classes_):
        n = len(meta_train_pool[meta_train_pool['dataset'] == ds])
        n_tt = meta_train_pool[meta_train_pool['dataset'] == ds]['binary_label'].sum()
        print(f"    {ds:30s} {n:6,} rows  ({n_tt} Tursiops)")
    print(f"\n  Test dataset: {held_out_ds}")
    print_split_stats('test', meta_test)

    # Cache pixels once per held-out fold and reuse across variants.
    shared_caches: dict = {}

    for variant_name, cfg in VARIANTS.items():
        try:
            res = run_fold(
                held_out_ds, variant_name, cfg,
                meta_train_pool, meta_test, dataset_enc,
                shared_caches,
            )
            all_results.append({
                'held_out_ds':          held_out_ds,
                'variant':              variant_name,
                'note':                 cfg['note'],
                'alpha_distil':         cfg['alpha_distil'],
                'gamma_adv':            cfg['gamma_adv'],
                'lambda_max':           cfg['lambda_max'],
                'best_val_auc':         res.get('best_val_auc', float('nan')),
                'best_val_f1':          res.get('best_val_f1',  float('nan')),
                'test_auc':             res.get('test_auc',     float('nan')),
                'test_ap':              res.get('test_ap',      float('nan')),
                'test_f1':              res.get('test_f1',      float('nan')),
                'best_threshold':       res.get('best_threshold', float('nan')),
                'training_prevalence':  res.get('training_prevalence', float('nan')),
                'best_epoch':           res.get('best_epoch',   -1),
                'training_min':         res.get('training_min', float('nan')),
                'avg_epoch_s':          res.get('avg_epoch_s',  float('nan')),
                'skipped':              res.get('skipped', False),
            })
        except Exception as e:
            print(f"\n[lodo_{held_out_ds}__{variant_name}] FAILED: {e}\n")
            import traceback; traceback.print_exc()
            all_results.append({
                'held_out_ds': held_out_ds,
                'variant': variant_name,
                'error': str(e),
            })

    # Free this fold's caches before the next held-out dataset.
    del shared_caches
    gc.collect()

# ── Master summary ─────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(all_results)
summary_path = RUNS_DIR / 'lodo_summary.csv'
summary_df.to_csv(summary_path, index=False)

print("\n" + "═"*72)
print("  ALL LODO FOLDS COMPLETE")
print("═"*72)

for metric in ('test_auc', 'test_ap', 'test_f1'):
    try:
        pivot = summary_df.pivot_table(
            index='variant', columns='held_out_ds',
            values=metric, aggfunc='first',
        )
        print(f"\n── {metric} ──")
        print(pivot.to_string())
    except Exception:
        pass

print(f"\nFull results → {summary_path}")
print(f"Per-fold outputs → {RUNS_DIR}/<held_out_ds>/<variant>/")
