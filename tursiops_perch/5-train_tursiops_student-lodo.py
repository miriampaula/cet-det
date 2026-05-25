#!/usr/bin/env python3
"""
Tursiops truncatus binary student — LODO (Leave-One-Dataset-Out) variant sweep.

Each held-out dataset is used as the test set; the model trains on all remaining
datasets. Variants are swept for every held-out fold, so you get a full
(variant × held_out_dataset) results matrix.

Usage:
  # Hold out both Adriatic_Sea and ALNITAK_CAVANILLES (default):
  python 5-train_tursiops_student_lodo.py

  # Custom hold-out set(s):
  python 5-train_tursiops_student-lodo.py --holdout Adriatic_Sea
  python 5-train_tursiops_student-lodo.py --holdout Adriatic_Sea ALNITAK_CAVANILLES
  python 5-train_tursiops_student-lodo.py --holdout DRYAD OLTREMARE

  Background:
  nohup python '/data2/mromaniuc/cet-det/tursiops_perch/5-train_tursiops_student-lodo.py' \
        --holdout Adriatic_Sea ALNITAK_CAVANILLES \
        > logs/lodo.log 2>&1 &
  echo $! > logs/lodo.pid
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

import timm as _timm
_ = _timm.create_model('efficientnet_b0', pretrained=True)
del _
print("[startup] EfficientNet-B0 weights cached.", flush=True)

from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
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
BATCH_SIZE    = 128
N_WORKERS     = 8
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

print(f"\n[startup] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[startup] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[startup] VRAM: {torch.cuda.mem_get_info(0)[1]/1e9:.1f} GB total")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


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
    'v02_grl_only': dict(
        alpha_distil=0.0, beta_class=1.0, gamma_adv=1.0,
        lambda_max=0.2, use_tanh=False, domain_head_size='small',
        aug_brightness=True, aug_contrast=True,
        aug_time_mask=True, aug_freq_mask=True, aug_cutout=True,
        note='binary + GRL only — no distillation',
    ),
    'v03_no_grl': dict(
        alpha_distil=0.0, beta_class=1.0, gamma_adv=0.0,
        lambda_max=0.0, use_tanh=False, domain_head_size='small',
        aug_brightness=True, aug_contrast=True,
        aug_time_mask=True, aug_freq_mask=True, aug_cutout=True,
        note='binary only — no distillation, no GRL (true ablation baseline)',
    ),
    'v04_strong_distil': dict(
        alpha_distil=5.0, beta_class=1.0, gamma_adv=1.0,
        lambda_max=0.2, use_tanh=False, domain_head_size='small',
        aug_brightness=True, aug_contrast=True,
        aug_time_mask=True, aug_freq_mask=True, aug_cutout=True,
        note='strong distillation pull toward Perch geometry',
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Augmentations
# ══════════════════════════════════════════════════════════════════════════════
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


def build_train_transform(use_brightness, use_contrast,
                           use_time_mask, use_freq_mask, use_cutout):
    pil_ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if use_brightness or use_contrast:
        pil_ops.append(transforms.ColorJitter(
            brightness=0.2 if use_brightness else 0,
            contrast  =0.2 if use_contrast   else 0,
        ))
    tensor_ops = [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if use_time_mask: tensor_ops.append(TimeMask(max_width=40, p=0.5, n_masks=1))
    if use_freq_mask: tensor_ops.append(FreqMask(max_height=24, p=0.5, n_masks=1))
    if use_cutout:    tensor_ops.append(Cutout(max_size=48, p=0.4))
    return transforms.Compose(pil_ops + tensor_ops)


val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


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

# Validate requested hold-out names
available = set(meta['dataset'].unique())
for ds in HOLDOUT_DATASETS:
    if ds not in available:
        raise ValueError(
            f"Hold-out dataset '{ds}' not found in metadata. "
            f"Available: {sorted(available)}"
        )

teacher_emb = np.load(TEACHER_PATH)
print(f"  Teacher emb: {teacher_emb.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class SpectrogramDataset(Dataset):
    def __init__(self, meta_subset, teacher_emb, transform, cache=True):
        self.meta        = meta_subset.reset_index(drop=True)
        self.teacher_emb = teacher_emb
        self.transform   = transform
        self.cache       = {}
        if cache:
            t0 = time.time()
            print(f"  Caching {len(self.meta)} images in RAM...", flush=True)
            # No tqdm — just a periodic progress print
            n = len(self.meta)
            for idx in range(n):
                path = self.meta.iloc[idx]['png_path']
                self.cache[idx] = Image.open(path).convert('RGB')
                if idx > 0 and idx % 10000 == 0:
                    print(f"    {idx}/{n} ({time.time()-t0:.0f}s)", flush=True)
            print(f"  Caching done in {time.time()-t0:.1f}s", flush=True)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = self.cache[idx] if self.cache else Image.open(row['png_path']).convert('RGB')
        img = self.transform(img)
        t_emb = torch.tensor(self.teacher_emb[int(row['audio_row'])],
                              dtype=torch.float32)
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
             dataset_enc: LabelEncoder) -> dict:
    """
    Train one variant with `held_out_ds` as the test set.
    `meta_train_pool` is everything EXCEPT held_out_ds.
    `meta_test`       is only held_out_ds rows.
    `dataset_enc`     is fitted on meta_train_pool['dataset'] only.
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

    # Sanity check: held-out must have at least some Tursiops to be useful
    n_tt_test = meta_test['binary_label'].sum()
    if n_tt_test == 0:
        print(f"  ⚠️  WARNING: {held_out_ds} has 0 Tursiops rows — "
              f"test AUC will be undefined. Skipping.")
        return {'held_out_ds': held_out_ds, 'variant': variant_name,
                'skipped': True, 'reason': 'no_tursiops_in_test'}

    train_tf = build_train_transform(
        cfg['aug_brightness'], cfg['aug_contrast'],
        cfg['aug_time_mask'],  cfg['aug_freq_mask'], cfg['aug_cutout'],
    )

    n_train_ds = len(dataset_enc.classes_)

    # ── Build datasets with timing ────────────────────────────────────────────
    t_cache = time.time()
    train_ds = SpectrogramDataset(meta_tr,   teacher_emb, train_tf)
    val_ds   = SpectrogramDataset(meta_val,  teacher_emb, val_tf)
    test_ds  = SpectrogramDataset(meta_test, teacher_emb, val_tf)
    print(f"  [timing] all datasets cached in {time.time()-t_cache:.1f}s total")

    # persistent_workers=True avoids respawning workers every epoch
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=N_WORKERS, pin_memory=True,
                          persistent_workers=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=N_WORKERS, pin_memory=True,
                          persistent_workers=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=N_WORKERS, pin_memory=True,
                          persistent_workers=True)

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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-6)

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
        # Accumulate tensors instead of converting each batch to numpy
        all_logits_list, all_labels_list = [], []

        for imgs, t_emb, labels, ds in train_dl:
            imgs   = imgs.to(device)
            t_emb  = t_emb.to(device)
            labels = labels.to(device)
            ds     = ds.to(device)
            optimizer.zero_grad()
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

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_losses.append(loss_class.item())
            dl_losses.append(loss_distil.item())
            do_losses.append(loss_domain.item())
            # Accumulate CPU tensors — avoids per-batch numpy conversion
            all_logits_list.append(out['class_logit'].detach().cpu())
            all_labels_list.append(labels.cpu())

        scheduler.step()

        # Concatenate once after full epoch
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
                imgs   = imgs.to(device)
                labels = labels.to(device)
                out    = model(imgs)
                val_losses.append(class_loss_fn(out['class_logit'], labels).item())
                val_logits_list.append(out['class_logit'].cpu())
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
              f"\n    val    loss={np.mean(val_losses):.4f}  auc={val_auc:.4f}  f1={val_f1:.4f}")

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
            out = model(imgs.to(device))
            test_logits_list.append(out['class_logit'].cpu())
            test_labels_list2.append(labels.cpu())

    test_probs  = torch.sigmoid(torch.cat(test_logits_list)).numpy()
    test_labels = torch.cat(test_labels_list2).numpy()

    test_auc = roc_auc_score(test_labels, test_probs)
    test_ap  = average_precision_score(test_labels, test_probs)

    best_thresh, thresh_sweep_results = threshold_sweep(val_labels, val_probs)
    test_preds = (test_probs >= best_thresh).astype(int)
    test_f1    = f1_score(test_labels, test_preds, zero_division=0)

    # Per-dataset breakdown inside the test set
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
        for imgs, _, _, _ in tqdm(test_dl, desc=f'{fold_tag} emb'):
            out = model(imgs.to(device))
            test_emb_list.append(out['embedding'].cpu().numpy())
    X_test_emb = np.concatenate(test_emb_list, axis=0)
    np.save(out_dir / 'X_test_emb.npy', X_test_emb)
    print(f"[{fold_tag}] test embeddings saved: {X_test_emb.shape}")

    # ── Save summary ──────────────────────────────────────────────────────────
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

    del model, optimizer, scheduler, train_dl, val_dl, test_dl
    del train_ds, val_ds, test_ds, ckpt
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

    # Partition metadata
    meta_test       = meta[meta['dataset'] == held_out_ds].copy()
    meta_train_pool = meta[meta['dataset'] != held_out_ds].copy()

    # Re-fit label encoder on training datasets only
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

    for variant_name, cfg in VARIANTS.items():
        try:
            res = run_fold(
                held_out_ds, variant_name, cfg,
                meta_train_pool, meta_test, dataset_enc,
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
