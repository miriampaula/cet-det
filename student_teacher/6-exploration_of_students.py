#!/usr/bin/env python3
"""
Student-teacher variant sweep with proper spectrogram augmentations.

Augmentations applied on-the-fly per epoch (never modifying source PNGs):
  - time_mask:       zero out random contiguous time columns
  - freq_mask:       zero out random contiguous mel bands
  - brightness:      scale pixel intensity (volume-like)
  - contrast:        scale around mean (SNR-like)
  - cutout:          zero a random rectangular patch

Each variant tweaks ONE thing from baseline so improvements are attributable.

Variant artifacts under: runs_student_variants/{variant_name}/

Usage:
  cd /data2/mromaniuc/cet-det/student_teacher
  mkdir -p logs
  nohup python 6-exploration_of_students.py > logs/variants.log 2>&1 &
  echo $! > logs/variants.pid
  tail -f logs/variants.log
"""

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
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path('/data2/mromaniuc/cet-det/student_teacher')
META_PATH    = BASE / 'meta_train_with_paths.parquet'
TEACHER_PATH = BASE / 'X_teacher_emb.npy'
RUNS_DIR     = BASE / 'runs_student_variants'
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 32
N_WORKERS     = 4
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════════════
# Spectrogram augmentations — tensor-space, random per call, never cached
# ══════════════════════════════════════════════════════════════════════════════
# All operate on (C, H, W) tensors where H = freq axis, W = time axis.
# A fresh random parameter is sampled every __call__, so each epoch sees a
# different augmented view of every PNG. The PNG on disk is never modified.

class TimeMask:
    """Zero out a contiguous block of time columns. SpecAugment-style."""
    def __init__(self, max_width=40, p=0.5, n_masks=1):
        self.max_width = max_width
        self.p         = p
        self.n_masks   = n_masks

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x
        x = x.clone()
        _, _, W = x.shape
        for _ in range(self.n_masks):
            w = int(torch.randint(1, self.max_width + 1, (1,)).item())
            if w >= W:
                continue
            t0 = int(torch.randint(0, W - w + 1, (1,)).item())
            x[:, :, t0:t0+w] = 0.0
        return x


class FreqMask:
    """Zero out a contiguous block of mel bands. SpecAugment-style."""
    def __init__(self, max_height=24, p=0.5, n_masks=1):
        self.max_height = max_height
        self.p          = p
        self.n_masks    = n_masks

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x
        x = x.clone()
        _, H, _ = x.shape
        for _ in range(self.n_masks):
            h = int(torch.randint(1, self.max_height + 1, (1,)).item())
            if h >= H:
                continue
            f0 = int(torch.randint(0, H - h + 1, (1,)).item())
            x[:, f0:f0+h, :] = 0.0
        return x


class Cutout:
    """Zero a random rectangular patch."""
    def __init__(self, max_size=48, p=0.5):
        self.max_size = max_size
        self.p        = p

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x
        x = x.clone()
        _, H, W = x.shape
        h = int(torch.randint(8, self.max_size + 1, (1,)).item())
        w = int(torch.randint(8, self.max_size + 1, (1,)).item())
        if h >= H or w >= W:
            return x
        y0 = int(torch.randint(0, H - h + 1, (1,)).item())
        x0 = int(torch.randint(0, W - w + 1, (1,)).item())
        x[:, y0:y0+h, x0:x0+w] = 0.0
        return x


def build_train_transform(use_brightness, use_contrast,
                           use_time_mask, use_freq_mask, use_cutout):
    """Brightness + contrast in PIL space (before ToTensor); time/freq mask
    + cutout in tensor space (after Normalize). Randomness inside each
    transform fires fresh per __getitem__ call.
    """
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
    if use_time_mask:
        tensor_ops.append(TimeMask(max_width=40, p=0.5, n_masks=1))
    if use_freq_mask:
        tensor_ops.append(FreqMask(max_height=24, p=0.5, n_masks=1))
    if use_cutout:
        tensor_ops.append(Cutout(max_size=48, p=0.4))

    return transforms.Compose(pil_ops + tensor_ops)


val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ══════════════════════════════════════════════════════════════════════════════
# Variant table — one thing changes from baseline per variant
# ══════════════════════════════════════════════════════════════════════════════
VARIANTS = {
    'v01_baseline': dict(
        lambda_max=0.5, alpha_distil=2.0, beta_class=1.0, gamma_adv=1.0,
        use_tanh=True, domain_head_size='small',
        aug_brightness=True, aug_contrast=True,
        aug_time_mask=True, aug_freq_mask=True, aug_cutout=False,
        note='reference: SpecAugment (time+freq mask) + light colour jitter',
    ),
    'v02_low_lambda': dict(
        lambda_max=0.2, alpha_distil=2.0, beta_class=1.0, gamma_adv=1.0,
        use_tanh=True, domain_head_size='small',
        aug_brightness=True, aug_contrast=True,
        aug_time_mask=True, aug_freq_mask=True, aug_cutout=False,
        note='tests whether GRL@0.5 over-collapses species clusters',
    ),
    'v03_high_lambda': dict(
        lambda_max=1.0, alpha_distil=2.0, beta_class=1.0, gamma_adv=1.0,
        use_tanh=True, domain_head_size='small',
        aug_brightness=True, aug_contrast=True,
        aug_time_mask=True, aug_freq_mask=True, aug_cutout=False,
        note='confirms direction; expect species cluster collapse',
    ),
    'v04_strong_distil': dict(
        lambda_max=0.5, alpha_distil=5.0, beta_class=1.0, gamma_adv=1.0,
        use_tanh=True, domain_head_size='small',
        aug_brightness=True, aug_contrast=True,
        aug_time_mask=True, aug_freq_mask=True, aug_cutout=False,
        note='preserve Perch geometry harder against GRL pull',
    ),
    'v05_no_tanh': dict(
        lambda_max=0.5, alpha_distil=2.0, beta_class=1.0, gamma_adv=1.0,
        use_tanh=False, domain_head_size='small',
        aug_brightness=True, aug_contrast=True,
        aug_time_mask=True, aug_freq_mask=True, aug_cutout=False,
        note='remove [-1,1] squashing; let magnitudes match Perch',
    ),
    'v06_no_aug': dict(
        lambda_max=0.5, alpha_distil=2.0, beta_class=1.0, gamma_adv=1.0,
        use_tanh=True, domain_head_size='small',
        aug_brightness=False, aug_contrast=False,
        aug_time_mask=False, aug_freq_mask=False, aug_cutout=False,
        note='clean signal — does any augment hurt distillation?',
    ),
    'v07_heavy_aug': dict(
        lambda_max=0.5, alpha_distil=2.0, beta_class=1.0, gamma_adv=1.0,
        use_tanh=True, domain_head_size='small',
        aug_brightness=True, aug_contrast=True,
        aug_time_mask=True, aug_freq_mask=True, aug_cutout=True,
        note='all five augmentations including cutout',
    ),
    'v08_big_domain': dict(
        lambda_max=0.5, alpha_distil=2.0, beta_class=1.0, gamma_adv=1.0,
        use_tanh=True, domain_head_size='big',
        aug_brightness=True, aug_contrast=True,
        aug_time_mask=True, aug_freq_mask=True, aug_cutout=False,
        note='stronger adversary forces backbone to do real work',
    ),
    'v09_best_combo': dict(
        lambda_max=0.2, alpha_distil=5.0, beta_class=1.0, gamma_adv=1.0,
        use_tanh=False, domain_head_size='small',
        aug_brightness=True, aug_contrast=True,
        aug_time_mask=True, aug_freq_mask=True, aug_cutout=True,
        note='educated guess: low λ + strong distil + no tanh + heavy aug',
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Shared data setup (loaded once)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading metadata and teacher embeddings...")
meta = pd.read_parquet(META_PATH)
assert meta['png_path'].isna().sum() == 0

species_enc = LabelEncoder()
dataset_enc = LabelEncoder()
meta['species_idx'] = species_enc.fit_transform(meta['coarse_class'])
meta['dataset_idx'] = dataset_enc.fit_transform(meta['dataset'])

N_CLASSES  = len(species_enc.classes_)
N_DATASETS = len(dataset_enc.classes_)
print(f"  Classes: {N_CLASSES}, Datasets: {N_DATASETS}")

gss_val  = GroupShuffleSplit(n_splits=1, test_size=0.085+0.09, random_state=SEED)
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.09/(0.085+0.09), random_state=SEED)
train_idx, valtest_idx = next(gss_val.split(meta, groups=meta['group_key']))
val_idx, test_idx = next(gss_test.split(
    meta.iloc[valtest_idx], groups=meta.iloc[valtest_idx]['group_key']
))
val_idx  = valtest_idx[val_idx]
test_idx = valtest_idx[test_idx]
print(f"  Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

teacher_emb = np.load(TEACHER_PATH)
print(f"  Teacher emb: {teacher_emb.shape}")


class SpectrogramDataset(Dataset):
    """Transform is called fresh on every __getitem__; the source PNG is
    never modified on disk. Each epoch sees a different augmented view.
    """
    def __init__(self, meta_subset, teacher_emb, transform):
        self.meta        = meta_subset.reset_index(drop=True)
        self.teacher_emb = teacher_emb
        self.transform   = transform
    def __len__(self):
        return len(self.meta)
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = Image.open(row['png_path']).convert('RGB')
        img = self.transform(img)
        t_emb   = torch.tensor(self.teacher_emb[int(row['audio_row'])],
                               dtype=torch.float32)
        species = torch.tensor(int(row['species_idx']), dtype=torch.long)
        dataset = torch.tensor(int(row['dataset_idx']), dtype=torch.long)
        return img, t_emb, species, dataset


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


class StudentModel(nn.Module):
    def __init__(self, n_classes, n_datasets,
                 teacher_dim=1536, use_tanh=True, domain_head_size='small'):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0', pretrained=True,
            num_classes=0, global_pool='avg',
        )
        feat_dim = self.backbone.num_features

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
            nn.Linear(256, n_classes),
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
            'class_logits':  self.class_head(proj),
            'domain_logits': self.domain_head(self.grl(proj)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# One variant
# ══════════════════════════════════════════════════════════════════════════════
def get_lambda(epoch, lambda_max):
    if epoch < PHASE1_EPOCHS:
        return 0.0
    elif epoch < PHASE1_EPOCHS + PHASE2_EPOCHS:
        return lambda_max * (epoch - PHASE1_EPOCHS) / PHASE2_EPOCHS
    else:
        return lambda_max


def run_variant(name, cfg):
    out_dir = RUNS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / 'best_model.pt'

    print(f"\n{'█'*72}")
    print(f"  VARIANT: {name}")
    print(f"  {cfg['note']}")
    print(f"  λ_max={cfg['lambda_max']}  α={cfg['alpha_distil']}  "
          f"tanh={cfg['use_tanh']}  domain={cfg['domain_head_size']}")
    aug_flags = [k.replace('aug_','') for k in
                 ('aug_brightness','aug_contrast','aug_time_mask',
                  'aug_freq_mask','aug_cutout') if cfg[k]]
    print(f"  augmentations: {aug_flags or '(none)'}")
    print('█'*72)

    train_tf = build_train_transform(
        cfg['aug_brightness'], cfg['aug_contrast'],
        cfg['aug_time_mask'],  cfg['aug_freq_mask'], cfg['aug_cutout'],
    )

    train_ds = SpectrogramDataset(meta.iloc[train_idx], teacher_emb, train_tf)
    val_ds   = SpectrogramDataset(meta.iloc[val_idx],   teacher_emb, val_tf)
    test_ds  = SpectrogramDataset(meta.iloc[test_idx],  teacher_emb, val_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=N_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=N_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=N_WORKERS, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StudentModel(
        N_CLASSES, N_DATASETS, TEACHER_DIM,
        use_tanh=cfg['use_tanh'],
        domain_head_size=cfg['domain_head_size'],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-6,
    )
    class_loss_fn  = nn.CrossEntropyLoss()
    domain_loss_fn = nn.CrossEntropyLoss()
    distil_loss_fn = nn.MSELoss()

    history = {k: [] for k in ('train_loss','train_acc','val_loss',
                                'val_acc','val_macro_f1',
                                'distil_loss','domain_loss')}
    best_f1, best_epoch, wait = 0.0, 0, 0
    t_start = time.time()

    for epoch in range(N_EPOCHS):
        lam = get_lambda(epoch, cfg['lambda_max'])
        model.grl.lam = lam
        phase = 1 if epoch < PHASE1_EPOCHS else (
                2 if epoch < PHASE1_EPOCHS+PHASE2_EPOCHS else 3)

        model.train()
        tr_losses, dl_losses, do_losses = [], [], []
        correct, total = 0, 0

        for imgs, t_emb, species, ds in train_dl:
            imgs, t_emb = imgs.to(device), t_emb.to(device)
            species, ds = species.to(device), ds.to(device)
            optimizer.zero_grad()
            out = model(imgs)

            loss_class  = class_loss_fn(out['class_logits'], species)
            loss_distil = distil_loss_fn(out['embedding'], t_emb)
            loss_domain = domain_loss_fn(out['domain_logits'], ds)

            if lam > 0:
                loss = (cfg['alpha_distil'] * loss_distil
                      + cfg['beta_class']   * loss_class
                      + cfg['gamma_adv']    * loss_domain)
            else:
                proj_sg = out['embedding'].detach()
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
            preds = out['class_logits'].argmax(dim=1)
            correct += (preds == species).sum().item()
            total   += len(species)

        scheduler.step()

        model.eval()
        val_losses, vc, vt = [], 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, t_emb, species, ds in val_dl:
                imgs    = imgs.to(device)
                species = species.to(device)
                out     = model(imgs)
                val_losses.append(class_loss_fn(out['class_logits'], species).item())
                preds = out['class_logits'].argmax(dim=1)
                vc += (preds == species).sum().item()
                vt += len(species)
                y_pred.extend(preds.cpu().numpy())
                y_true.extend(species.cpu().numpy())

        val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        history['train_loss'].append(float(np.mean(tr_losses)))
        history['train_acc'].append(correct/total)
        history['val_loss'].append(float(np.mean(val_losses)))
        history['val_acc'].append(vc/vt)
        history['val_macro_f1'].append(float(val_f1))
        history['distil_loss'].append(float(np.mean(dl_losses)))
        history['domain_loss'].append(float(np.mean(do_losses)))

        print(f"  [{name}] ep {epoch+1:03d}/{N_EPOCHS} [P{phase}]"
              f"  loss={np.mean(tr_losses):.4f}"
              f"  distil={np.mean(dl_losses):.4f}"
              f"  domain={np.mean(do_losses):.4f}"
              f"  acc={correct/total:.4f}"
              f"  val_f1={val_f1:.4f}"
              f"  λ={lam:.3f}")

        if val_f1 > best_f1:
            best_f1, best_epoch, wait = val_f1, epoch+1, 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_f1':      val_f1,
                'species_enc': species_enc,
                'dataset_enc': dataset_enc,
            }, weights_path)
            print(f"    ✓ saved best ({best_f1:.4f})")
        elif epoch >= EARLY_STOP_START_EPOCH:
            wait += 1
            if wait >= PATIENCE:
                print(f"  [{name}] early stop at epoch {epoch+1}")
                break

    elapsed = time.time() - t_start
    print(f"\n[{name}] best val_macro_f1 = {best_f1:.4f} @ ep {best_epoch}  "
          f"({elapsed/60:.1f} min)")

    # ── Test on held-out test set ──────────────────────────────────────────────
    ckpt = torch.load(weights_path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, _, species, _ in test_dl:
            out = model(imgs.to(device))
            y_pred.extend(out['class_logits'].argmax(dim=1).cpu().numpy())
            y_true.extend(species.numpy())
    test_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"[{name}] test_macro_f1 = {test_f1:.4f}")

    # ── Extract full embeddings (clean transform, no augment) ──────────────────
    print(f"[{name}] extracting full embeddings...")
    full_dl = DataLoader(
        SpectrogramDataset(meta, teacher_emb, val_tf),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=N_WORKERS, pin_memory=True,
    )
    all_emb = []
    with torch.no_grad():
        for imgs, _, _, _ in tqdm(full_dl, desc=f'{name} emb'):
            out = model(imgs.to(device))
            all_emb.append(out['embedding'].cpu().numpy())
    X_student = np.concatenate(all_emb, axis=0)
    np.save(out_dir / 'X_student_emb.npy', X_student)
    print(f"[{name}] embeddings saved: {X_student.shape}")

    # ── Persist everything ─────────────────────────────────────────────────────
    summary = {
        'variant':           name,
        'note':              cfg['note'],
        'config':            cfg,
        'best_val_macro_f1': float(best_f1),
        'best_epoch':        best_epoch,
        'test_macro_f1':     float(test_f1),
        'n_epochs_run':      epoch + 1,
        'training_min':      elapsed / 60,
        'history':           history,
    }
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({**cfg, 'variant': name}, f, indent=2)

    del model, optimizer, scheduler, train_dl, val_dl, test_dl, full_dl
    del train_ds, val_ds, test_ds, ckpt
    gc.collect()
    torch.cuda.empty_cache()
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
all_results = []
for name, cfg in VARIANTS.items():
    try:
        res = run_variant(name, cfg)
        all_results.append({
            'variant':           name,
            'note':              cfg['note'],
            'lambda_max':        cfg['lambda_max'],
            'alpha_distil':      cfg['alpha_distil'],
            'use_tanh':          cfg['use_tanh'],
            'domain_head_size':  cfg['domain_head_size'],
            'n_augs': sum(cfg[k] for k in ('aug_brightness','aug_contrast',
                                            'aug_time_mask','aug_freq_mask',
                                            'aug_cutout')),
            'best_val_macro_f1': res['best_val_macro_f1'],
            'test_macro_f1':     res['test_macro_f1'],
            'best_epoch':        res['best_epoch'],
            'training_min':      res['training_min'],
        })
    except Exception as e:
        print(f"\n[{name}] FAILED: {e}\n")
        import traceback; traceback.print_exc()
        all_results.append({'variant': name, 'note': cfg['note'], 'error': str(e)})

summary_df = pd.DataFrame(all_results)
summary_df.to_csv(RUNS_DIR / 'variants_summary.csv', index=False)
print("\n" + "═"*72)
print("  ALL VARIANTS COMPLETE")
print("═"*72)
print(summary_df.to_string(index=False))
print(f"\nSummary CSV → {RUNS_DIR / 'variants_summary.csv'}")
print(f"All variant artifacts → {RUNS_DIR}")