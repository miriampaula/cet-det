# student_teacher_train.py
"""
EfficientNet-B0 student trained with:
  - Distillation loss  (MSE vs Perch V2 teacher embeddings)
  - Classification loss (cross-entropy, 10 species)
  - Adversarial loss    (GRL + domain head, 11 datasets)

Three training phases:
  Phase 1: distill + classify only (warmup, no GRL)
  Phase 2: all three losses, λ ramps up
  Phase 3: full λ, cosine LR decay to zero

Outputs:
  runs_student/best_model.pt
  runs_student/history.json
  runs_student/X_student_emb.npy   ← for LODO evaluation
"""

import os, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
META_PATH    = Path('/data2/mromaniuc/cet-det/student_teacher/meta_train_with_paths.parquet')
TEACHER_PATH = Path('/data2/mromaniuc/cet-det/student_teacher/X_teacher_emb.npy')
RUNS_DIR     = Path('/data2/mromaniuc/cet-det/student_teacher/runs_student')
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 32
N_WORKERS     = 4
SEED          = 42

# Loss weights
ALPHA_DISTIL  = 2.0   # distillation (MSE vs teacher)
BETA_CLASS    = 1.0   # classification (cross-entropy)
GAMMA_ADV     = 1.0   # adversarial domain (GRL)

# Training phases
PHASE1_EPOCHS = 10    # distill + classify, no GRL
PHASE2_EPOCHS = 30    # ramp λ from 0 → lambda_max
PHASE3_EPOCHS = 20    # full λ, LR decays to zero
N_EPOCHS      = PHASE1_EPOCHS + PHASE2_EPOCHS + PHASE3_EPOCHS

LAMBDA_MAX    = 0.5
LR            = 3e-4
WEIGHT_DECAY  = 1e-4

# Early stopping — only active from Phase 3 onwards.
# Phase 2 val_f1 wobble during λ ramp is expected; stopping there is a bug.
PATIENCE               = 15
EARLY_STOP_START_EPOCH = PHASE1_EPOCHS + PHASE2_EPOCHS  # epoch 40 (0-indexed)

TEACHER_DIM   = 1536

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
meta = pd.read_parquet(META_PATH)
assert meta['png_path'].isna().sum() == 0, "Missing PNG paths!"

# Encode labels
species_enc = LabelEncoder()
dataset_enc = LabelEncoder()
meta['species_idx'] = species_enc.fit_transform(meta['coarse_class'])
meta['dataset_idx'] = dataset_enc.fit_transform(meta['dataset'])

N_CLASSES  = len(species_enc.classes_)
N_DATASETS = len(dataset_enc.classes_)
print(f"Classes: {N_CLASSES}, Datasets: {N_DATASETS}")
print(f"Species: {list(species_enc.classes_)}")

# Train/val/test split — reproduce same split as Stage 1 using audio_row order
# Use same 82.5/8.5/9.0 split with SEED=42
from sklearn.model_selection import GroupShuffleSplit

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.085+0.09, random_state=SEED)
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.09/(0.085+0.09), random_state=SEED)

train_idx, valtest_idx = next(gss_val.split(meta, groups=meta['group_key']))
val_idx, test_idx = next(gss_test.split(
    meta.iloc[valtest_idx], groups=meta.iloc[valtest_idx]['group_key']
))
val_idx  = valtest_idx[val_idx]
test_idx = valtest_idx[test_idx]

print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
# ImageNet normalisation — matches EfficientNet pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),       # time flip — valid augmentation
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2),          # mild intensity variation
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class SpectrogramDataset(Dataset):
    def __init__(self, meta_subset, teacher_emb, transform):
        self.meta        = meta_subset.reset_index(drop=True)
        self.teacher_emb = teacher_emb   # (N, 1536) full array — index by audio_row
        self.transform   = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = Image.open(row['png_path']).convert('RGB')
        img = self.transform(img)

        # Teacher embedding aligned by audio_row
        t_emb = torch.tensor(
            self.teacher_emb[int(row['audio_row'])],
            dtype=torch.float32
        )
        species = torch.tensor(int(row['species_idx']), dtype=torch.long)
        dataset = torch.tensor(int(row['dataset_idx']), dtype=torch.long)

        return img, t_emb, species, dataset


print("Loading teacher embeddings...")
teacher_emb = np.load(TEACHER_PATH)
print(f"  Shape: {teacher_emb.shape}")

train_ds = SpectrogramDataset(meta.iloc[train_idx], teacher_emb, train_tf)
val_ds   = SpectrogramDataset(meta.iloc[val_idx],   teacher_emb, val_tf)
test_ds  = SpectrogramDataset(meta.iloc[test_idx],  teacher_emb, val_tf)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=N_WORKERS, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=N_WORKERS, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=N_WORKERS, pin_memory=True)

# ── Gradient Reversal Layer ───────────────────────────────────────────────────
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


# ── Model ─────────────────────────────────────────────────────────────────────
class StudentModel(nn.Module):
    def __init__(self, n_classes, n_datasets, teacher_dim=1536):
        super().__init__()

        # EfficientNet-B0 backbone, imagenet pretrained
        self.backbone = timm.create_model(
            'efficientnet_b0', pretrained=True,
            num_classes=0,      # remove classifier head
            global_pool='avg',  # global average pooling → (B, 1280)
        )
        feat_dim = self.backbone.num_features  # 1280 for B0

        # Projection: backbone features → teacher embedding space (1536)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, teacher_dim, bias=False),
            nn.LayerNorm(teacher_dim),
            nn.Dropout(0.3),
            nn.Tanh(),
        )

        # Classification head
        self.class_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(teacher_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, n_classes),
        )

        # Domain head with GRL
        self.grl = GRL()
        self.domain_head = nn.Sequential(
            nn.Linear(teacher_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, n_datasets),
        )

    def forward(self, x):
        feat  = self.backbone(x)           # (B, 1280)
        proj  = self.projection(feat)      # (B, 1536) — student embedding
        class_logits  = self.class_head(proj)
        domain_logits = self.domain_head(self.grl(proj))
        return {
            'embedding':      proj,
            'class_logits':   class_logits,
            'domain_logits':  domain_logits,
        }


# ── Training setup ────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = StudentModel(N_CLASSES, N_DATASETS, TEACHER_DIM).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=N_EPOCHS, eta_min=1e-6
)

class_loss_fn  = nn.CrossEntropyLoss()
domain_loss_fn = nn.CrossEntropyLoss()
distil_loss_fn = nn.MSELoss()


def get_lambda(epoch):
    """λ schedule: 0 during phase 1, ramps during phase 2, full during phase 3."""
    if epoch < PHASE1_EPOCHS:
        return 0.0
    elif epoch < PHASE1_EPOCHS + PHASE2_EPOCHS:
        progress = (epoch - PHASE1_EPOCHS) / PHASE2_EPOCHS
        return LAMBDA_MAX * progress
    else:
        return LAMBDA_MAX


# ── Training loop ─────────────────────────────────────────────────────────────
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss':   [], 'val_acc':   [],
    'val_macro_f1': [],
    'distil_loss': [], 'domain_loss': [],
}

best_f1    = 0.0
best_epoch = 0
wait       = 0
weights_path = RUNS_DIR / 'best_model.pt'

print(f"\nTraining for {N_EPOCHS} epochs "
      f"(phase1={PHASE1_EPOCHS}, phase2={PHASE2_EPOCHS}, phase3={PHASE3_EPOCHS})")
print(f"Early stopping active from epoch {EARLY_STOP_START_EPOCH + 1} (Phase 3) "
      f"with patience={PATIENCE}")

for epoch in range(N_EPOCHS):
    lam = get_lambda(epoch)
    model.grl.lam = lam
    phase = 1 if epoch < PHASE1_EPOCHS else (2 if epoch < PHASE1_EPOCHS + PHASE2_EPOCHS else 3)

    # ── Train ────────────────────────────────────────────────────────────────
    model.train()
    train_losses, distil_losses, domain_losses = [], [], []
    correct, total = 0, 0

    for imgs, t_emb, species, dataset in train_dl:
        imgs, t_emb = imgs.to(device), t_emb.to(device)
        species, dataset = species.to(device), dataset.to(device)

        optimizer.zero_grad()
        out = model(imgs)

        loss_class  = class_loss_fn(out['class_logits'], species)
        loss_distil = distil_loss_fn(out['embedding'], t_emb)
        loss_domain = domain_loss_fn(out['domain_logits'], dataset)

        if lam > 0:
            loss = (ALPHA_DISTIL * loss_distil
                  + BETA_CLASS  * loss_class
                  + GAMMA_ADV   * loss_domain)
        else:
            # Phase 1: no adversarial, but train domain head on stop-gradient
            # so it's a strong discriminator when λ activates
            with torch.no_grad():
                proj_sg = out['embedding'].detach()
            domain_logits_sg = model.domain_head(proj_sg)
            loss_domain_sg   = domain_loss_fn(domain_logits_sg, dataset)
            loss = (ALPHA_DISTIL * loss_distil
                  + BETA_CLASS  * loss_class
                  + loss_domain_sg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_losses.append(loss_class.item())
        distil_losses.append(loss_distil.item())
        domain_losses.append(loss_domain.item())
        preds = out['class_logits'].argmax(dim=1)
        correct += (preds == species).sum().item()
        total   += len(species)

    scheduler.step()

    # ── Validate ─────────────────────────────────────────────────────────────
    model.eval()
    val_losses, val_correct, val_total = [], 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, t_emb, species, dataset in val_dl:
            imgs = imgs.to(device)
            species = species.to(device)
            out = model(imgs)
            loss_val = class_loss_fn(out['class_logits'], species)
            val_losses.append(loss_val.item())
            preds = out['class_logits'].argmax(dim=1)
            val_correct += (preds == species).sum().item()
            val_total   += len(species)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(species.cpu().numpy())

    val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    history['train_loss'].append(np.mean(train_losses))
    history['train_acc'].append(correct / total)
    history['val_loss'].append(np.mean(val_losses))
    history['val_acc'].append(val_correct / val_total)
    history['val_macro_f1'].append(float(val_f1))
    history['distil_loss'].append(np.mean(distil_losses))
    history['domain_loss'].append(np.mean(domain_losses))

    print(f"  ep {epoch+1:03d}/{N_EPOCHS} [P{phase}]"
          f"  loss={np.mean(train_losses):.4f}"
          f"  distil={np.mean(distil_losses):.4f}"
          f"  acc={correct/total:.4f}"
          f"  val_f1={val_f1:.4f}"
          f"  λ={lam:.3f}")

    # ── Checkpoint & early stopping ───────────────────────────────────────────
    # Always save best checkpoint regardless of phase.
    # Early stopping counter only runs from Phase 3 onwards — val_f1 wobble
    # during the Phase 2 λ ramp is expected behaviour, not stagnation.
    if val_f1 > best_f1:
        best_f1    = val_f1
        best_epoch = epoch + 1
        wait       = 0
        torch.save({
            'epoch':        epoch,
            'model_state':  model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'val_f1':       val_f1,
            'species_enc':  species_enc,
            'dataset_enc':  dataset_enc,
        }, weights_path)
        print(f"    ✓ saved best ({best_f1:.4f})")
    elif epoch >= EARLY_STOP_START_EPOCH:
        # Only count patience during Phase 3
        wait += 1
        if wait >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break
    # Phase 1 / Phase 2: never increment wait — λ ramp causes expected dip

print(f"\nBest val_macro_f1: {best_f1:.4f} @ epoch {best_epoch}")

# ── Test ──────────────────────────────────────────────────────────────────────
checkpoint = torch.load(weights_path, weights_only=False)
model.load_state_dict(checkpoint['model_state'])
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, t_emb, species, dataset in test_dl:
        imgs = imgs.to(device)
        out  = model(imgs)
        preds = out['class_logits'].argmax(dim=1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(species.numpy())

test_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
print(f"Test macro_f1: {test_f1:.4f}")

# ── Extract full embeddings for LODO ─────────────────────────────────────────
print("\nExtracting student embeddings for LODO...")
full_dl = DataLoader(
    SpectrogramDataset(meta, teacher_emb, val_tf),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=N_WORKERS, pin_memory=True
)

all_embeddings = []
model.eval()
with torch.no_grad():
    for imgs, t_emb, species, dataset in tqdm(full_dl):
        imgs = imgs.to(device)
        out  = model(imgs)
        all_embeddings.append(out['embedding'].cpu().numpy())

X_student_emb = np.concatenate(all_embeddings, axis=0)
emb_path = RUNS_DIR / 'X_student_emb.npy'
np.save(emb_path, X_student_emb)
print(f"Student embeddings saved → {emb_path}")
print(f"Shape: {X_student_emb.shape}")

# ── Save history ──────────────────────────────────────────────────────────────
cfg = {
    'best_val_macro_f1': best_f1,
    'best_epoch':        best_epoch,
    'test_macro_f1':     test_f1,
    'n_epochs_run':      epoch + 1,
    'history':           history,
    'alpha_distil':      ALPHA_DISTIL,
    'beta_class':        BETA_CLASS,
    'gamma_adv':         GAMMA_ADV,
    'lambda_max':        LAMBDA_MAX,
}
with open(RUNS_DIR / 'history.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print(f"History saved → {RUNS_DIR / 'history.json'}")