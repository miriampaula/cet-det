#!/usr/bin/env python3
"""
Student-teacher LODO evaluation — drop-in replacement for the Perch LODO.

Loads:
  X_student_emb.npy          (11769, 1536)  student embeddings
  meta_train_with_paths.parquet             11769 rows, columns include:
    coarse_class, dataset, group_key, audio_row

Runs the same three experiments as perch_v2_cetacean_safe_lodo.ipynb:
  Exp 1 — within-corpus joint baseline  (flat vs hierarchical)
  Exp 2 — joint LODO across 11 datasets
  Exp 3 — pure species LODO (4 clean holdout datasets)

All metrics match the Perch notebook exactly so numbers are directly comparable.

Usage:
  nohup python 5-student_lodo.py > logs/student_lodo.log 2>&1 &
  tail -f logs/student_lodo.log
"""

import os, json, time, warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # no display needed — saves to file
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, classification_report, accuracy_score,
    balanced_accuracy_score, confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
np.random.seed(42)
torch.manual_seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = Path('/data2/mromaniuc/cet-det/student_teacher')
EMB_PATH      = BASE / 'runs_student/X_student_emb.npy'
META_PATH     = BASE / 'meta_train_with_paths.parquet'
OUT_DIR       = BASE / 'lodo_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR       = OUT_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {DEVICE}  torch: {torch.__version__}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading embeddings and metadata...")
X_all = np.load(EMB_PATH, mmap_mode='r')
meta_all = pd.read_parquet(META_PATH)

print(f"  X_student_emb: {X_all.shape}  dtype={X_all.dtype}")
print(f"  meta:          {len(meta_all):,} rows")

# audio_row is the index into X_all
assert 'audio_row' in meta_all.columns, "meta must have audio_row column"
assert meta_all['audio_row'].max() < len(X_all)

# group_key is required for leak-free splits
assert 'group_key' in meta_all.columns

ALL_DATASETS = sorted(meta_all['dataset'].unique().tolist())
print(f"\n  datasets: {ALL_DATASETS}")
print(f"  species:  {sorted(meta_all['coarse_class'].unique().tolist())}")

# ── Build joint label ─────────────────────────────────────────────────────────
# Matches the Perch notebook exactly.
# In meta_train, coarse_class is already cleaned:
#   species names  →  keep as-is
#   'background'   →  background
# No anthropogenic / unknown rows expected, but fold anything non-species into
# background defensively.

SPECIES = {
    'Balaenoptera_physalus', 'Delphinus_delphis', 'Delphinids',
    'Globicephala_melas', 'Grampus_griseus', 'Orcinus_orca',
    'Physeter_macrocephalus', 'Stenella_coeruleoalba', 'Tursiops_truncatus',
}
NON_MAMMAL_JOINT = 'background'

meta_all = meta_all.copy()
meta_all['label_joint'] = meta_all['coarse_class'].apply(
    lambda c: c if c in SPECIES else NON_MAMMAL_JOINT
)

print(f"\njoint label distribution:")
print(meta_all['label_joint'].value_counts().to_string())

# ── Shared MLP ────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(512, 256), dropout=0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, Xva, yva, n_classes,
              hidden=(512, 256), dropout=0.3,
              lr=1e-3, weight_decay=1e-4, batch_size=1024,
              epochs=40, patience=6, class_weight=True, verbose=False):
    model = MLP(Xtr.shape[1], n_classes, hidden=hidden, dropout=dropout).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if class_weight:
        counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
        w      = counts.sum() / (n_classes * np.clip(counts, 1, None))
        crit   = nn.CrossEntropyLoss(
                     weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    else:
        crit = nn.CrossEntropyLoss()

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                       batch_size=batch_size, shuffle=True,  num_workers=0)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                       batch_size=batch_size, shuffle=False, num_workers=0)

    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                preds.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                trues.append(yb.numpy())
        va_f1 = f1_score(np.concatenate(trues), np.concatenate(preds),
                         average='macro', zero_division=0)
        if verbose:
            print(f"    ep{ep:02d}  val_f1={va_f1:.4f}")
        if va_f1 > best_f1 + 1e-4:
            best_f1    = va_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


def predict_mlp(model, X, batch_size=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(torch.from_numpy(X[i:i+batch_size]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_eval_mlp(Xtr, ytr, Xva, yva, Xte, yte, n_classes,
                   class_names=None, verbose=False, **kwargs):
    sc   = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr).astype(np.float32)
    Xva_s = sc.transform(Xva).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)
    t0    = time.time()
    model, best_va = train_mlp(
        Xtr_s, ytr.astype(np.int64), Xva_s, yva.astype(np.int64),
        n_classes=n_classes, verbose=verbose, **kwargs)
    logits_te = predict_mlp(model, Xte_s)
    elapsed   = time.time() - t0
    ypr       = logits_te.argmax(1)
    present   = sorted(np.unique(yte).tolist())
    full      = list(range(n_classes))
    return {
        'yte': yte, 'ypr': ypr, 'logits_te': logits_te,
        'val_macro_f1':             float(best_va),
        'test_macro_f1':            float(f1_score(yte, ypr, average='macro',
                                                   labels=present, zero_division=0)),
        'test_macro_f1_full_vocab': float(f1_score(yte, ypr, average='macro',
                                                   labels=full,    zero_division=0)),
        'test_weighted_f1':         float(f1_score(yte, ypr, average='weighted',
                                                   labels=present, zero_division=0)),
        'test_accuracy':            float(accuracy_score(yte, ypr)),
        'test_balanced_acc':        float(balanced_accuracy_score(yte, ypr)),
        'elapsed_s': float(elapsed),
        'class_names': class_names,
        'test_classes_present': present,
        'model': model, 'scaler': sc,
    }


# ── Split helpers ─────────────────────────────────────────────────────────────
def split_by_group(meta_subset, test_size=0.2, val_size=0.2, random_state=42):
    groups = meta_subset['group_key'].values
    gss1   = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trval, te = next(gss1.split(np.zeros(len(meta_subset)), groups=groups))
    gss2   = GroupShuffleSplit(n_splits=1,
                               test_size=val_size/(1-test_size),
                               random_state=random_state)
    tr_r, va_r = next(gss2.split(np.zeros(len(trval)), groups=groups[trval]))
    return trval[tr_r], trval[va_r], te


def split_train_val(meta_subset, val_size=0.1, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    tr, va = next(gss.split(np.zeros(len(meta_subset)),
                             groups=meta_subset['group_key'].values))
    return tr, va


def get_embeddings(meta_subset, row_col='audio_row'):
    rows = meta_subset[row_col].values
    return np.asarray(X_all[rows], dtype=np.float32)


# ── Report helper ─────────────────────────────────────────────────────────────
def report_run(name, r, save_fig=True):
    sep = '=' * 72
    print(f"\n{sep}\n  {name}\n{sep}")
    print(f"  val_macro_f1               = {r['val_macro_f1']:.3f}")
    print(f"  test_macro_f1 (test-only)  = {r['test_macro_f1']:.3f}")
    print(f"  test_macro_f1 (full vocab) = {r['test_macro_f1_full_vocab']:.3f}")
    print(f"  test_weighted_f1           = {r['test_weighted_f1']:.3f}")
    print(f"  test_accuracy              = {r['test_accuracy']:.3f}")
    print(f"  test_balanced_acc          = {r['test_balanced_acc']:.3f}")
    print(f"  trained in {r['elapsed_s']:.1f}s")

    classes = r.get('class_names')
    present = r.get('test_classes_present')
    if classes and present is not None:
        present_names = [classes[i] for i in present]
        print(f"\n  test classes ({len(present)}): {present_names}\n")
        print(classification_report(r['yte'], r['ypr'],
                                    labels=present, target_names=present_names,
                                    digits=3, zero_division=0))

    if save_fig and classes and len(classes) <= 25:
        cm      = confusion_matrix(r['yte'], r['ypr'],
                                   labels=list(range(len(classes))))
        cm_rows = cm[present, :] if present else cm
        ylab    = [classes[i] for i in present] if present else classes
        cm_norm = cm_rows / np.clip(cm_rows.sum(axis=1, keepdims=True), 1, None)
        fig, ax = plt.subplots(figsize=(max(6, 0.45*len(classes)+3),
                                        max(4, 0.45*len(ylab)+2)))
        sns.heatmap(cm_norm, annot=cm_rows, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=ylab, ax=ax, cbar=False)
        ax.set_xlabel('predicted'); ax.set_ylabel('true')
        ax.set_title(f'{name}')
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
        plt.tight_layout()
        fname = FIG_DIR / (name.replace(' ', '_').replace('/', '-') + '.png')
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"  figure → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# EXP 1 — Within-corpus joint baseline
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*72)
print("  EXP 1 — within-corpus joint baseline")
print("█"*72)

tr_pos, va_pos, te_pos = split_by_group(meta_all, test_size=0.3,
                                         val_size=0.2, random_state=42)
meta_tr = meta_all.iloc[tr_pos].copy()
meta_va = meta_all.iloc[va_pos].copy()
meta_te = meta_all.iloc[te_pos].copy()
print(f"split: train={len(tr_pos):,}  val={len(va_pos):,}  test={len(te_pos):,}")

le1 = LabelEncoder().fit(sorted(meta_all['label_joint'].unique()))
ytr1 = le1.transform(meta_tr['label_joint'])
yva1 = le1.transform(meta_va['label_joint'])
yte1 = le1.transform(meta_te['label_joint'])

r_exp1 = train_eval_mlp(
    get_embeddings(meta_tr), ytr1,
    get_embeddings(meta_va), yva1,
    get_embeddings(meta_te), yte1,
    n_classes=len(le1.classes_),
    class_names=list(le1.classes_),
)
report_run("Exp 1 — within-corpus joint (student emb)", r_exp1)

# ══════════════════════════════════════════════════════════════════════════════
# EXP 2 — Joint LODO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*72)
print("  EXP 2 — joint LODO")
print("█"*72)

JOINT_LODO_HOLDOUTS = [
    'ALNITAK_CAVANILLES', 'ECOSS_testtrain', 'ECOSS_enhanced', 'ECOSS_annot',
    'DCLDE_2026', 'DOLPHINFREE', 'DRYAD',
    'Adriatic_Sea', 'OLTREMARE', 'MONISH',
]
# WATKINS kept in training — rare species anchor; not held out


def run_joint_lodo(held_out, min_train_per_class=5):
    tr_m  = meta_all[meta_all['dataset'] != held_out].copy()
    te_m  = meta_all[meta_all['dataset'] == held_out].copy()

    counts      = tr_m['label_joint'].value_counts()
    train_keep  = set(counts[counts >= min_train_per_class].index)
    test_keep   = set(te_m['label_joint'].unique()) & train_keep
    test_dropped = set(te_m['label_joint'].unique()) - train_keep

    print(f"\n  [{held_out}]  train classes: {sorted(train_keep)}")
    if test_dropped:
        print(f"  [{held_out}]  test classes not in training (dropped): {sorted(test_dropped)}")
    for c, n in te_m['label_joint'].value_counts().items():
        tag = 'OK' if c in train_keep else 'DROPPED'
        print(f"      {c:32s}  n={n:>5d}  ({tag})")

    if len(train_keep) < 2 or len(test_keep) < 1:
        print(f"  [{held_out}]  skipping — insufficient classes")
        return None

    tr_m = tr_m[tr_m['label_joint'].isin(train_keep)].copy()
    te_m = te_m[te_m['label_joint'].isin(test_keep)].copy()

    le   = LabelEncoder().fit(sorted(train_keep))
    tr_m = tr_m.assign(y=le.transform(tr_m['label_joint'].astype(str)))
    te_m = te_m.assign(y=le.transform(te_m['label_joint'].astype(str)))

    tr_pos, va_pos = split_train_val(tr_m, val_size=0.1, random_state=42)
    print(f"  [{held_out}]  train={len(tr_pos):,}  val={len(va_pos):,}  test={len(te_m):,}")

    r = train_eval_mlp(
        get_embeddings(tr_m.iloc[tr_pos]), tr_m.iloc[tr_pos]['y'].values,
        get_embeddings(tr_m.iloc[va_pos]), tr_m.iloc[va_pos]['y'].values,
        get_embeddings(te_m),              te_m['y'].values,
        n_classes=len(le.classes_),
        class_names=list(le.classes_),
    )

    # Species-restricted metrics
    sp_ix    = [i for i, c in enumerate(le.classes_) if c != NON_MAMMAL_JOINT]
    sp_mask  = np.isin(te_m['y'].values, sp_ix)
    yte_full = te_m['y'].values
    if sp_mask.sum() > 0:
        sp_present = sorted(np.unique(yte_full[sp_mask]).tolist())
        bg_idx     = int(le.transform([NON_MAMMAL_JOINT])[0]) \
                     if NON_MAMMAL_JOINT in le.classes_ else -1
        r['sp_n_test_rows']       = int(sp_mask.sum())
        r['sp_n_classes_present'] = len(sp_present)
        r['sp_macro_f1']          = float(f1_score(yte_full[sp_mask],
                                                    r['ypr'][sp_mask],
                                                    average='macro',
                                                    labels=sp_present,
                                                    zero_division=0))
        r['sp_balanced_acc']      = float(balanced_accuracy_score(
                                              yte_full[sp_mask], r['ypr'][sp_mask]))
        r['sp_frac_routed_to_bg'] = float((r['ypr'][sp_mask] == bg_idx).mean()) \
                                    if bg_idx >= 0 else 0.0
    else:
        r['sp_n_test_rows'] = r['sp_n_classes_present'] = 0
        r['sp_macro_f1'] = r['sp_balanced_acc'] = r['sp_frac_routed_to_bg'] = None

    r['held_out'] = held_out
    return r


results_joint_lodo = {}
for ds in tqdm(JOINT_LODO_HOLDOUTS, desc='joint-LODO'):
    r = run_joint_lodo(ds)
    if r is not None:
        results_joint_lodo[ds] = r

rows2 = []
for ds, r in results_joint_lodo.items():
    rows2.append({
        'held_out':                  ds,
        'n_train':                   r.get('n_train', '—'),
        'n_test':                    len(r['yte']),
        'val_macro_f1':              r['val_macro_f1'],
        'test_macro_f1 (test-only)': r['test_macro_f1'],
        'test_macro_f1 (full vocab)':r['test_macro_f1_full_vocab'],
        'test_accuracy':             r['test_accuracy'],
        'test_balanced_acc':         r['test_balanced_acc'],
        'sp_n_rows':                 r['sp_n_test_rows'],
        'sp_n_classes':              r['sp_n_classes_present'],
        'sp_macro_f1':               r['sp_macro_f1'],
        'sp_frac_routed_to_bg':      r['sp_frac_routed_to_bg'],
    })

summary2 = pd.DataFrame(rows2).sort_values('test_macro_f1 (test-only)', ascending=False)
print("\n\nEXP 2 SUMMARY")
print(summary2.to_string(index=False))
summary2.to_csv(OUT_DIR / 'exp2_joint_lodo_summary.csv', index=False)
print(f"\nAverage test_macro_f1 (test-only): "
      f"{summary2['test_macro_f1 (test-only)'].mean():.3f}")

for ds, r in results_joint_lodo.items():
    report_run(f"Exp2 joint LODO held-out {ds}", r)

# ══════════════════════════════════════════════════════════════════════════════
# EXP 3 — Pure species LODO (4 clean holdouts)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█"*72)
print("  EXP 3 — pure species LODO")
print("█"*72)

SPECIES_LODO_HOLDOUTS = ['WATKINS', 'MONISH', 'ECOSS_testtrain', 'ALNITAK_CAVANILLES']
SPECIES_LODO_EXCLUDE  = {
    'WATKINS': {'Balaenoptera_acutorostrata'},   # 17 rows, single dataset
}
sp_meta_full = meta_all[meta_all['coarse_class'].isin(SPECIES)].copy()
print(f"species-only rows: {len(sp_meta_full):,}")


def run_species_lodo(held_out, min_train_per_class=5):
    exclude    = SPECIES_LODO_EXCLUDE.get(held_out, set())
    tr_m       = sp_meta_full[sp_meta_full['dataset'] != held_out].copy()
    te_m       = sp_meta_full[sp_meta_full['dataset'] == held_out].copy()

    if exclude:
        tr_m = tr_m[~tr_m['coarse_class'].isin(exclude)].copy()
        te_m = te_m[~te_m['coarse_class'].isin(exclude)].copy()

    counts     = tr_m['coarse_class'].value_counts()
    train_keep = set(counts[counts >= min_train_per_class].index)
    test_keep  = set(te_m['coarse_class'].unique()) & train_keep
    test_drop  = set(te_m['coarse_class'].unique()) - train_keep

    print(f"\n  [{held_out}]  train species: {sorted(train_keep)}")
    if test_drop:
        print(f"  [{held_out}]  test species not in training (dropped): {sorted(test_drop)}")
    for c, n in te_m['coarse_class'].value_counts().items():
        tag = 'OK' if c in train_keep else 'DROPPED'
        print(f"      {c:35s}  n={n:>4d}  ({tag})")

    if len(train_keep) < 2 or len(test_keep) < 1:
        print(f"  [{held_out}]  skipping — insufficient classes")
        return None

    tr_m = tr_m[tr_m['coarse_class'].isin(train_keep)].copy()
    te_m = te_m[te_m['coarse_class'].isin(test_keep)].copy()

    le   = LabelEncoder().fit(sorted(train_keep))
    tr_m = tr_m.assign(y=le.transform(tr_m['coarse_class'].astype(str)))
    te_m = te_m.assign(y=le.transform(te_m['coarse_class'].astype(str)))

    tr_pos, va_pos = split_train_val(tr_m, val_size=0.1, random_state=42)
    print(f"  [{held_out}]  train={len(tr_pos):,}  val={len(va_pos):,}  test={len(te_m):,}")

    r = train_eval_mlp(
        get_embeddings(tr_m.iloc[tr_pos]), tr_m.iloc[tr_pos]['y'].values,
        get_embeddings(tr_m.iloc[va_pos]), tr_m.iloc[va_pos]['y'].values,
        get_embeddings(te_m),              te_m['y'].values,
        n_classes=len(le.classes_),
        class_names=list(le.classes_),
    )
    r['held_out'] = held_out
    return r


results_species_lodo = {}
for ds in tqdm(SPECIES_LODO_HOLDOUTS, desc='species-LODO'):
    r = run_species_lodo(ds)
    if r is not None:
        results_species_lodo[ds] = r

rows3 = []
for ds, r in results_species_lodo.items():
    rows3.append({
        'held_out':                  ds,
        'n_test':                    len(r['yte']),
        'n_test_classes':            len(r['test_classes_present']),
        'val_macro_f1':              r['val_macro_f1'],
        'test_macro_f1 (test-only)': r['test_macro_f1'],
        'test_macro_f1 (full vocab)':r['test_macro_f1_full_vocab'],
        'test_accuracy':             r['test_accuracy'],
        'test_balanced_acc':         r['test_balanced_acc'],
    })

summary3 = pd.DataFrame(rows3).sort_values('test_macro_f1 (test-only)', ascending=False)
print("\n\nEXP 3 SUMMARY")
print(summary3.to_string(index=False))
summary3.to_csv(OUT_DIR / 'exp3_species_lodo_summary.csv', index=False)
print(f"\nAverage test_macro_f1 (test-only): "
      f"{summary3['test_macro_f1 (test-only)'].mean():.3f}")

for ds, r in results_species_lodo.items():
    report_run(f"Exp3 species LODO held-out {ds}", r)

# ── Save combined summary JSON ─────────────────────────────────────────────────
out = {
    'exp1': {k: float(v) if isinstance(v, (np.floating, float)) else v
             for k, v in r_exp1.items()
             if k not in ('yte','ypr','logits_te','model','scaler','class_names',
                          'test_classes_present')},
    'exp2_avg_test_macro_f1': float(summary2['test_macro_f1 (test-only)'].mean()),
    'exp3_avg_test_macro_f1': float(summary3['test_macro_f1 (test-only)'].mean()),
    'exp2_per_fold': {ds: {k: (float(v) if isinstance(v,(np.floating,float))
                               else (None if v is None else v))
                            for k, v in r.items()
                            if k not in ('yte','ypr','logits_te','model','scaler',
                                         'class_names','test_classes_present',
                                         'rows_te')}
                      for ds, r in results_joint_lodo.items()},
    'exp3_per_fold': {ds: {k: (float(v) if isinstance(v,(np.floating,float))
                               else (None if v is None else v))
                            for k, v in r.items()
                            if k not in ('yte','ypr','logits_te','model','scaler',
                                         'class_names','test_classes_present')}
                      for ds, r in results_species_lodo.items()},
}
with open(OUT_DIR / 'lodo_summary.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nAll results → {OUT_DIR}")
print("Done.")
