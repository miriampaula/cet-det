#!/usr/bin/env python3
"""
Student-teacher LODO evaluation — one-vs-rest MLP variant.

Instead of a single N-way MLP, trains one binary MLP per species
(species vs everything else). At inference, the species with the
highest positive-class probability wins; if all are below threshold
the sample is labelled background.

Usage:
    python 8-student-lodo-one-vs-rest.py /path/to/run_a /path/to/run_b
    python 8-student-lodo-one-vs-rest.py /path/to/run_a --meta /my/meta.parquet --out /my/out
    python 8-student-lodo-one-vs-rest.py /path/to/run_a --threshold 0.4
"""

import argparse
import os, json, time, warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_META_PATH = Path('/data2/mromaniuc/cet-det/student_teacher/meta_train_with_paths.parquet')
DEFAULT_OUT_DIR   = Path('/data2/mromaniuc/cet-det/student_teacher/lodo_results_ovr')
EMB_FILENAME      = 'X_student_emb.npy'
DEFAULT_THRESHOLD = 0.5   # minimum positive-class prob to assign a species label

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SPECIES = {
    'Balaenoptera_physalus', 'Delphinus_delphis', 'Delphinids',
    'Globicephala_melas', 'Grampus_griseus', 'Orcinus_orca',
    'Physeter_macrocephalus', 'Stenella_coeruleoalba', 'Tursiops_truncatus',
}
NON_MAMMAL_JOINT = 'background'

JOINT_LODO_HOLDOUTS = [
    'ALNITAK_CAVANILLES', 'ECOSS_testtrain', 'ECOSS_enhanced', 'ECOSS_annot',
    'DCLDE_2026', 'DOLPHINFREE', 'DRYAD',
    'Adriatic_Sea', 'OLTREMARE', 'MONISH',
]
SPECIES_LODO_HOLDOUTS = ['WATKINS', 'MONISH', 'ECOSS_testtrain', 'ALNITAK_CAVANILLES']
SPECIES_LODO_EXCLUDE  = {
    'WATKINS': {'Balaenoptera_acutorostrata'},
}


# ══════════════════════════════════════════════════════════════════════════════
# Binary MLP (one per species)
# ══════════════════════════════════════════════════════════════════════════════
class BinaryMLP(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256), dropout=0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))   # single logit → sigmoid prob
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)       # (B,)


def train_binary_mlp(Xtr, ytr_bin, Xva, yva_bin,
                     lr=1e-3, weight_decay=1e-4, batch_size=1024,
                     epochs=40, patience=6, verbose=False):
    """Train one binary classifier. ytr_bin / yva_bin are 0/1 float arrays."""
    model = BinaryMLP(Xtr.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # class weight: penalise false negatives harder (species are always minority)
    pos_weight = torch.tensor(
        [(ytr_bin == 0).sum() / max((ytr_bin == 1).sum(), 1)],
        dtype=torch.float32, device=DEVICE,
    )
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tr_dl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr),
                      torch.from_numpy(ytr_bin.astype(np.float32))),
        batch_size=batch_size, shuffle=True, num_workers=0)
    va_dl = DataLoader(
        TensorDataset(torch.from_numpy(Xva),
                      torch.from_numpy(yva_bin.astype(np.float32))),
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
        probs, trues = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                probs.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
                trues.append(yb.numpy())
        probs = np.concatenate(probs)
        trues = np.concatenate(trues)
        preds = (probs >= 0.5).astype(int)
        va_f1 = f1_score(trues, preds, average='binary', zero_division=0)
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


def predict_proba(model, X, batch_size=2048):
    """Return positive-class probabilities, shape (N,)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(
                torch.sigmoid(
                    model(torch.from_numpy(X[i:i+batch_size]).to(DEVICE))
                ).cpu().numpy()
            )
    return np.concatenate(out)


# ══════════════════════════════════════════════════════════════════════════════
# OVR ensemble: train one binary MLP per species, combine at inference
# ══════════════════════════════════════════════════════════════════════════════
def train_eval_ovr(Xtr, ytr, Xva, yva, Xte, yte,
                   class_names, species_indices, bg_index,
                   threshold=DEFAULT_THRESHOLD, verbose=False):
    """
    class_names    : list of all class names (including background)
    species_indices: list of integer indices that are cetacean species
    bg_index       : integer index of the background class
    threshold      : min prob for a species to be assigned (else → background)
    """
    sc    = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr).astype(np.float32)
    Xva_s = sc.transform(Xva).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)

    n_classes   = len(class_names)
    t0          = time.time()

    # Train one binary MLP per species (not for background — it's the fallback)
    prob_matrix_te = np.zeros((len(Xte_s), n_classes), dtype=np.float32)
    prob_matrix_te[:, bg_index] = 1.0   # default: everything is background

    binary_val_f1s = {}
    for sp_idx in species_indices:
        sp_name   = class_names[sp_idx]
        ytr_bin   = (ytr == sp_idx).astype(np.float32)
        yva_bin   = (yva == sp_idx).astype(np.float32)

        if ytr_bin.sum() < 5:
            print(f"    [OVR] skipping {sp_name} — only {int(ytr_bin.sum())} train positives")
            continue

        model, val_f1 = train_binary_mlp(Xtr_s, ytr_bin, Xva_s, yva_bin, verbose=verbose)
        binary_val_f1s[sp_name] = float(val_f1)
        prob_matrix_te[:, sp_idx] = predict_proba(model, Xte_s)

        if verbose:
            print(f"    [OVR] {sp_name:35s}  val_f1={val_f1:.3f}  "
                  f"n_pos_train={int(ytr_bin.sum())}")

    # Decision: argmax over species columns only if above threshold;
    # otherwise background
    sp_probs  = prob_matrix_te[:, species_indices]          # (N, n_species)
    sp_maxval = sp_probs.max(axis=1)                        # (N,)
    sp_argmax = np.array(species_indices)[sp_probs.argmax(axis=1)]  # (N,)

    ypr = np.where(sp_maxval >= threshold, sp_argmax, bg_index).astype(int)

    elapsed = time.time() - t0
    present = sorted(np.unique(yte).tolist())
    full    = list(range(n_classes))

    return {
        'yte': yte, 'ypr': ypr, 'prob_matrix_te': prob_matrix_te,
        'val_macro_f1':             float(np.mean(list(binary_val_f1s.values()))
                                         if binary_val_f1s else 0.0),
        'binary_val_f1s':           binary_val_f1s,
        'test_macro_f1':            float(f1_score(yte, ypr, average='macro',
                                                   labels=present, zero_division=0)),
        'test_macro_f1_full_vocab': float(f1_score(yte, ypr, average='macro',
                                                   labels=full,    zero_division=0)),
        'test_weighted_f1':         float(f1_score(yte, ypr, average='weighted',
                                                   labels=present, zero_division=0)),
        'test_accuracy':            float(accuracy_score(yte, ypr)),
        'test_balanced_acc':        float(balanced_accuracy_score(yte, ypr)),
        'elapsed_s':                float(elapsed),
        'class_names':              class_names,
        'test_classes_present':     present,
        'threshold':                threshold,
        'scaler':                   sc,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Split helpers
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# Report helper
# ══════════════════════════════════════════════════════════════════════════════
def report_run(name, r, fig_dir, save_fig=True):
    sep = '=' * 72
    print(f"\n{sep}\n  {name}\n{sep}")
    print(f"  avg binary val_f1          = {r['val_macro_f1']:.3f}")
    print(f"  test_macro_f1 (test-only)  = {r['test_macro_f1']:.3f}")
    print(f"  test_macro_f1 (full vocab) = {r['test_macro_f1_full_vocab']:.3f}")
    print(f"  test_weighted_f1           = {r['test_weighted_f1']:.3f}")
    print(f"  test_accuracy              = {r['test_accuracy']:.3f}")
    print(f"  test_balanced_acc          = {r['test_balanced_acc']:.3f}")
    print(f"  threshold                  = {r['threshold']:.2f}")
    print(f"  trained in {r['elapsed_s']:.1f}s")

    if r.get('binary_val_f1s'):
        print(f"\n  per-species binary val F1:")
        for sp, f1 in sorted(r['binary_val_f1s'].items()):
            print(f"    {sp:35s}  {f1:.3f}")

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
        fname = fig_dir / (name.replace(' ', '_').replace('/', '-') + '.png')
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"  figure → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-run evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_run(X_all, meta_all, run_name, out_dir, threshold):
    run_out = out_dir / run_name
    fig_dir = run_out / 'figures'
    run_out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    print(f"\n{'█'*72}")
    print(f"  RUN: {run_name}  (OVR, threshold={threshold:.2f})")
    print(f"{'█'*72}")

    def get_embeddings(meta_subset):
        return np.asarray(X_all[meta_subset['audio_row'].values], dtype=np.float32)

    # ── Exp 1 — within-corpus ─────────────────────────────────────────────────
    print(f"\n{'─'*60}\n  EXP 1 — within-corpus joint baseline (OVR)\n{'─'*60}")
    tr_pos, va_pos, te_pos = split_by_group(meta_all, test_size=0.3,
                                             val_size=0.2, random_state=42)
    meta_tr = meta_all.iloc[tr_pos]
    meta_va = meta_all.iloc[va_pos]
    meta_te = meta_all.iloc[te_pos]

    le1          = LabelEncoder().fit(sorted(meta_all['label_joint'].unique()))
    class_names1 = list(le1.classes_)
    sp_idx1      = [i for i, c in enumerate(class_names1) if c != NON_MAMMAL_JOINT]
    bg_idx1      = int(le1.transform([NON_MAMMAL_JOINT])[0])

    r_e1 = train_eval_ovr(
        get_embeddings(meta_tr), le1.transform(meta_tr['label_joint']),
        get_embeddings(meta_va), le1.transform(meta_va['label_joint']),
        get_embeddings(meta_te), le1.transform(meta_te['label_joint']),
        class_names=class_names1, species_indices=sp_idx1, bg_index=bg_idx1,
        threshold=threshold,
    )
    report_run(f"{run_name} — Exp1 OVR within-corpus", r_e1, fig_dir)

    # ── Exp 2 — joint LODO ───────────────────────────────────────────────────
    print(f"\n{'─'*60}\n  EXP 2 — joint LODO (OVR)\n{'─'*60}")

    def run_joint_lodo(held_out, min_train_per_class=5):
        tr_m = meta_all[meta_all['dataset'] != held_out].copy()
        te_m = meta_all[meta_all['dataset'] == held_out].copy()

        counts     = tr_m['label_joint'].value_counts()
        train_keep = set(counts[counts >= min_train_per_class].index)
        test_keep  = set(te_m['label_joint'].unique()) & train_keep
        test_drop  = set(te_m['label_joint'].unique()) - train_keep

        print(f"\n  [{held_out}]  train classes: {sorted(train_keep)}")
        if test_drop:
            print(f"  [{held_out}]  dropped from test: {sorted(test_drop)}")

        if len(train_keep) < 2 or len(test_keep) < 1:
            print(f"  [{held_out}]  skipping — insufficient classes")
            return None

        tr_m = tr_m[tr_m['label_joint'].isin(train_keep)].copy()
        te_m = te_m[te_m['label_joint'].isin(test_keep)].copy()
        le   = LabelEncoder().fit(sorted(train_keep))
        tr_m = tr_m.assign(y=le.transform(tr_m['label_joint'].astype(str)))
        te_m = te_m.assign(y=le.transform(te_m['label_joint'].astype(str)))

        class_names = list(le.classes_)
        sp_indices  = [i for i, c in enumerate(class_names) if c != NON_MAMMAL_JOINT]
        bg_idx      = int(le.transform([NON_MAMMAL_JOINT])[0]) \
                      if NON_MAMMAL_JOINT in class_names else -1

        tr_pos, va_pos = split_train_val(tr_m, val_size=0.1, random_state=42)
        print(f"  [{held_out}]  train={len(tr_pos):,}  "
              f"val={len(va_pos):,}  test={len(te_m):,}  "
              f"species={[class_names[i] for i in sp_indices]}")

        r = train_eval_ovr(
            get_embeddings(tr_m.iloc[tr_pos]), tr_m.iloc[tr_pos]['y'].values,
            get_embeddings(tr_m.iloc[va_pos]), tr_m.iloc[va_pos]['y'].values,
            get_embeddings(te_m),              te_m['y'].values,
            class_names=class_names, species_indices=sp_indices, bg_index=bg_idx,
            threshold=threshold,
        )

        # Species-restricted metrics
        sp_mask = np.isin(te_m['y'].values, sp_indices)
        yte_f   = te_m['y'].values
        if sp_mask.sum() > 0:
            sp_present = sorted(np.unique(yte_f[sp_mask]).tolist())
            r.update({
                'sp_n_test_rows':       int(sp_mask.sum()),
                'sp_n_classes_present': len(sp_present),
                'sp_macro_f1':          float(f1_score(yte_f[sp_mask], r['ypr'][sp_mask],
                                                        average='macro', labels=sp_present,
                                                        zero_division=0)),
                'sp_balanced_acc':      float(balanced_accuracy_score(
                                                  yte_f[sp_mask], r['ypr'][sp_mask])),
                'sp_frac_routed_to_bg': float((r['ypr'][sp_mask] == bg_idx).mean())
                                        if bg_idx >= 0 else 0.0,
            })
        else:
            r.update({'sp_n_test_rows': 0, 'sp_n_classes_present': 0,
                      'sp_macro_f1': None, 'sp_balanced_acc': None,
                      'sp_frac_routed_to_bg': None})
        r['held_out'] = held_out
        return r

    results_joint = {}
    for ds in tqdm(JOINT_LODO_HOLDOUTS, desc=f'{run_name} joint-LODO OVR'):
        r = run_joint_lodo(ds)
        if r is not None:
            results_joint[ds] = r

    rows2 = [{
        'held_out':                   ds,
        'n_test':                     len(r['yte']),
        'avg_binary_val_f1':          r['val_macro_f1'],
        'test_macro_f1 (test-only)':  r['test_macro_f1'],
        'test_macro_f1 (full vocab)': r['test_macro_f1_full_vocab'],
        'test_accuracy':              r['test_accuracy'],
        'test_balanced_acc':          r['test_balanced_acc'],
        'sp_n_rows':                  r['sp_n_test_rows'],
        'sp_n_classes':               r['sp_n_classes_present'],
        'sp_macro_f1':                r['sp_macro_f1'],
        'sp_frac_routed_to_bg':       r['sp_frac_routed_to_bg'],
    } for ds, r in results_joint.items()]
    summary2 = pd.DataFrame(rows2).sort_values('test_macro_f1 (test-only)', ascending=False)
    print(f"\n\n{run_name} — EXP 2 OVR SUMMARY")
    print(summary2.to_string(index=False))
    print(f"Average test_macro_f1: {summary2['test_macro_f1 (test-only)'].mean():.3f}")
    summary2.to_csv(run_out / 'exp2_joint_lodo_ovr_summary.csv', index=False)

    for ds, r in results_joint.items():
        report_run(f"{run_name} Exp2-OVR held-out {ds}", r, fig_dir)

    # ── Exp 3 — pure species LODO ─────────────────────────────────────────────
    print(f"\n{'─'*60}\n  EXP 3 — pure species LODO (OVR)\n{'─'*60}")
    sp_meta = meta_all[meta_all['coarse_class'].isin(SPECIES)].copy()

    def run_species_lodo(held_out, min_train_per_class=5):
        exclude = SPECIES_LODO_EXCLUDE.get(held_out, set())
        tr_m    = sp_meta[sp_meta['dataset'] != held_out].copy()
        te_m    = sp_meta[sp_meta['dataset'] == held_out].copy()
        if exclude:
            tr_m = tr_m[~tr_m['coarse_class'].isin(exclude)].copy()
            te_m = te_m[~te_m['coarse_class'].isin(exclude)].copy()

        counts     = tr_m['coarse_class'].value_counts()
        train_keep = set(counts[counts >= min_train_per_class].index)
        test_keep  = set(te_m['coarse_class'].unique()) & train_keep
        test_drop  = set(te_m['coarse_class'].unique()) - train_keep

        if test_drop:
            print(f"  [{held_out}]  dropped: {sorted(test_drop)}")
        if len(train_keep) < 2 or len(test_keep) < 1:
            print(f"  [{held_out}]  skipping — insufficient classes")
            return None

        tr_m = tr_m[tr_m['coarse_class'].isin(train_keep)].copy()
        te_m = te_m[te_m['coarse_class'].isin(test_keep)].copy()
        le   = LabelEncoder().fit(sorted(train_keep))
        tr_m = tr_m.assign(y=le.transform(tr_m['coarse_class'].astype(str)))
        te_m = te_m.assign(y=le.transform(te_m['coarse_class'].astype(str)))

        class_names = list(le.classes_)
        # In pure species LODO there's no background class — every class is a species
        # so OVR still applies: each species vs all others
        sp_indices  = list(range(len(class_names)))
        # Use argmax only (no background fallback) — threshold ignored here
        sc    = StandardScaler().fit(get_embeddings(tr_m))

        tr_pos, va_pos = split_train_val(tr_m, val_size=0.1, random_state=42)
        Xtr_s = sc.transform(get_embeddings(tr_m.iloc[tr_pos])).astype(np.float32)
        Xva_s = sc.transform(get_embeddings(tr_m.iloc[va_pos])).astype(np.float32)
        Xte_s = sc.transform(get_embeddings(te_m)).astype(np.float32)
        ytr   = tr_m.iloc[tr_pos]['y'].values
        yva   = tr_m.iloc[va_pos]['y'].values
        yte   = te_m['y'].values

        print(f"  [{held_out}]  train={len(ytr):,}  val={len(yva):,}  test={len(yte):,}")

        t0 = time.time()
        prob_matrix_te  = np.zeros((len(Xte_s), len(class_names)), dtype=np.float32)
        binary_val_f1s  = {}
        for sp_idx in sp_indices:
            sp_name  = class_names[sp_idx]
            ytr_bin  = (ytr == sp_idx).astype(np.float32)
            yva_bin  = (yva == sp_idx).astype(np.float32)
            if ytr_bin.sum() < 5:
                continue
            model, val_f1 = train_binary_mlp(Xtr_s, ytr_bin, Xva_s, yva_bin)
            binary_val_f1s[sp_name]       = float(val_f1)
            prob_matrix_te[:, sp_idx]     = predict_proba(model, Xte_s)

        ypr     = prob_matrix_te.argmax(axis=1)
        present = sorted(np.unique(yte).tolist())
        full    = list(range(len(class_names)))
        r = {
            'yte': yte, 'ypr': ypr,
            'val_macro_f1':             float(np.mean(list(binary_val_f1s.values()))
                                             if binary_val_f1s else 0.0),
            'binary_val_f1s':           binary_val_f1s,
            'test_macro_f1':            float(f1_score(yte, ypr, average='macro',
                                                       labels=present, zero_division=0)),
            'test_macro_f1_full_vocab': float(f1_score(yte, ypr, average='macro',
                                                       labels=full, zero_division=0)),
            'test_weighted_f1':         float(f1_score(yte, ypr, average='weighted',
                                                       labels=present, zero_division=0)),
            'test_accuracy':            float(accuracy_score(yte, ypr)),
            'test_balanced_acc':        float(balanced_accuracy_score(yte, ypr)),
            'elapsed_s':                float(time.time() - t0),
            'class_names':              class_names,
            'test_classes_present':     present,
            'threshold':                None,
            'held_out':                 held_out,
        }
        return r

    results_species = {}
    for ds in tqdm(SPECIES_LODO_HOLDOUTS, desc=f'{run_name} species-LODO OVR'):
        r = run_species_lodo(ds)
        if r is not None:
            results_species[ds] = r

    rows3 = [{
        'held_out':                   ds,
        'n_test':                     len(r['yte']),
        'n_test_classes':             len(r['test_classes_present']),
        'avg_binary_val_f1':          r['val_macro_f1'],
        'test_macro_f1 (test-only)':  r['test_macro_f1'],
        'test_macro_f1 (full vocab)': r['test_macro_f1_full_vocab'],
        'test_accuracy':              r['test_accuracy'],
        'test_balanced_acc':          r['test_balanced_acc'],
    } for ds, r in results_species.items()]
    summary3 = pd.DataFrame(rows3).sort_values('test_macro_f1 (test-only)', ascending=False)
    print(f"\n\n{run_name} — EXP 3 OVR SUMMARY")
    print(summary3.to_string(index=False))
    print(f"Average test_macro_f1: {summary3['test_macro_f1 (test-only)'].mean():.3f}")
    summary3.to_csv(run_out / 'exp3_species_lodo_ovr_summary.csv', index=False)

    for ds, r in results_species.items():
        report_run(f"{run_name} Exp3-OVR held-out {ds}", r, fig_dir)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    def _clean(d):
        skip = {'yte', 'ypr', 'prob_matrix_te', 'model', 'scaler',
                'class_names', 'test_classes_present', 'rows_te'}
        return {k: (float(v) if isinstance(v, (np.floating, float))
                    else (None if v is None else v))
                for k, v in d.items() if k not in skip}

    out_json = {
        'run':       run_name,
        'threshold': threshold,
        'exp1':      _clean(r_e1),
        'exp2_avg_test_macro_f1': float(summary2['test_macro_f1 (test-only)'].mean()),
        'exp3_avg_test_macro_f1': float(summary3['test_macro_f1 (test-only)'].mean()),
        'exp2_per_fold': {ds: _clean(r) for ds, r in results_joint.items()},
        'exp3_per_fold': {ds: _clean(r) for ds, r in results_species.items()},
    }
    with open(run_out / 'lodo_ovr_summary.json', 'w') as f:
        json.dump(out_json, f, indent=2)

    print(f"\n  Results → {run_out}")
    return out_json


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='OVR-MLP LODO evaluation for one or more student embedding runs.')
    parser.add_argument(
        'directories', nargs='+', type=Path,
        help=f'Directories each containing "{EMB_FILENAME}"')
    parser.add_argument(
        '--meta', type=Path, default=DEFAULT_META_PATH,
        help='Path to meta_train_with_paths.parquet (default: %(default)s)')
    parser.add_argument(
        '--out', type=Path, default=DEFAULT_OUT_DIR,
        help='Root output directory (default: %(default)s)')
    parser.add_argument(
        '--threshold', type=float, default=DEFAULT_THRESHOLD,
        help='Min positive-class prob to assign a species label (default: %(default)s)')
    args = parser.parse_args()

    print(f"device: {DEVICE}  torch: {torch.__version__}")
    print(f"OVR threshold: {args.threshold}")

    print(f"\nLoading metadata from {args.meta} ...")
    meta_all = pd.read_parquet(args.meta)
    assert 'audio_row' in meta_all.columns
    assert 'group_key' in meta_all.columns
    meta_all = meta_all.copy()
    meta_all['label_joint'] = meta_all['coarse_class'].apply(
        lambda c: c if c in SPECIES else NON_MAMMAL_JOINT)
    print(f"  {len(meta_all):,} rows")
    print(meta_all['label_joint'].value_counts().to_string())

    args.out.mkdir(parents=True, exist_ok=True)
    all_summaries = {}

    for run_dir in args.directories:
        run_dir  = run_dir.resolve()
        emb_path = run_dir / EMB_FILENAME
        if not emb_path.exists():
            print(f"\n[SKIP] {emb_path} not found")
            continue
        X_all = np.load(emb_path, mmap_mode='r')
        assert len(X_all) == len(meta_all), \
            f"Shape mismatch: {len(X_all)} emb vs {len(meta_all)} meta"
        summary = evaluate_run(X_all, meta_all,
                               run_name=run_dir.name,
                               out_dir=args.out,
                               threshold=args.threshold)
        all_summaries[run_dir.name] = summary

    if len(all_summaries) > 1:
        print(f"\n{'═'*72}\n  CROSS-RUN COMPARISON (OVR)\n{'═'*72}")
        rows = [{
            'run':                    rn,
            'exp1_test_macro_f1':     s['exp1']['test_macro_f1'],
            'exp1_test_balanced_acc': s['exp1']['test_balanced_acc'],
            'exp2_avg_macro_f1':      s['exp2_avg_test_macro_f1'],
            'exp3_avg_macro_f1':      s['exp3_avg_test_macro_f1'],
        } for rn, s in all_summaries.items()]
        comp = pd.DataFrame(rows)
        print(comp.to_string(index=False))
        comp.to_csv(args.out / 'cross_run_comparison_ovr.csv', index=False)

    print("\nAll done.")


if __name__ == '__main__':
    main()