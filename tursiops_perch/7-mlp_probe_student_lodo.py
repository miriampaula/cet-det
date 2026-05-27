#!/usr/bin/env python3
"""
MLP probing on student embeddings — matched LODO, Tursiops vs background.

For each X_all_emb.npy, trains an MLP probe using only the split that matches
the checkpoint's own held-out dataset (the scientifically meaningful test).

Outputs per checkpoint:
    runs_lodo/<held_out>/<variant>/mlp_probe/
        results.json
        confusion_<held_out>.png
        probe_model_<held_out>.pt

Summary:
    runs_lodo/mlp_probe_summary.csv

Usage:
    cd /data2/mromaniuc/cet-det/tursiops_perch


PYTHONUNBUFFERED=1 python -u 7-mlp_probe_student_lodo.py \
    --holdout Adriatic_Sea ALNITAK_CAVANILLES DRYAD OLTREMARE \
    --also-perch \
    > logs/mlp_probe_all.log 2>&1 &


    # Only the v03_no_grl checkpoints for specific folds:
    python 7-mlp_probe_student_lodo.py \
        --holdout ALNITAK_CAVANILLES DRYAD OLTREMARE \
        --variant v03_no_grl

    # Add Perch baseline for comparison:
    python 7-mlp_probe_student_lodo.py \
        --holdout ALNITAK_CAVANILLES DRYAD OLTREMARE \
        --variant v03_no_grl --also-perch

    # Re-run from scratch:
    python mlp_probe_student_lodo.py ... --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, confusion_matrix,
                              precision_score, recall_score)
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_META     = Path('student_teacher/meta_train_with_paths.parquet')
DEFAULT_RUNS     = Path('student_teacher/runs_lodo')
TEACHER_EMB_PATH = Path('student_teacher/X_teacher_emb.npy')

# ── Hyperparameters ───────────────────────────────────────────────────────────
EMB_DIM         = 1536
HIDDEN_DIMS     = [512, 256]
BATCH_SIZE      = 1024
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
MAX_EPOCHS      = 100
PATIENCE        = 10
THRESHOLD_SWEEP = np.linspace(0.05, 0.95, 37)
VAL_FRAC        = 0.15
DEVICE          = 'cuda'


# ── MLP probe ─────────────────────────────────────────────────────────────────

class MLPProbe(nn.Module):
    def __init__(self, in_dim=EMB_DIM, hidden_dims=HIDDEN_DIMS,
                 n_classes=2, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Data helpers ──────────────────────────────────────────────────────────────

def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=0)


def lodo_splits(meta, held_out_ds, val_frac=VAL_FRAC, seed=42):
    rng        = np.random.default_rng(seed)
    test_mask  = meta['dataset'] == held_out_ds
    train_pool = meta[~test_mask]

    val_idx_list = []
    for _, grp in train_pool.groupby('dataset'):
        n_val  = max(1, int(len(grp) * val_frac))
        chosen = rng.choice(grp.index.values, size=n_val, replace=False)
        val_idx_list.append(chosen)

    val_idx   = np.concatenate(val_idx_list)
    val_set   = set(val_idx)
    train_idx = train_pool.index[~train_pool.index.isin(val_set)].values
    test_idx  = meta.index[test_mask].values
    return train_idx, val_idx, test_idx


# ── Training ──────────────────────────────────────────────────────────────────

def train_probe(X_tr, y_tr, X_val, y_val, device):
    pos_w      = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    criterion  = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, float(pos_w)], device=device))

    model = MLPProbe().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    tr_loader  = make_loader(X_tr,  y_tr,  BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    best_f1, best_state, patience_left = -1.0, None, PATIENCE

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()

        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                p = torch.softmax(model(xb.to(device)), 1)[:, 1].cpu().numpy()
                val_probs.append(p); val_true.append(yb.numpy())
        val_probs = np.concatenate(val_probs)
        val_true  = np.concatenate(val_true)

        f1_epoch = max(
            f1_score(val_true, (val_probs >= t).astype(int),
                     average='macro', zero_division=0)
            for t in THRESHOLD_SWEEP
        )
        if f1_epoch > best_f1:
            best_f1   = f1_epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_probe(model, X_test, y_test, device):
    loader = make_loader(X_test, y_test, BATCH_SIZE, shuffle=False)
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = torch.softmax(model(xb.to(device)), 1)[:, 1].cpu().numpy()
            all_probs.append(p); all_true.append(yb.numpy())
    probs = np.concatenate(all_probs)
    true  = np.concatenate(all_true)

    n_pos = int(true.sum())
    auc   = roc_auc_score(true, probs) if 0 < n_pos < len(true) else float('nan')
    ap    = average_precision_score(true, probs) if n_pos > 0 else float('nan')

    best_f1, best_thr, best_prec, best_rec = 0.0, 0.5, 0.0, 0.0
    for t in THRESHOLD_SWEEP:
        preds = (probs >= t).astype(int)
        f1    = f1_score(true, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1   = f1
            best_thr  = float(t)
            best_prec = float(precision_score(true, preds, zero_division=0))
            best_rec  = float(recall_score(true, preds, zero_division=0))

    return (dict(n_test=len(true), n_pos=n_pos,
                 pct_pos=float(n_pos / len(true) * 100),
                 test_auc=float(auc), test_ap=float(ap),
                 test_f1=float(best_f1), precision=best_prec,
                 recall=best_rec, threshold=best_thr),
            probs, true, best_thr)


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def save_confusion(true, probs, threshold, out_png, title):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(true, preds)

    # normalised (row = true class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(title, fontsize=11)

    for ax, data, fmt, ttl in [
        (axes[0], cm,      'd',    'Counts'),
        (axes[1], cm_norm, '.2f',  'Row-normalised'),
    ]:
        im = ax.imshow(data, cmap='Blues', vmin=0,
                       vmax=None if fmt == 'd' else 1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred: bg', 'Pred: Tursiops'])
        ax.set_yticklabels(['True: bg', 'True: Tursiops'])
        ax.set_title(ttl, fontsize=10)
        for i in range(2):
            for j in range(2):
                val = data[i, j]
                txt = f'{val:{fmt}}' if fmt == 'd' else f'{val:.2f}'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=12, color='white' if val > (data.max() * 0.6) else 'black')

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Confusion matrix -> {out_png}')


# ── Probe runner (matched held-out only) ─────────────────────────────────────

def run_probe_for_checkpoint(emb_path, held_out_ckpt, variant_label,
                              meta, out_dir, device):
    """
    Trains probe on everything except held_out_ckpt, tests on held_out_ckpt only.
    This is the matched comparison: the embedding was extracted from a model
    that never saw held_out_ckpt during training.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(emb_path)
    y = meta['label'].values

    train_idx, val_idx, test_idx = lodo_splits(meta, held_out_ckpt)

    X_tr,  y_tr  = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx],   y[val_idx]
    X_te,  y_te  = X[test_idx],  y[test_idx]

    print(f'    train : {len(y_tr):,}  pos={y_tr.sum():,}')
    print(f'    val   : {len(y_val):,}  pos={y_val.sum():,}')
    print(f'    test  : {len(y_te):,}  pos={y_te.sum():,}  '
          f'({y_te.mean()*100:.1f}% positive)')

    probe   = train_probe(X_tr, y_tr, X_val, y_val, device)
    metrics, probs, true, thr = evaluate_probe(probe, X_te, y_te, device)
    metrics['held_out'] = held_out_ckpt
    metrics['variant']  = variant_label

    print(f'    test_auc={metrics["test_auc"]:.4f}  '
          f'test_ap={metrics["test_ap"]:.4f}  '
          f'test_f1={metrics["test_f1"]:.4f}  '
          f'thr={thr:.2f}')

    # save probe + confusion matrix
    torch.save(probe.state_dict(),
               out_dir / f'probe_model_{held_out_ckpt}.pt')
    save_confusion(
        true, probs, thr,
        out_dir / f'confusion_{held_out_ckpt}.png',
        title=f'{variant_label}  |  held_out={held_out_ckpt}  '
              f'thr={thr:.2f}  AUC={metrics["test_auc"]:.4f}  '
              f'F1={metrics["test_f1"]:.4f}'
    )

    with open(out_dir / 'results.json', 'w') as f:
        json.dump([metrics], f, indent=2)

    return metrics


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover(runs_dir, holdout_filter=None, variant_filter=None):
    found = []
    for emb in sorted(runs_dir.glob('*/*/X_all_emb.npy')):
        variant  = emb.parent.name
        held_out = emb.parent.parent.name
        if holdout_filter and held_out not in holdout_filter:
            continue
        if variant_filter  and variant  not in variant_filter:
            continue
        found.append((held_out, variant, emb))
    return found


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MLP probe on student embeddings — matched held-out only.')
    parser.add_argument('--meta',       type=Path, default=DEFAULT_META)
    parser.add_argument('--runs',       type=Path, default=DEFAULT_RUNS)
    parser.add_argument('--holdout',    nargs='+', default=None,
                        help='Which held-out datasets to probe (default: all found)')
    parser.add_argument('--variant',    nargs='+', default=None,
                        help='Which variants to probe (default: all found)')
    parser.add_argument('--also-perch', action='store_true',
                        help='Also probe raw Perch embeddings for the same held-out folds')
    parser.add_argument('--overwrite',  action='store_true',
                        help='Re-run even if results.json already exists')
    args = parser.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    # ── Metadata ──────────────────────────────────────────────────────────────
    print(f'Loading metadata: {args.meta}')
    meta = pd.read_parquet(args.meta)
    meta = meta[meta['coarse_class'].isin(['Tursiops_truncatus', 'background'])].copy()
    meta['label'] = (meta['coarse_class'] == 'Tursiops_truncatus').astype(int)
    meta = meta.reset_index(drop=True)
    print(f'  {len(meta):,} rows  '
          f'(Tursiops: {meta["label"].sum():,}  bg: {(meta["label"]==0).sum():,})')

    # ── Discover student embeddings ───────────────────────────────────────────
    targets = discover(args.runs, args.holdout, args.variant)
    if not targets:
        sys.exit(f'[ERROR] No X_all_emb.npy found under {args.runs}.')

    # ── Optionally add Perch baseline for the same held-out folds ────────────
    if args.also_perch:
        if not TEACHER_EMB_PATH.exists():
            print(f'[WARN] --also-perch: {TEACHER_EMB_PATH} not found — skipping')
        else:
            holdout_set = {ho for ho, _, _ in targets}
            for ho in sorted(holdout_set):
                # store under a synthetic path so output goes to a sensible place
                perch_out = args.runs / ho / 'perch_baseline'
                perch_out.mkdir(parents=True, exist_ok=True)
                targets.append((ho, 'perch_baseline', TEACHER_EMB_PATH))

    print(f'\nWill probe {len(targets)} checkpoint(s):')
    for ho, var, emb in targets:
        print(f'  held_out={ho:28s}  variant={var}')

    # ── Run ───────────────────────────────────────────────────────────────────
    all_results = []
    for held_out, variant, emb_path in targets:
        label   = f'{variant}__{held_out}'
        out_dir = (args.runs / held_out / variant / 'mlp_probe'
                   if variant != 'perch_baseline'
                   else args.runs / held_out / 'perch_baseline' / 'mlp_probe')

        print(f'\n{"="*70}')
        print(f'  held_out : {held_out}')
        print(f'  variant  : {variant}')
        print(f'  emb      : {emb_path}')

        result_file = out_dir / 'results.json'
        if result_file.exists() and not args.overwrite:
            print(f'  [SKIP] results.json exists — use --overwrite to redo')
            with open(result_file) as f:
                all_results.extend(json.load(f))
            continue

        metrics = run_probe_for_checkpoint(
            emb_path, held_out, label, meta, out_dir, device
        )
        all_results.append(metrics)

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_results:
        df           = pd.DataFrame(all_results)
        summary_path = args.runs / 'mlp_probe_summary.csv'
        df.to_csv(summary_path, index=False)

        print(f'\n{"="*70}')
        print(f'Summary -> {summary_path}')
        print(f'\n── test_auc  (held_out × variant) ──')
        for col in ['test_auc', 'test_ap', 'test_f1']:
            try:
                pivot = df.pivot_table(index='held_out', columns='variant',
                                       values=col, aggfunc='first')
                print(f'\n{col}:')
                print(pivot.round(4).to_string())
            except Exception:
                pass

    print('\n[All done]')


if __name__ == '__main__':
    main()