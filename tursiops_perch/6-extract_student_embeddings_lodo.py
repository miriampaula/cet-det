#!/usr/bin/env python3
"""
Extract student embeddings over the full Tursiops corpus for every
available LODO checkpoint, then optionally visualise with PCA / UMAP / t-SNE.

For each fold × variant found under runs_lodo/, this script:
  1. Loads the best checkpoint (best_model.pt)
  2. Runs inference over ALL 65 k rows (train + val + test)
  3. Saves X_all_emb.npy  (65256 × 1536, fp32) next to the checkpoint
  4. Optionally produces PCA / UMAP / t-SNE scatters coloured by species and dataset

Outputs per fold/variant:
    runs_lodo/<held_out>/<variant>/X_all_emb.npy           <- embeddings
    runs_lodo/<held_out>/<variant>/X_all_emb_pca.png       <- PCA scatter  (--viz)
    runs_lodo/<held_out>/<variant>/X_all_emb_umap.png      <- UMAP scatter (--viz, default on)
    runs_lodo/<held_out>/<variant>/X_all_emb_tsne.png      <- t-SNE scatter (--tsne)

Usage:
    cd /data2/mromaniuc/cet-det/tursiops_perch

    # All checkpoints, PCA + UMAP:
    python 6-extract_student_embeddings_lodo.py --viz

    # PCA + UMAP + t-SNE:
    python 6-extract_student_embeddings_lodo.py --viz --tsne

    # Specific folds only:
    python 6-extract_student_embeddings_lodo.py --viz \
        --holdout ALNITAK_CAVANILLES DRYAD OLTREMARE \
        --variant v03_no_grl
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm

# ── Default paths (relative to cet-det/tursiops_perch/) ──────────────────────
DEFAULT_META = Path('student_teacher/meta_train_with_paths.parquet')
DEFAULT_RUNS = Path('student_teacher/runs_lodo')
IMG_SIZE     = 224
BATCH_SIZE   = 256
N_WORKERS    = 6
EMB_DIM      = 1536

# ── Model definition — must match training script exactly ─────────────────────

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class TursiopsStudent(nn.Module):
    def __init__(self, n_domains: int = 1):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0', pretrained=False, num_classes=0
        )
        self.projection = nn.Sequential(
            nn.Linear(1280, EMB_DIM, bias=False),
            nn.LayerNorm(EMB_DIM),
            nn.Dropout(0.3),
        )
        self.class_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(EMB_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(EMB_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_domains),
        )

    def forward(self, x, lam: float = 0.0):
        feat   = self.backbone(x)
        emb    = self.projection(feat)
        logit  = self.class_head(emb)
        rev    = GradientReversalFn.apply(emb, lam)
        domain = self.domain_head(rev)
        return logit, domain, emb


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpectrogramDataset(torch.utils.data.Dataset):
    _tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self._tf(img)


# ── Extraction ────────────────────────────────────────────────────────────────

def extract(model, paths, device, batch_size=BATCH_SIZE, n_workers=N_WORKERS):
    ds     = SpectrogramDataset(paths)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=True,
        persistent_workers=(n_workers > 0),
        prefetch_factor=4 if n_workers > 0 else None,
    )
    model.eval()
    all_emb = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            _, _, emb = model(batch, lam=0.0)
            all_emb.append(emb.cpu().float().numpy())
            if (i + 1) % 50 == 0:
                done = min((i + 1) * batch_size, len(paths))
                print(f'    {done:,}/{len(paths):,}', flush=True)
    return np.concatenate(all_emb, axis=0)


# ── Colour palettes ───────────────────────────────────────────────────────────

SPECIES_COLORS = {
    'background':             '#888888',
    'Tursiops_truncatus':     '#7F77DD',
    'Delphinus_delphis':      '#378ADD',
    'Physeter_macrocephalus': '#D85A30',
    'Orcinus_orca':           '#1D9E75',
    'Delphinids':             '#BA7517',
    'Globicephala_melas':     '#D4537E',
    'Balaenoptera_physalus':  '#5DCAA5',
    'Grampus_griseus':        '#F09595',
    'Stenella_coeruleoalba':  '#639922',
}
DATASET_COLORS = {
    'ALNITAK_CAVANILLES': '#378ADD', 'ECOSS_testtrain': '#D85A30',
    'WATKINS':            '#1D9E75', 'DRYAD':           '#E24B4A',
    'DOLPHINFREE':        '#BA7517', 'ECOSS_annot':     '#D4537E',
    'MONISH':             '#5DCAA5', 'Adriatic_Sea':    '#F09595',
    'OLTREMARE':          '#639922', 'ECOSS_enhanced':  '#888888',
}


def auto_cmap(labels, base):
    extra = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
    out = dict(base); ei = 0
    for l in sorted(set(labels)):
        if l not in out:
            out[l] = extra[ei % len(extra)]; ei += 1
    return out


# ── Visualisation ─────────────────────────────────────────────────────────────

def _scatter_panel(ax, coords, labels, cmap, title, s=2, alpha=0.4):
    import matplotlib.patches as mpatches
    colors = [cmap.get(l, '#cccccc') for l in labels]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=s, alpha=alpha, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    patches = [mpatches.Patch(color=c, label=l) for l, c in cmap.items()]
    ax.legend(handles=patches, fontsize=6, title_fontsize=7,
              loc='upper right', framealpha=0.6)


def save_scatter(coords, meta, out_png, method_name, held_out, variant):
    import matplotlib.pyplot as plt
    sp_col   = 'coarse_class' if 'coarse_class' in meta.columns else 'label'
    species  = meta[sp_col].astype(str).values
    datasets = meta['dataset'].astype(str).values
    sp_cmap  = auto_cmap(species,  SPECIES_COLORS)
    ds_cmap  = auto_cmap(datasets, DATASET_COLORS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#fafafa')
    fig.suptitle(f'{method_name}  —  held_out={held_out}  variant={variant}'
                 f'  ({len(species):,} samples)', fontsize=12)

    _scatter_panel(axes[0], coords, species,  sp_cmap, f'{method_name} — by species')
    _scatter_panel(axes[1], coords, datasets, ds_cmap, f'{method_name} — by dataset')

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {out_png}')


def run_visualisations(X, meta, out_dir, held_out, variant, run_tsne=False):
    from sklearn.decomposition import PCA
    from umap import UMAP

    # ── PCA intermediate (50D) for UMAP / t-SNE input ────────────────────────
    n_pca = min(50, X.shape[1], X.shape[0] - 1)
    print(f'  PCA ({n_pca}D intermediate)...', flush=True)
    X50 = PCA(n_components=n_pca, random_state=42).fit_transform(X)

    # ── PCA 2D ────────────────────────────────────────────────────────────────
    print('  PCA 2D...', flush=True)
    coords_pca = PCA(n_components=2, random_state=42).fit_transform(X)
    np.save(out_dir / 'X_all_emb_pca2.npy', coords_pca)
    save_scatter(coords_pca, meta, out_dir / 'X_all_emb_pca.png',
                 'PCA', held_out, variant)

    # ── UMAP ─────────────────────────────────────────────────────────────────
    print('  UMAP (this takes a few minutes on 65k points)...', flush=True)
    coords_umap = UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1,
        metric='cosine', random_state=42, verbose=True,
        low_memory=False,
    ).fit_transform(X50)
    np.save(out_dir / 'X_all_emb_umap2.npy', coords_umap)
    save_scatter(coords_umap, meta, out_dir / 'X_all_emb_umap.png',
                 'UMAP', held_out, variant)

    # ── t-SNE (optional — slow) ───────────────────────────────────────────────
    if run_tsne:
        from sklearn.manifold import TSNE
        print('  t-SNE (slow — ~10-20 min on 65k)...', flush=True)
        coords_tsne = TSNE(
            n_components=2, perplexity=40, metric='cosine',
            random_state=42, init='pca', verbose=1,
        ).fit_transform(X50)
        np.save(out_dir / 'X_all_emb_tsne2.npy', coords_tsne)
        save_scatter(coords_tsne, meta, out_dir / 'X_all_emb_tsne.png',
                     't-SNE', held_out, variant)


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def discover_checkpoints(runs_dir, holdout_filter=None, variant_filter=None):
    found = []
    for ckpt in sorted(runs_dir.glob('*/*/best_model.pt')):
        variant  = ckpt.parent.name
        held_out = ckpt.parent.parent.name
        if holdout_filter and held_out not in holdout_filter:
            continue
        if variant_filter  and variant  not in variant_filter:
            continue
        found.append((held_out, variant, ckpt))
    return found


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Extract student embeddings + PCA/UMAP/t-SNE.')
    parser.add_argument('--meta',    type=Path, default=DEFAULT_META)
    parser.add_argument('--runs',    type=Path, default=DEFAULT_RUNS)
    parser.add_argument('--holdout', nargs='+', default=None)
    parser.add_argument('--variant', nargs='+', default=None)
    parser.add_argument('--viz',     action='store_true',
                        help='Run PCA + UMAP visualisations after extraction')
    parser.add_argument('--tsne',    action='store_true',
                        help='Also run t-SNE (slow, only if --viz is set)')
    parser.add_argument('--viz-only', action='store_true',
                        help='Skip extraction, only re-run viz on existing X_all_emb.npy')
    parser.add_argument('--batch',   type=int, default=BATCH_SIZE)
    parser.add_argument('--workers', type=int, default=N_WORKERS)
    parser.add_argument('--device',  type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f'Loading metadata: {args.meta}')
    meta  = pd.read_parquet(args.meta)
    paths = meta['png_path'].values
    print(f'  {len(meta):,} rows  |  columns: {list(meta.columns)}')

    checkpoints = discover_checkpoints(args.runs, args.holdout, args.variant)
    if not checkpoints:
        sys.exit(f'[ERROR] No best_model.pt found under {args.runs}.')

    print(f'\nFound {len(checkpoints)} checkpoint(s):')
    for ho, var, ckpt in checkpoints:
        print(f'  {ho:30s}  {var:25s}  {ckpt}')

    for held_out, variant, ckpt_path in checkpoints:
        out_dir = ckpt_path.parent
        out_emb = out_dir / 'X_all_emb.npy'

        print(f'\n{"="*70}')
        print(f'  held_out : {held_out}')
        print(f'  variant  : {variant}')

        # ── Extraction ────────────────────────────────────────────────────────
        if args.viz_only:
            if not out_emb.exists():
                print(f'  [SKIP viz] {out_emb} not found — run without --viz-only first')
                continue
            print(f'  [viz-only] Loading existing {out_emb}')
            embs = np.load(out_emb)
        else:
            if out_emb.exists():
                print(f'  [SKIP extraction] {out_emb} already exists')
                embs = np.load(out_emb)
            else:
                raw   = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                state = raw['model_state'] if isinstance(raw, dict) and 'model_state' in raw else raw
                state = {k.replace('module.', ''): v for k, v in state.items()}

                if isinstance(raw, dict):
                    print(f'  Checkpoint — epoch: {raw.get("epoch")}  '
                          f'val_auc: {raw.get("val_auc"):.6f}  '
                          f'val_f1: {raw.get("val_f1"):.6f}')

                domain_w  = [k for k in state if 'domain_head' in k and k.endswith('.weight')]
                n_domains = state[domain_w[-1]].shape[0] if domain_w else 1

                model = TursiopsStudent(n_domains=n_domains).to(device)
                missing, unexpected = model.load_state_dict(state, strict=False)
                real_missing    = [k for k in missing    if 'dropout' not in k.lower()]
                real_unexpected = [k for k in unexpected if 'dropout' not in k.lower()]
                if real_missing:
                    print(f'  [WARN] Missing    : {real_missing[:5]}')
                if real_unexpected:
                    print(f'  [WARN] Unexpected : {real_unexpected[:5]}')
                if not real_missing and not real_unexpected:
                    print(f'  Model loaded cleanly  (n_domains={n_domains})')

                print(f'  Extracting {len(paths):,} samples '
                      f'(batch={args.batch}, workers={args.workers})...')
                embs = extract(model, paths, device,
                               batch_size=args.batch, n_workers=args.workers)
                print(f'  Done. Shape: {embs.shape}  dtype: {embs.dtype}')

                norms = np.linalg.norm(embs, axis=1)
                print(f'  Norm — mean: {norms.mean():.3f}  std: {norms.std():.3f}'
                      f'  NaN: {np.isnan(embs).any()}  Inf: {np.isinf(embs).any()}')

                np.save(out_emb, embs)
                print(f'  Saved -> {out_emb}')

                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        # ── Visualisation ─────────────────────────────────────────────────────
        if args.viz or args.viz_only:
            run_visualisations(embs, meta, out_dir,
                               held_out, variant, run_tsne=args.tsne)

    print('\n[All done]')


if __name__ == '__main__':
    main()