#!/usr/bin/env python3
"""
Visualise any set of embedding files (teacher or student) with PCA, UMAP, t-SNE.
Coloured by species and by dataset.

Usage:
    # Teacher embeddings:
    python 4-visualise-embeddings.py \
        /data2/mromaniuc/cet-det/tursiops_perch/student_teacher/X_teacher_emb.npy

    # Student embeddings from LODO runs:
    python 4-visualise-embeddings.py \
        runs_lodo/Adriatic_Sea/v01_grl_distil/X_test_emb.npy \
        runs_lodo/Adriatic_Sea/v02_grl_only/X_test_emb.npy

    # Mix of anything:
    python 4-visualise-embeddings.py \
        X_teacher_emb.npy \
        runs_lodo/Adriatic_Sea/v01_grl_distil/X_test_emb.npy \
        --meta /data2/mromaniuc/cet-det/tursiops_perch/student_teacher/meta_train_with_paths.parquet \
        --out ./viz_out \
        --label my_experiment

    # If embeddings are a subset of meta rows (e.g. test-only LODO embeddings),
    # pass the matching metadata file directly or filter with --dataset:
    python visualise_embeddings.py \
        runs_lodo/Adriatic_Sea/v01_grl_distil/X_test_emb.npy \
        --meta meta_train_with_paths.parquet \
        --dataset Adriatic_Sea       # filters meta to matching rows only

Options:
    --meta PATH       Path to .parquet metadata file (must have coarse_class, dataset columns)
    --out  DIR        Output directory for PNGs and saved 2D coords (default: ./viz_out)
    --label STR       Optional prefix added to output filenames
    --dataset DS...   Filter metadata to these dataset(s) before plotting — use when
                      your embeddings are a subset (e.g. LODO test set)
    --no-tsne         Skip t-SNE (faster, useful for quick checks)
    --no-umap         Skip UMAP
    --pca-only        Only run PCA (fastest)
    --s SIZE          Scatter point size (default: 3)
    --alpha FLOAT     Scatter alpha (default: 0.5)
    --dpi INT         Output PNG DPI (default: 150)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_META = Path(
    '/data2/mromaniuc/cet-det/tursiops_perch/student_teacher/meta_train_with_paths.parquet'
)
DEFAULT_OUT  = Path('./viz_out')

# ── Colour palettes ───────────────────────────────────────────────────────────
SPECIES_COLORS = {
    'background':               '#888888',
    'Delphinus_delphis':        '#378ADD',
    'Physeter_macrocephalus':   '#D85A30',
    'Orcinus_orca':             '#1D9E75',
    'Tursiops_truncatus':       '#7F77DD',
    'Delphinids':               '#BA7517',
    'Globicephala_melas':       '#D4537E',
    'Balaenoptera_physalus':    '#5DCAA5',
    'Grampus_griseus':          '#F09595',
    'Stenella_coeruleoalba':    '#639922',
}

DATASET_COLORS = {
    'ALNITAK_CAVANILLES': '#378ADD',
    'ECOSS_testtrain':    '#D85A30',
    'Watkins':            '#1D9E75',
    'WATKINS':            '#1D9E75',
    'DCLDE_2026':         '#7F77DD',
    'DOLPHINFREE':        '#BA7517',
    'ECOSS_annot':        '#D4537E',
    'MONISH':             '#5DCAA5',
    'Adriatic_Sea':       '#F09595',
    'OLTREMARE':          '#639922',
    'DRYAD':              '#E24B4A',
    'ECOSS_enhanced':     '#888888',
}

PCA_INTERMEDIATE_DIMS = 50
UMAP_KWARGS = dict(n_components=2, n_neighbors=30, min_dist=0.1,
                   metric='cosine', random_state=42, verbose=True)
TSNE_KWARGS = dict(n_components=2, perplexity=40, metric='cosine',
                   random_state=42, init='pca', verbose=1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_legend(ax, label_color_map, title, ncol=1):
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in label_color_map.items()
               if l != '__unknown__']
    ax.legend(handles=patches, title=title, fontsize=6,
              title_fontsize=7, ncol=ncol,
              loc='upper right', framealpha=0.6,
              bbox_to_anchor=(1.0, 1.0))


def scatter(ax, coords, labels, color_map, title, s=3, alpha=0.5):
    colors = [color_map.get(l, '#cccccc') for l in labels]
    ax.scatter(coords[:, 0], coords[:, 1],
               c=colors, s=s, alpha=alpha, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def auto_color_map(labels, base_map):
    """Extend base_map with auto-assigned colours for any unseen labels."""
    extra_colors = [
        '#4477AA', '#EE6677', '#228833', '#CCBB44',
        '#66CCEE', '#AA3377', '#BBBBBB', '#EE8866',
    ]
    out = dict(base_map)
    extra_idx = 0
    for l in sorted(set(labels)):
        if l not in out:
            out[l] = extra_colors[extra_idx % len(extra_colors)]
            extra_idx += 1
    return out


def reduce(X, run_umap=True, run_tsne=True):
    """Return dict of 2D projections."""
    results = {}

    # PCA → 50D intermediate
    n_pca_dims = min(PCA_INTERMEDIATE_DIMS, X.shape[1], X.shape[0] - 1)
    print(f"  PCA (→ {n_pca_dims})...")
    pca50 = PCA(n_components=n_pca_dims, random_state=42)
    X50   = pca50.fit_transform(X)
    print(f"    Explained variance (50D): {pca50.explained_variance_ratio_.sum():.3f}")

    # PCA → 2D
    print("  PCA (→ 2)...")
    pca2 = PCA(n_components=2, random_state=42)
    results['PCA'] = pca2.fit_transform(X)

    if run_umap:
        print("  UMAP...")
        results['UMAP'] = UMAP(**UMAP_KWARGS).fit_transform(X50)

    if run_tsne:
        print("  t-SNE...")
        results['t-SNE'] = TSNE(**TSNE_KWARGS).fit_transform(X50)

    return results


def plot_and_save(projections, species, datasets,
                  label, out_dir, s=3, alpha=0.5, dpi=150):
    """
    projections: dict of method_name → (N, 2) array
    """
    sp_cmap  = auto_color_map(species,  SPECIES_COLORS)
    ds_cmap  = auto_color_map(datasets, DATASET_COLORS)

    n_methods = len(projections)
    fig, axes = plt.subplots(2, n_methods, figsize=(6 * n_methods, 11))
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    fig.patch.set_facecolor('#fafafa')

    for col, (method_name, coords) in enumerate(projections.items()):
        scatter(axes[0, col], coords, species,
                sp_cmap, f'{method_name} — by species', s=s, alpha=alpha)
        make_legend(axes[0, col], sp_cmap, 'species')

        scatter(axes[1, col], coords, datasets,
                ds_cmap, f'{method_name} — by dataset', s=s, alpha=alpha)
        make_legend(axes[1, col], ds_cmap, 'dataset')

    fig.suptitle(f'{label}  ({len(species):,} samples)', fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = out_dir / f'{label}_pca_umap_tsne.png'
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")

    # Save 2D coords
    for method_name, coords in projections.items():
        key = method_name.lower().replace('-', '')
        np.save(out_dir / f'{label}_X_{key}2.npy', coords)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Visualise teacher or student embeddings with PCA/UMAP/t-SNE.')
    parser.add_argument(
        'embeddings', nargs='+', type=Path,
        help='One or more .npy embedding files. Each gets its own output figure.')
    parser.add_argument('--meta',    type=Path, default=DEFAULT_META)
    parser.add_argument('--out',     type=Path, default=DEFAULT_OUT)
    parser.add_argument('--label',   type=str,  default=None,
                        help='Output filename prefix (default: stem of .npy file)')
    parser.add_argument('--dataset', nargs='+', default=None,
                        help='Filter metadata to these dataset(s) — use for LODO test embeddings')
    parser.add_argument('--no-tsne', action='store_true')
    parser.add_argument('--no-umap', action='store_true')
    parser.add_argument('--pca-only', action='store_true')
    parser.add_argument('--s',     type=float, default=3)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--dpi',   type=int,   default=150)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    run_umap = not (args.no_umap or args.pca_only)
    run_tsne = not (args.no_tsne or args.pca_only)

    # Load metadata
    print(f"Loading metadata from {args.meta} ...")
    meta = pd.read_parquet(args.meta)
    if args.dataset:
        meta = meta[meta['dataset'].isin(args.dataset)].reset_index(drop=True)
        print(f"  Filtered to {args.dataset} → {len(meta):,} rows")
    print(f"  Metadata rows: {len(meta):,}")

    species  = meta['coarse_class'].astype(str).values
    datasets = meta['dataset'].astype(str).values

    # Process each embedding file
    for emb_path in args.embeddings:
        emb_path = emb_path.resolve()
        if not emb_path.exists():
            print(f"\n[SKIP] {emb_path} not found")
            continue

        label = args.label if args.label else emb_path.stem

        # If multiple files and no explicit label, disambiguate by parent dir
        if len(args.embeddings) > 1 and args.label is None:
            label = f"{emb_path.parent.name}__{emb_path.stem}"

        print(f"\n{'='*60}")
        print(f"  File  : {emb_path}")
        print(f"  Label : {label}")

        X = np.load(emb_path)
        print(f"  Shape : {X.shape}  dtype: {X.dtype}")

        if len(X) != len(meta):
            print(f"  ⚠️  Shape mismatch: {len(X)} embeddings vs {len(meta)} "
                  f"metadata rows — did you forget --dataset?")
            continue

        projections = reduce(X, run_umap=run_umap, run_tsne=run_tsne)
        plot_and_save(projections, species, datasets,
                      label=label, out_dir=args.out,
                      s=args.s, alpha=args.alpha, dpi=args.dpi)

    print("\nAll done.")


if __name__ == '__main__':
    main()