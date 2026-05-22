# visualise_teacher_embeddings.py
"""
PCA, UMAP and t-SNE of embeddings from multiple directories,
coloured by species and by dataset.

Usage:
    python visualise_teacher_embeddings.py \
        /path/to/run_a /path/to/run_b /path/to/run_c

Each directory must contain:
    - X_student_emb.npy   (embeddings)

Metadata is shared and loaded from META_PATH.
One 6-panel PNG is saved per directory, named after the folder.
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

# ── Shared metadata path ──────────────────────────────────────────────────────
META_PATH = Path('/data2/mromaniuc/cet-det/cet_perchv2/meta_train.parquet')
OUT_DIR   = Path('/data2/mromaniuc/cet-det/student_teacher/visualisations')

# ── Embedding filename to look for inside each directory ─────────────────────
EMB_FILENAME = 'X_student_emb.npy'

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
    'DCLDE_2026':         '#7F77DD',
    'DOLPHINFREE':        '#BA7517',
    'ECOSS_annot':        '#D4537E',
    'MONISH':             '#5DCAA5',
    'Adriatic_Sea':       '#F09595',
    'OLTREMARE':          '#639922',
    'DRYAD':              '#E24B4A',
    'ECOSS_enhanced':     '#888888',
}

# ── Dimensionality reduction config (shared across all runs) ──────────────────
PCA_INTERMEDIATE_DIMS = 50
UMAP_KWARGS  = dict(n_components=2, n_neighbors=30, min_dist=0.1,
                    metric='cosine', random_state=42, verbose=True)
TSNE_KWARGS  = dict(n_components=2, perplexity=40, metric='cosine',
                    random_state=42, init='pca', verbose=1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_legend(ax, label_color_map, title, ncol=1):
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in label_color_map.items()]
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


def reduce_and_plot(X, species, datasets, run_name, out_dir):
    """Run PCA/UMAP/t-SNE and save a 6-panel figure for one embedding matrix."""
    n = len(X)
    print(f"\n{'='*60}")
    print(f"  Run: {run_name}  ({n} samples, dim={X.shape[1]})")
    print(f"{'='*60}")

    # PCA 50D (intermediate for UMAP / t-SNE)
    print("  PCA (→ 50)...")
    pca50 = PCA(n_components=PCA_INTERMEDIATE_DIMS, random_state=42)
    X50   = pca50.fit_transform(X)
    print(f"    Explained variance: {pca50.explained_variance_ratio_.sum():.3f}")

    # PCA 2D
    print("  PCA (→ 2)...")
    pca2  = PCA(n_components=2, random_state=42)
    X_pca = pca2.fit_transform(X)

    # UMAP
    print("  UMAP...")
    X_umap = UMAP(**UMAP_KWARGS).fit_transform(X50)

    # t-SNE
    print("  t-SNE...")
    X_tsne = TSNE(**TSNE_KWARGS).fit_transform(X50)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('#fafafa')

    methods = [('PCA', X_pca), ('UMAP', X_umap), ('t-SNE', X_tsne)]

    for col, (method_name, coords) in enumerate(methods):
        scatter(axes[0, col], coords, species,
                SPECIES_COLORS, f'{method_name} — by species')
        make_legend(axes[0, col], SPECIES_COLORS, 'species')

        scatter(axes[1, col], coords, datasets,
                DATASET_COLORS, f'{method_name} — by dataset')
        make_legend(axes[1, col], DATASET_COLORS, 'dataset')

    fig.suptitle(f'{run_name}  —  {n} samples',
                 fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = out_dir / f'{run_name}_pca_umap_tsne.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")

    # Save 2D coords alongside the PNG
    np.save(out_dir / f'{run_name}_X_pca2.npy',  X_pca)
    np.save(out_dir / f'{run_name}_X_umap2.npy', X_umap)
    np.save(out_dir / f'{run_name}_X_tsne2.npy', X_tsne)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Visualise embeddings from one or more run directories.')
    parser.add_argument(
        'directories', nargs='+', type=Path,
        help=f'Directories each containing a "{EMB_FILENAME}" file.')
    parser.add_argument(
        '--meta', type=Path, default=META_PATH,
        help='Path to meta_train.parquet (default: %(default)s)')
    parser.add_argument(
        '--out', type=Path, default=OUT_DIR,
        help='Output directory for PNGs (default: %(default)s)')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Load shared metadata once
    print(f"Loading metadata from {args.meta} ...")
    meta = pd.read_parquet(args.meta)
    species  = meta['coarse_class'].astype(str).values
    datasets = meta['dataset'].astype(str).values
    print(f"  Metadata rows: {len(meta)}")

    # Process each directory
    for run_dir in args.directories:
        run_dir = run_dir.resolve()
        emb_path = run_dir / EMB_FILENAME

        if not emb_path.exists():
            print(f"\n[SKIP] {emb_path} not found — skipping {run_dir.name}")
            continue

        X = np.load(emb_path)
        assert len(X) == len(meta), \
            f"Shape mismatch in {run_dir.name}: {len(X)} embeddings vs {len(meta)} metadata rows"

        reduce_and_plot(X, species, datasets,
                        run_name=run_dir.name,
                        out_dir=args.out)

    print("\nAll done.")


if __name__ == '__main__':
    main()