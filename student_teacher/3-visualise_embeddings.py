# visualise_teacher_embeddings.py
"""
PCA, UMAP and t-SNE of Perch V2 teacher embeddings,
coloured by species and by dataset.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from umap import UMAP
from sklearn.manifold import TSNE

# ── Paths ────────────────────────────────────────────────────────────────────
EMB_PATH  = Path('/data2/mromaniuc/cet-det/student_teacher/runs_student/X_student_emb.npy')
META_PATH = Path('/data2/mromaniuc/cet-det/cet_perchv2/meta_train.parquet')
OUT_DIR   = Path('/data2/mromaniuc/cet-det/student_teacher/visualisations')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading embeddings and metadata...")
X   = np.load(EMB_PATH)
meta = pd.read_parquet(META_PATH)
assert len(X) == len(meta), f"Shape mismatch: {len(X)} embeddings vs {len(meta)} rows"
print(f"  Embeddings: {X.shape}")

species  = meta['coarse_class'].astype(str).values
datasets = meta['dataset'].astype(str).values

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

def make_legend(ax, label_color_map, title, ncol=1):
    patches = [
        mpatches.Patch(color=c, label=l)
        for l, c in label_color_map.items()
    ]
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


# ── Step 1: PCA to 50 dims (speeds up UMAP/t-SNE) ───────────────────────────
print("PCA (1536 → 50)...")
pca50 = PCA(n_components=50, random_state=42)
X_pca50 = pca50.fit_transform(X)
print(f"  Explained variance (50 PCs): {pca50.explained_variance_ratio_.sum():.3f}")

# PCA 2D for visualisation
print("PCA (1536 → 2)...")
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X)

# ── Step 2: UMAP ─────────────────────────────────────────────────────────────
print("UMAP...")
umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
            metric='cosine', random_state=42, verbose=True)
X_umap = umap.fit_transform(X_pca50)

# ── Step 3: t-SNE ─────────────────────────────────────────────────────────────
print("t-SNE...")
tsne = TSNE(n_components=2, perplexity=40,
            metric='cosine', random_state=42,
            init='pca', verbose=1)
X_tsne = tsne.fit_transform(X_pca50)

# ── Plot: 3 methods × 2 colourmodes = 6 panels ───────────────────────────────
print("Plotting...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#fafafa')

methods = [
    ('PCA',   X_pca2),
    ('UMAP',  X_umap),
    ('t-SNE', X_tsne),
]

for col, (method_name, coords) in enumerate(methods):
    # Top row: colour by species
    scatter(axes[0, col], coords, species,
            SPECIES_COLORS,
            f'{method_name} — by species')
    make_legend(axes[0, col], SPECIES_COLORS, 'species', ncol=1)

    # Bottom row: colour by dataset
    scatter(axes[1, col], coords, datasets,
            DATASET_COLORS,
            f'{method_name} — by dataset')
    make_legend(axes[1, col], DATASET_COLORS, 'dataset', ncol=1)

plt.suptitle('Perch V2 teacher embeddings — 11,769 samples',
             fontsize=13, y=1.01)
plt.tight_layout()

out_path = OUT_DIR / 'student_embeddings_pca_umap_tsne.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved → {out_path}")

# ── Also save the 2D coords for later reuse ──────────────────────────────────
np.save(OUT_DIR / 'X_pca2_student.npy',  X_pca2)
np.save(OUT_DIR / 'X_umap2_student.npy', X_umap)
np.save(OUT_DIR / 'X_tsne2_student.npy', X_tsne)
print("2D coordinates saved for reuse.")