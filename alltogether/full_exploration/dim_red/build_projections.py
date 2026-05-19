"""
build_projections.py
====================
One-shot script: indexes spectrograms, computes PCA/UMAP/t-SNE projections
over the full Perch-v2 embedding corpus, and writes a single parquet that
the Dash viewer (app.py) consumes.

Run once. Re-run only if datasets, labels, or projection params change.

Expected runtime on ~300k rows, 1280-d embeddings:
    - PCA-50, PCA-2/3d:   ~30 s
    - UMAP-2d + 3d:       ~10-15 min (CPU) / ~1-2 min (cuML GPU)
    - t-SNE 50k subsample:~5-10 min (openTSNE multicore)
    - Spectrogram index:  ~1-2 min per dataset (one disk walk each)
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# -------------------------------------------------------------------------
# Configuration — keep in sync with the notebook
# -------------------------------------------------------------------------
DATASETS = {
    'Adriatic_Sea':         '/data2/mromaniuc/cet-det/models/perch_v2/ADRIATIC_SEA',
    'ALNITAK_CAVANILLES':   '/data2/mromaniuc/cet-det/models/perch_v2/ALNITAK_CAVANILLES',
    'DCLDE_2026':           '/data2/mromaniuc/cet-det/models/perch_v2/DCLDE_2026',
    'DOLPHINFREE':          '/data2/mromaniuc/cet-det/models/perch_v2/DOLPHINFREE',
    'DRYAD':                '/data2/mromaniuc/cet-det/models/perch_v2/DRYAD',
    'ECOSS_annot':          '/data2/mromaniuc/cet-det/models/perch_v2/ECOSS/annotated_sounds',
    'ECOSS_enhanced':       '/data2/mromaniuc/cet-det/models/perch_v2/ECOSS/enhanced4AI_sounds',
    'ECOSS_testtrain':      '/data2/mromaniuc/cet-det/models/perch_v2/ECOSS/testingtraining_sounds',
    'FREMANTLE':            '/data2/mromaniuc/cet-det/models/perch_v2/FREMANTLE',
    'OLTREMARE':            '/data2/mromaniuc/cet-det/models/perch_v2/OLTREMARE',
    'MONISH':               '/data2/mromaniuc/cet-det/models/perch_v2/MONISH',
    'WATKINS':              '/data2/mromaniuc/cet-det/models/perch_v2/WATKINS',
}

OUT_PARQUET     = Path('./projector_data.parquet')
TSNE_SUBSAMPLE  = 50_000
PCA_INTERMEDIATE_DIM = 50
RANDOM_SEED     = 42

# -------------------------------------------------------------------------
# Spectrogram pairing — index-based
# -------------------------------------------------------------------------
# We rely on the invariant that embeddings and spectrograms are emitted in
# the same order per dataset (verified against row counts).  No filename
# parsing required — `sorted(listdir)[i]` pairs with embedding row `i`.
#
# Special case: DOLPHINFREE has spectrograms only for `whistle`-labeled rows
# (background was excluded as not clean). 1,150 PNGs ↔ 1,150 whistle rows,
# in the metadata order those whistle rows appear; background rows get None.

DATASET_PNG_FILTER = {
    # name → predicate(meta_subset) that selects rows expected to have a PNG,
    #        in the same order the PNGs were emitted.
    # Default (no entry) = all rows.
    'DOLPHINFREE': lambda m: m['label'].astype(str) == 'whistle',
}


def attach_png_paths(meta_all: pd.DataFrame) -> pd.DataFrame:
    """Pair embeddings to PNGs by sorted-listdir index within each dataset."""
    print("\n[1/5] Indexing spectrograms (index-based pairing)")
    png_paths = pd.Series([None] * len(meta_all), index=meta_all.index, dtype=object)

    for name, root in DATASETS.items():
        spec_dir = Path(root) / 'spectrograms'
        if not spec_dir.exists():
            print(f"  {name:<22s} no spectrograms/ dir, skipping")
            continue

        # Sort PNGs the same way `ls` does — Python's default sort is locale-free
        # and stable, matching what was used to generate them.
        pngs = sorted(
            (entry.path for entry in os.scandir(spec_dir)
             if entry.name.endswith('.png'))
        )
        n_pngs = len(pngs)

        mask = (meta_all['dataset'] == name).to_numpy()
        sub_idx = np.where(mask)[0]
        n_rows = len(sub_idx)

        # Apply the optional row-filter (e.g. DOLPHINFREE whistle-only)
        if name in DATASET_PNG_FILTER:
            sub = meta_all.iloc[sub_idx]
            keep = DATASET_PNG_FILTER[name](sub).to_numpy()
            target_idx = sub_idx[keep]
            n_target = len(target_idx)
            print(f"  {name:<22s} filter keeps {n_target:,} / {n_rows:,} rows "
                  f"for pairing against {n_pngs:,} PNGs")
        else:
            target_idx = sub_idx
            n_target = n_rows

        if n_target != n_pngs:
            print(f"  {name:<22s} MISMATCH: {n_target:,} target rows vs {n_pngs:,} PNGs — "
                  f"leaving png_path = None for this dataset")
            continue

        # Index-based pairing
        for row_i, png_path in zip(target_idx, pngs):
            png_paths.iloc[row_i] = png_path
        print(f"  {name:<22s} paired {n_target:,} rows ↔ {n_pngs:,} PNGs  ✓")

    meta_all = meta_all.copy()
    meta_all['png_path'] = png_paths.values
    total_hit = meta_all['png_path'].notna().sum()
    print(f"  TOTAL: {total_hit:,} / {len(meta_all):,} rows have a spectrogram "
          f"({100*total_hit/len(meta_all):.1f}%)")
    return meta_all


# -------------------------------------------------------------------------
# Projections
# -------------------------------------------------------------------------
def fit_pca(X: np.ndarray, n_components: int, seed: int = RANDOM_SEED):
    """PCA via sklearn. Returns (Y, variance_ratios_array_of_length_n_components)."""
    from sklearn.decomposition import PCA
    t0 = time.time()
    print(f"  PCA → {n_components}d on X={X.shape} ...", end=' ', flush=True)
    pca = PCA(n_components=n_components, random_state=seed)
    Y = pca.fit_transform(X).astype(np.float32)
    var = pca.explained_variance_ratio_.astype(np.float64)
    print(f"done in {time.time()-t0:.1f}s  (cumulative explained var: {var.sum():.3f})")
    return Y, var


def fit_umap(X: np.ndarray, n_components: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """UMAP via cuML if available, else umap-learn.

    Note: umap-learn silently disables parallelism when random_state is set.
    We pass it to cuML (which honors it without losing parallelism), but omit
    it from the CPU fallback to keep all cores busy. The resulting layout is
    deterministic in cluster structure but may be rotated/mirrored between runs.
    """
    t0 = time.time()
    try:
        from cuml.manifold import UMAP as cuUMAP  # type: ignore
        print(f"  UMAP-{n_components}d via cuML on X={X.shape} ...", end=' ', flush=True)
        Y = cuUMAP(n_components=n_components, n_neighbors=30,
                   min_dist=0.1, random_state=seed).fit_transform(X)
        Y = np.asarray(Y, dtype=np.float32)
    except Exception as e:
        print(f"\n  (cuML UMAP unavailable: {type(e).__name__}; falling back to umap-learn)")
        import umap
        print(f"  UMAP-{n_components}d via umap-learn on X={X.shape} ...", end=' ', flush=True)
        Y = umap.UMAP(n_components=n_components, n_neighbors=30,
                      min_dist=0.1,
                      # random_state intentionally omitted: setting it forces
                      # n_jobs=1 in umap-learn, which would be ~5-10x slower.
                      low_memory=True, verbose=True).fit_transform(X).astype(np.float32)
    print(f"done in {time.time()-t0:.1f}s")
    return Y


def fit_tsne(X: np.ndarray, n_components: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """t-SNE via openTSNE (multicore CPU).

    Note on negative_gradient_method:
        - 'fft' (default) is fast but ONLY supports n_components=2
        - 'bh' (Barnes-Hut) supports any output dim, slower but works
    """
    from openTSNE import TSNE
    t0 = time.time()
    ngm = 'fft' if n_components == 2 else 'bh'
    print(f"  t-SNE-{n_components}d via openTSNE ({ngm}) on X={X.shape} ...", end=' ', flush=True)
    Y = TSNE(
        n_components=n_components,
        perplexity=30,
        n_jobs=-1,
        random_state=seed,
        negative_gradient_method=ngm,
        verbose=False,
    ).fit(X)
    Y = np.asarray(Y, dtype=np.float32)
    print(f"done in {time.time()-t0:.1f}s")
    return Y


def stratified_indices(labels: np.ndarray, k: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """Pick k indices balanced across `labels`."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'lab': labels, 'idx': np.arange(len(labels))})
    n_classes = df['lab'].nunique()
    per_class = max(1, k // n_classes)
    picked = (
        df.groupby('lab', group_keys=False)
          .apply(lambda g: g.sample(n=min(per_class, len(g)), random_state=rng.integers(1<<31)),
                 include_groups=False)
    )
    if len(picked) > k:
        picked = picked.sample(n=k, random_state=seed)
    return np.sort(picked['idx'].to_numpy())


def cached(cache_dir: Path | None, name: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs); cache result to cache_dir/<name>.npy.

    If the cache file exists, load it and skip the computation.
    If cache_dir is None, no caching — always run fresh.
    """
    if cache_dir is None:
        return fn(*args, **kwargs)
    cache_path = cache_dir / f'{name}.npy'
    if cache_path.exists():
        print(f"  [cache hit] {cache_path.name}", flush=True)
        return np.load(cache_path)
    out = fn(*args, **kwargs)
    np.save(cache_path, out)
    print(f"  [cached] {cache_path.name}", flush=True)
    return out


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-pkl', required=True,
                        help='Path to a pickled (meta_all, X_all) tuple, OR a directory '
                             'containing meta_all.parquet and X_all.npy')
    parser.add_argument('--out', default=str(OUT_PARQUET), help='Output parquet path')
    parser.add_argument('--cache-dir', default=None,
                        help='Directory to cache intermediate PCA/UMAP/t-SNE arrays '
                             '(skips re-computation on re-run; useful when one step crashes)')
    parser.add_argument('--skip-tsne', action='store_true')
    parser.add_argument('--tsne-subsample', type=int, default=TSNE_SUBSAMPLE)
    args = parser.parse_args()

    cache_dir = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)
        print(f"  caching intermediates to {cache_dir}")

    # ---- Load meta_all + X_all (you produce these in your notebook) ----
    print(f"[0/5] Loading meta_all + X_all from {args.meta_pkl}")
    p = Path(args.meta_pkl)
    if p.is_dir():
        meta_all = pd.read_parquet(p / 'meta_all.parquet')
        X_all = np.load(p / 'X_all.npy', mmap_mode='r')
    else:
        import pickle
        with open(p, 'rb') as f:
            meta_all, X_all = pickle.load(f)
    X_all = np.ascontiguousarray(X_all, dtype=np.float32)
    print(f"  meta_all: {len(meta_all):,} rows, {len(meta_all.columns)} cols")
    print(f"  X_all:    {X_all.shape}, {X_all.nbytes/1e9:.2f} GB")
    assert len(meta_all) == len(X_all)

    # ---- 1. Index spectrograms ----
    meta_all = attach_png_paths(meta_all)

    # ---- 2. PCA-50 (intermediate) + PCA-2d + PCA-3d ----
    print("\n[2/5] PCA")
    # fit_pca returns (Y, variance_ratio). We cache them separately so the
    # cache wrapper stays simple (single ndarray per name).
    if cache_dir and (cache_dir / 'pca50.npy').exists() and (cache_dir / 'pca50_var.npy').exists():
        print(f"  [cache hit] pca50.npy + pca50_var.npy")
        X_pca50 = np.load(cache_dir / 'pca50.npy')
        pca_var = np.load(cache_dir / 'pca50_var.npy')
    else:
        X_pca50, pca_var = fit_pca(X_all, PCA_INTERMEDIATE_DIM)
        if cache_dir:
            np.save(cache_dir / 'pca50.npy', X_pca50)
            np.save(cache_dir / 'pca50_var.npy', pca_var)
            print(f"  [cached] pca50.npy + pca50_var.npy")
    pca_2d  = X_pca50[:, :2].copy()
    pca_3d  = X_pca50[:, :3].copy()
    print(f"  variance: PC1={pca_var[0]:.3f}  PC2={pca_var[1]:.3f}  PC3={pca_var[2]:.3f}  "
          f"PC1-3 sum={pca_var[:3].sum():.3f}  PC1-50 sum={pca_var.sum():.3f}")

    # ---- 3. UMAP-2d + UMAP-3d on PCA-50 ----
    print("\n[3/5] UMAP")
    umap_2d = cached(cache_dir, 'umap_2d', fit_umap, X_pca50, 2)
    umap_3d = cached(cache_dir, 'umap_3d', fit_umap, X_pca50, 3)

    # ---- 4. t-SNE-2d on stratified subsample ----
    # t-SNE-3d intentionally skipped: openTSNE's FFT method is 2d-only, and
    # Barnes-Hut 3d is both slower and qualitatively worse than UMAP-3d.
    # We still allocate tsne_3d_* columns (as NaN) so app.py's schema is stable.
    tsne_2d = np.full((len(meta_all), 2), np.nan, dtype=np.float32)
    tsne_3d = np.full((len(meta_all), 3), np.nan, dtype=np.float32)
    if not args.skip_tsne:
        print(f"\n[4/5] t-SNE-2d on {args.tsne_subsample:,} subsample")
        # The full-corpus-shaped tsne_2d array is what we cache (NaN outside subsample).
        def _compute_tsne_2d():
            strat_col = 'label_t2' if 'label_t2' in meta_all.columns else 'dataset'
            labels = meta_all[strat_col].fillna('__none__').to_numpy()
            idx = stratified_indices(labels, args.tsne_subsample)
            print(f"  picked {len(idx):,} indices (stratified on {strat_col})")
            Xs = X_pca50[idx]
            out = np.full((len(meta_all), 2), np.nan, dtype=np.float32)
            out[idx] = fit_tsne(Xs, n_components=2)
            return out
        tsne_2d = cached(cache_dir, 'tsne_2d', _compute_tsne_2d)
        print("  t-SNE-3d skipped (FFT 2d-only; BH 3d is worse than UMAP-3d)")
    else:
        print("\n[4/5] t-SNE SKIPPED")

    # ---- 5. Assemble and save ----
    print("\n[5/5] Writing parquet")
    proj = pd.DataFrame({
        'pca_2d_x':  pca_2d[:, 0].astype(np.float16),
        'pca_2d_y':  pca_2d[:, 1].astype(np.float16),
        'pca_3d_x':  pca_3d[:, 0].astype(np.float16),
        'pca_3d_y':  pca_3d[:, 1].astype(np.float16),
        'pca_3d_z':  pca_3d[:, 2].astype(np.float16),
        'umap_2d_x': umap_2d[:, 0].astype(np.float16),
        'umap_2d_y': umap_2d[:, 1].astype(np.float16),
        'umap_3d_x': umap_3d[:, 0].astype(np.float16),
        'umap_3d_y': umap_3d[:, 1].astype(np.float16),
        'umap_3d_z': umap_3d[:, 2].astype(np.float16),
        'tsne_2d_x': tsne_2d[:, 0].astype(np.float16),
        'tsne_2d_y': tsne_2d[:, 1].astype(np.float16),
        'tsne_3d_x': tsne_3d[:, 0].astype(np.float16),
        'tsne_3d_y': tsne_3d[:, 1].astype(np.float16),
        'tsne_3d_z': tsne_3d[:, 2].astype(np.float16),
    }, index=meta_all.index)

    out = pd.concat([meta_all.reset_index(drop=True), proj.reset_index(drop=True)], axis=1)

    # Embed PCA variance ratios in parquet metadata so the app can show them.
    import pyarrow as pa
    import pyarrow.parquet as pq
    import json
    table = pa.Table.from_pandas(out)
    existing_meta = table.schema.metadata or {}
    new_meta = {
        **existing_meta,
        b'pca_variance_ratio_50d': json.dumps(pca_var.tolist()).encode('utf-8'),
    }
    table = table.replace_schema_metadata(new_meta)
    pq.write_table(table, args.out)
    print(f"  wrote {args.out}  ({Path(args.out).stat().st_size/1e6:.1f} MB)")
    print(f"  embedded PCA variance ratios (50 values) in parquet metadata")
    print(f"\nNext: python app.py --parquet {args.out}")


if __name__ == '__main__':
    main()
