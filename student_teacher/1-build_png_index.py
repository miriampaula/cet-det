# build_png_index.py — v3 clean
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

META_PATH = Path('/data2/mromaniuc/cet-det/cet_perchv2/meta_train.parquet')
SPEC_ROOT = Path('/data2/mromaniuc/cet-det/models/perch_v2')
OUT_PATH  = Path('/data2/mromaniuc/cet-det/student_teacher/meta_train_with_paths.parquet')

DATASET_SPEC_DIRS = {
    'Adriatic_Sea':    'ADRIATIC_SEA/spectrograms',
    'ALNITAK_CAVANILLES': 'ALNITAK_CAVANILLES/spectrograms',
    'DCLDE_2026':      'DCLDE_2026/spectrograms',
    'DOLPHINFREE':     'DOLPHINFREE/spectrograms',
    'DRYAD':           'DRYAD/spectrograms',
    'ECOSS_annot':     'ECOSS/annotated_sounds/spectrograms',
    'ECOSS_enhanced':  'ECOSS/enhanced4AI_sounds/spectrograms',
    'ECOSS_testtrain': 'ECOSS/testingtraining_sounds/spectrograms',
    'FREMANTLE':       'FREMANTLE/spectrograms',
    'MONISH':          'MONISH/spectrograms',
    'OLTREMARE':       'OLTREMARE/spectrograms',
    'Watkins':         'WATKINS/spectrograms',
}

OFFSET_BASED = {
    'Adriatic_Sea', 'ALNITAK_CAVANILLES', 'DCLDE_2026', 'DOLPHINFREE',
    'DRYAD', 'ECOSS_annot', 'ECOSS_enhanced', 'ECOSS_testtrain', 'OLTREMARE',
}
WINDOW_BASED = {'Watkins', 'MONISH', 'FREMANTLE'}

# offset pattern: exactly 8 digits, dot, 1 digit, 's'
OFFSET_PAT = re.compile(r'_(\d{8}\.\ds)_')
WIDX_PAT   = re.compile(r'_(\d{4})_')


def build_indices(datasets):
    """
    For each dataset, scan its spectrograms dir once and build two indices:
      offset_idx[dataset][offset_token] = [(before_str, filepath), ...]
      window_idx[dataset][widx_str]     = [(before_str, filepath), ...]
    """
    offset_idx = {}
    window_idx = {}

    for ds in datasets:
        if ds not in DATASET_SPEC_DIRS:
            logging.warning(f"No dir mapping for dataset: {ds}")
            continue
        spec_dir = SPEC_ROOT / DATASET_SPEC_DIRS[ds]
        if not spec_dir.exists():
            logging.warning(f"Directory missing: {spec_dir}")
            continue

        files = list(spec_dir.iterdir())
        logging.info(f"  {ds:25s} → {spec_dir.name}  ({len(files)} files)")

        oi = defaultdict(list)
        wi = defaultdict(list)

        for f in files:
            stem = f.stem
            # Try offset pattern first
            m = OFFSET_PAT.search(stem)
            if m:
                offset_token = m.group(1)
                before = stem[:m.start()]
                oi[offset_token].append((before, str(f)))
                continue
            # Try window index pattern
            m = WIDX_PAT.search(stem)
            if m:
                widx = str(int(m.group(1)))
                before = stem[:m.start()]
                wi[widx].append((before, str(f)))

        offset_idx[ds] = oi
        window_idx[ds] = wi

    return offset_idx, window_idx


def resolve(row, offset_idx, window_idx):
    ds = row['dataset']

    if ds in OFFSET_BASED:
        if pd.isna(row.get('offset_s')) or row.get('wav_path') is None:
            return None
        wav_stem = Path(row['wav_path']).stem
        offset_token = f"{float(row['offset_s']):010.1f}s"
        oi = offset_idx.get(ds, {})
        candidates = oi.get(offset_token, [])
        # match: wav_stem must appear in the 'before' part of the filename
        matches = [fp for before, fp in candidates if wav_stem in before]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            label = str(row.get('label', ''))
            narrowed = [fp for fp in matches if Path(fp).stem.endswith(f'_{label}')]
            return narrowed[0] if narrowed else matches[0]
        return None

    elif ds in WINDOW_BASED:
        if pd.isna(row.get('window_idx')):
            return None
        source = row.get('source_file') or row.get('wav_path')
        if not source:
            return None
        wav_stem = Path(source).stem
        widx = str(int(row['window_idx']))
        wi = window_idx.get(ds, {})
        candidates = wi.get(widx, [])
        matches = [fp for before, fp in candidates if wav_stem in before]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return matches[0]
        return None

    return None


def main():
    logging.info(f"Loading {META_PATH}")
    meta = pd.read_parquet(META_PATH)
    logging.info(f"  {len(meta)} rows, datasets: {sorted(meta['dataset'].unique())}")

    logging.info("Scanning spectrogram directories...")
    datasets = meta['dataset'].unique().tolist()
    offset_idx, window_idx = build_indices(datasets)

    logging.info("Matching rows to PNGs...")
    png_paths = []
    by_dataset_missing = defaultdict(int)

    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        path = resolve(row, offset_idx, window_idx)
        png_paths.append(path)
        if path is None:
            by_dataset_missing[row['dataset']] += 1

    meta['png_path'] = png_paths
    n_missing = sum(1 for p in png_paths if p is None)

    logging.info(f"\nResolved: {len(meta) - n_missing}/{len(meta)}")
    if by_dataset_missing:
        logging.info("Missing by dataset:")
        for ds, n in sorted(by_dataset_missing.items(), key=lambda x: -x[1]):
            total = (meta['dataset'] == ds).sum()
            logging.info(f"  {ds:25s} {n}/{total} missing")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(OUT_PATH, index=False)
    logging.info(f"Saved → {OUT_PATH}")


if __name__ == '__main__':
    main()
