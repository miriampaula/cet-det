"""
extract_spectrograms.py
-----------------------
Reads one or more metadata CSV files (same format produced by the embedding
pipeline) and computes the exact Perch V2 log-mel spectrogram for every row,
saving each as a .png image.

Output filename convention:
    {dataset}_{provider}_{subdataset}_{ecotype}_{wav_stem}_{offset_s:010.1f}s_{label}.png

    Fields that are absent or empty in the CSV are skipped (no double underscores).

    e.g.  DCLDE_2026_DFO_WDLP_CarmanahPt_SRKW_ST6249.220308174941_0000000000.0s_background.png
          OLTREMARE_20211120_102109_192_0000000030.0s_echolocation.png

Usage
-----
    # Single CSV
    python extract_spectrograms.py \\
        --csv /data2/.../ALNITAK_CAVANILLES/metadata.csv \\
        --out_dir /data2/.../ALNITAK_CAVANILLES/spectrograms \\
        --dataset ALNITAK_CAVANILLES

    # Multiple CSVs at once
    python extract_spectrograms.py \\
        --csv /data2/.../ADRIATIC_SEA/metadata.csv \\
               /data2/.../DOLPHINFREE/metadata.csv \\
        --out_dir /data2/.../spectrograms \\
        --dataset ADRIATIC_SEA DOLPHINFREE

    # Auto-infer dataset name from parent directory of CSV
    python extract_spectrograms.py \\
        --csv /data2/.../ADRIATIC_SEA/metadata.csv \\
        --out_dir /data2/.../spectrograms

Notes
-----
- Spectrograms are saved as .png images using the CET-L20 colormap (colorcet).
- If --annotation_json is provided, annotations overlapping each window are
  drawn as a semi-transparent white highlight box. If freq_min/freq_max exist
  in the JSON the box is a proper 2D rect; otherwise it spans the full freq axis.
- Skips rows whose .png already exists (safe to re-run / resume).
- Rows that fail (missing audio, wrong length after load) are logged to
  {out_dir}/failed_{dataset}.tsv for inspection.
"""

import argparse
import ast
import csv
import json
import logging
import sys
from pathlib import Path

import colorcet as cc
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import soxr
from tqdm import tqdm

# ── Perch V2 frontend constants (van Merriënboer et al. 2025, Appendix A.2) ──
TARGET_SR      = 32_000
WINDOW_S       = 5.0
TARGET_SAMPLES = int(WINDOW_S * TARGET_SR)   # 160,000

N_FFT    = 1024
HOP      = 320        # 10 ms @ 32 kHz
WIN_LEN  = 640        # 20 ms @ 32 kHz
N_MELS   = 128
FMIN     = 60.0
FMAX     = 16_000.0

# Pre-compute mel filterbank and window sum once
_hann_window  = np.hanning(WIN_LEN).astype(np.float32)
_window_sum   = float(_hann_window.sum())
_mel_fb       = librosa.filters.mel(
    sr=TARGET_SR,
    n_fft=N_FFT,
    n_mels=N_MELS,
    fmin=FMIN,
    fmax=FMAX,
    htk=True,    # HTK formula — matches Perch V2
    norm=None,   # no area normalisation
).astype(np.float32)


def _safe(s: str) -> str:
    """Sanitise a field for use in a filename: replace spaces and slashes."""
    return s.strip().replace('/', '+').replace(' ', '_')


def perch_v2_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Exact reproduction of the Perch V2 frontend.

    Parameters
    ----------
    audio : np.ndarray, shape (160000,), float32
        Mono audio at 32 kHz, exactly 5 seconds.

    Returns
    -------
    log_mel : np.ndarray, shape (128, 500), float32
    """
    assert audio.shape == (TARGET_SAMPLES,), (
        f"Expected ({TARGET_SAMPLES},), got {audio.shape}"
    )

    # STFT — uncentered (first frame starts at sample 0), Hann window
    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN_LEN,
        window='hann',
        center=False,          # uncentered — matches Perch V2
        dtype=np.complex64,
    )

    # Magnitude spectrum + SciPy-style scaling (divide by sum of window values)
    magnitude = np.abs(stft) / _window_sum    # (513, 500)

    # Apply mel filterbank
    mel_spec = _mel_fb @ magnitude             # (128, 500)

    # Log with floor + scale — NOT power_to_db
    log_mel = np.log(np.maximum(mel_spec, 1e-5)) * 0.1

    return log_mel.astype(np.float32)


def load_segment(wav_path: str, offset_s: float) -> np.ndarray:
    """Load a 5-second segment from wav_path at offset_s, resampled to 32 kHz."""
    audio, native_sr = librosa.load(
        wav_path,
        sr=None,
        offset=offset_s,
        duration=WINDOW_S,
        mono=True,
    )
    if native_sr != TARGET_SR:
        audio = soxr.resample(audio.astype(np.float64), native_sr, TARGET_SR).astype(np.float32)
    else:
        audio = audio.astype(np.float32)

    # Pad if slightly short (rounding at file end), hard-fail if way too short
    if len(audio) < TARGET_SAMPLES:
        shortfall = TARGET_SAMPLES - len(audio)
        if shortfall > TARGET_SR * 0.1:   # more than 100ms missing → skip
            raise ValueError(
                f"Segment too short: got {len(audio)} samples "
                f"(need {TARGET_SAMPLES}, shortfall {shortfall})"
            )
        audio = np.pad(audio, (0, shortfall))

    return audio[:TARGET_SAMPLES]


def make_filename(dataset: str, row: dict, wav_path: str, offset_s: float, label: str) -> str:
    """
    Build output filename from all available metadata fields.

    Format:
        {dataset}_{provider}_{subdataset}_{ecotype}_{wav_stem}_{offset_s:010.1f}s_{label}.png

    Optional fields (provider, subdataset, ecotype) are included only when
    present and non-empty in the CSV row — no double underscores are produced.
    """
    wav_stem = Path(wav_path).stem
    safe_label = _safe(label)

    # Optional contextual fields — check multiple possible column names
    provider   = _safe(row.get('provider')   or row.get('Provider')   or '')
    subdataset = _safe(row.get('dataset')    or row.get('subdataset') or row.get('deployment_id') or '')
    ecotype    = _safe(row.get('ecotypes_str') or row.get('ecotype')  or '')

    parts = [dataset]
    if provider:
        parts.append(provider)
    if subdataset:
        parts.append(subdataset)
    if ecotype:
        parts.append(ecotype)
    parts.append(wav_stem)
    parts.append(f"{offset_s:010.1f}s")
    parts.append(safe_label)

    return '_'.join(parts) + '.png'


def process_csv(csv_path: str, dataset: str, out_dir: Path, annotations: list | None = None) -> dict:
    """Process all rows in one metadata CSV. Returns summary counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_log = out_dir / f"failed_{dataset}.tsv"

    counts = {'total': 0, 'done': 0, 'skipped': 0, 'failed': 0}

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    counts['total'] = len(rows)

    failed_rows = []

    for row in tqdm(rows, desc=dataset, unit='seg'):
        wav_path = row['wav_path']
        offset_s = float(row['offset_s'])
        label    = row.get('label_refined') or row.get('label_clean') or row['label']

        out_name = make_filename(dataset, row, wav_path, offset_s, label)
        out_path = out_dir / out_name

        if out_path.exists():
            counts['skipped'] += 1
            continue

        try:
            audio   = load_segment(wav_path, offset_s)
            log_mel = perch_v2_spectrogram(audio)

            fig, ax = plt.subplots(figsize=(5, 1.28), dpi=100)
            ax.imshow(log_mel, origin='lower', aspect='auto', cmap=cc.cm.CET_L20)
            ax.axis('off')

            # ── Annotation highlight boxes ────────────────────────────────────
            if annotations is not None:
                win_start = offset_s
                win_end   = offset_s + WINDOW_S
                n_frames  = log_mel.shape[1]   # 500
                n_bins    = log_mel.shape[0]   # 128

                for ann in annotations:
                    overlap = min(win_end, ann['end']) - max(win_start, ann['start'])
                    if overlap < 0.01:
                        continue

                    # Convert annotation times to frame coordinates within this window
                    t0 = max(ann['start'] - win_start, 0.0)
                    t1 = min(ann['end']   - win_start, WINDOW_S)
                    x0 = (t0 / WINDOW_S) * n_frames - 0.5
                    x1 = (t1 / WINDOW_S) * n_frames - 0.5

                    # Frequency bounds → mel bin coordinates (if available)
                    if 'freq_min' in ann and 'freq_max' in ann:
                        # Map Hz → mel bin index using the HTK scale
                        mel_freqs = librosa.mel_frequencies(
                            n_mels=n_bins, fmin=FMIN, fmax=FMAX, htk=True
                        )
                        y0 = float(np.searchsorted(mel_freqs, ann['freq_min'])) - 0.5
                        y1 = float(np.searchsorted(mel_freqs, ann['freq_max'])) - 0.5
                    else:
                        # No freq info — span full frequency axis
                        y0, y1 = -0.5, n_bins - 0.5

                    width  = x1 - x0
                    height = y1 - y0

                    # Semi-transparent white fill
                    fill = mpatches.FancyBboxPatch(
                        (x0, y0), width, height,
                        boxstyle='square,pad=0',
                        linewidth=0,
                        facecolor='white',
                        alpha=0.18,
                        transform=ax.transData,
                    )
                    ax.add_patch(fill)

                    # Thin white border
                    border = mpatches.FancyBboxPatch(
                        (x0, y0), width, height,
                        boxstyle='square,pad=0',
                        linewidth=0.8,
                        edgecolor='white',
                        facecolor='none',
                        alpha=0.7,
                        transform=ax.transData,
                    )
                    ax.add_patch(border)

            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            counts['done'] += 1
        except Exception as e:
            counts['failed'] += 1
            failed_rows.append({**row, 'error': str(e)})
            logging.warning(f"FAILED {wav_path} @ {offset_s}s — {e}")

    if failed_rows:
        keys = list(failed_rows[0].keys())
        with open(failed_log, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, delimiter='\t')
            writer.writeheader()
            writer.writerows(failed_rows)
        logging.info(f"Failed rows written to {failed_log}")

    return counts


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )

    parser = argparse.ArgumentParser(
        description='Extract Perch V2 spectrograms from metadata CSV files.'
    )
    parser.add_argument(
        '--csv', nargs='+', required=True,
        help='Path(s) to metadata CSV file(s).'
    )
    parser.add_argument(
        '--out_dir', required=True,
        help='Directory where .png spectrogram files will be saved.'
    )
    parser.add_argument(
        '--dataset', nargs='*', default=None,
        help=(
            'Dataset name(s) corresponding to each CSV (used in output filenames). '
            'If omitted, the parent directory name of each CSV is used.'
        )
    )
    args = parser.parse_args()

    csv_paths = args.csv
    out_dir   = Path(args.out_dir)

    # Resolve dataset names
    if args.dataset is None:
        dataset_names = [Path(p).parent.name for p in csv_paths]
    elif len(args.dataset) == 1 and len(csv_paths) > 1:
        # Single name provided → use it as prefix for all
        dataset_names = [f"{args.dataset[0]}_{i}" for i in range(len(csv_paths))]
    elif len(args.dataset) != len(csv_paths):
        parser.error(
            f"--dataset has {len(args.dataset)} values but --csv has {len(csv_paths)}. "
            "Provide one name per CSV, or omit --dataset entirely."
        )
    else:
        dataset_names = args.dataset

    # Process each CSV
    total_summary = {'total': 0, 'done': 0, 'skipped': 0, 'failed': 0}

    for csv_path, dataset in zip(csv_paths, dataset_names):
        logging.info(f"Processing {csv_path!r} as dataset={dataset!r}")
        counts = process_csv(csv_path, dataset, out_dir)
        logging.info(
            f"  {dataset}: {counts['done']} computed, "
            f"{counts['skipped']} skipped (already exist), "
            f"{counts['failed']} failed  (total rows: {counts['total']})"
        )
        for k in total_summary:
            total_summary[k] += counts[k]

    logging.info(
        f"\n{'='*50}\n"
        f"TOTAL: {total_summary['done']} computed, "
        f"{total_summary['skipped']} skipped, "
        f"{total_summary['failed']} failed  "
        f"(across {total_summary['total']} rows)\n"
        f"Output dir: {out_dir}\n"
        f"{'='*50}"
    )

    if total_summary['failed'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()