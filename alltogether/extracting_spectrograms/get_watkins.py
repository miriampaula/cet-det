"""
extract_spectrograms_watkins.py
-------------------------------
Computes the exact Perch V2 log-mel spectrogram for every window in the
Watkins pre-built windows_df.pkl, saving each as a .png image.

This script is intentionally separate from extract_spectrograms.py because
the Watkins pipeline does NOT load audio from WAV files at inference time.
Instead, audio was pre-processed (resampled to 32 kHz, mel-matched noise
extension, random placement, peak normalization) and stored in the pickle.
The spectrogram is computed from row['audio'] — the exact same signal that
was fed to the Perch V2 embedding model — ensuring a 1-to-1 correspondence
between embeddings and spectrograms.

Output filename convention:
    WATKINS_{species}_{source_file_stem}_{window_idx:04d}_{window_kind}_{label}.png

    e.g.  WATKINS_Tursiops_truncatus_89514001_0000_extended_Tursiops_truncatus.png

Usage
-----
    python extract_spectrograms_watkins.py \\
        --pkl     /data2/.../WATKINS/windows_df.pkl \\
        --out_dir /data2/.../WATKINS/spectrograms

    # Resume-safe: skips any .png that already exists.
    # Failed rows are logged to {out_dir}/failed_WATKINS.tsv

Notes
-----
- The Perch V2 frontend parameters exactly match those in extract_spectrograms.py:
    N_FFT=1024, HOP=320, WIN_LEN=640, N_MELS=128, FMIN=60, FMAX=16000 Hz,
    HTK mel scale, no area normalization, uncentered STFT.
- All audio in the pickle is already at 32 kHz and exactly 160,000 samples.
  A safety pad/clip is applied just in case.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import colorcet as cc
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Perch V2 frontend constants ───────────────────────────────────────────────
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
_hann_window = np.hanning(WIN_LEN).astype(np.float32)
_window_sum  = float(_hann_window.sum())
_mel_fb      = librosa.filters.mel(
    sr=TARGET_SR,
    n_fft=N_FFT,
    n_mels=N_MELS,
    fmin=FMIN,
    fmax=FMAX,
    htk=True,   # HTK formula — matches Perch V2
    norm=None,  # no area normalisation
).astype(np.float32)


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
        center=False,       # uncentered — matches Perch V2
        dtype=np.complex64,
    )

    # Magnitude spectrum + SciPy-style scaling
    magnitude = np.abs(stft) / _window_sum    # (513, 500)

    # Apply mel filterbank
    mel_spec = _mel_fb @ magnitude             # (128, 500)

    # Log with floor + scale — NOT power_to_db
    log_mel = np.log(np.maximum(mel_spec, 1e-5)) * 0.1

    return log_mel.astype(np.float32)


def make_filename(row: dict) -> str:
    """
    Build output filename from window metadata.

    Format:
        WATKINS_{species}_{source_file_stem}_{window_idx:04d}_{window_kind}_{label}.png
    """
    species    = str(row.get('species', 'unknown')).replace(' ', '_')
    src_stem   = Path(str(row.get('source_file', 'unknown'))).stem
    win_idx    = int(row.get('window_idx', 0))
    win_kind   = str(row.get('window_kind', 'unknown'))
    label      = str(row.get('label', 'unknown')).replace(' ', '_').replace('/', '+')

    return f"MONISH_{species}_{src_stem}_{win_idx:04d}_{win_kind}_{label}.png"


def process_pickle(pkl_path: str, out_dir: Path) -> dict:
    """Load windows_df pickle and compute spectrograms. Returns summary counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_log = out_dir / 'failed_WATKINS.tsv'

    logging.info(f"Loading pickle: {pkl_path}")
    windows_df = pd.read_pickle(pkl_path)
    logging.info(f"Loaded {len(windows_df)} windows across "
                 f"{windows_df['species'].nunique()} species")

    counts = {'total': len(windows_df), 'done': 0, 'skipped': 0, 'failed': 0}
    failed_rows = []

    for i, row in tqdm(windows_df.iterrows(), total=len(windows_df), desc='MONISH', unit='win'):
        out_name = make_filename(row)
        out_path = out_dir / out_name

        if out_path.exists():
            counts['skipped'] += 1
            continue

        try:
            # ── Use the exact audio that was fed to Perch ──────────────────
            audio = row['audio'].astype(np.float32)

            # Safety: pad or clip to exactly TARGET_SAMPLES
            if len(audio) < TARGET_SAMPLES:
                audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
            audio = audio[:TARGET_SAMPLES]

            log_mel = perch_v2_spectrogram(audio)

            fig, ax = plt.subplots(figsize=(5, 1.28), dpi=100)
            ax.imshow(log_mel, origin='lower', aspect='auto', cmap=cc.cm.CET_L20)
            ax.axis('off')

            # Mark signal boundaries for extended windows
            win_kind  = str(row.get('window_kind', ''))
            sig_start = row.get('sig_start_sample', None)
            sig_len   = row.get('sig_len_sample', None)

            if win_kind in ('extended', 'tail_extended') \
                    and sig_start is not None and sig_len is not None:
                n_frames = log_mel.shape[1]  # 500
                x_start  = (sig_start / TARGET_SAMPLES) * n_frames - 0.5
                x_end    = ((sig_start + sig_len) / TARGET_SAMPLES) * n_frames - 0.5
                ax.axvline(x_start, color='cyan',  linewidth=1.0, linestyle='--', alpha=0.7)
                ax.axvline(x_end,   color='lime',  linewidth=1.0, linestyle='--', alpha=0.7)

            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            counts['done'] += 1

        except Exception as e:
            counts['failed'] += 1
            failed_rows.append({
                'species':     row.get('species', ''),
                'source_file': row.get('source_file', ''),
                'window_idx':  row.get('window_idx', ''),
                'window_kind': row.get('window_kind', ''),
                'error':       str(e),
            })
            logging.warning(
                f"FAILED [{i}] {row.get('species','?')}/{row.get('source_file','?')} "
                f"win={row.get('window_idx','?')} — {e}"
            )

    if failed_rows:
        with open(failed_log, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=list(failed_rows[0].keys()), delimiter='\t'
            )
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
        description='Extract Perch V2 spectrograms for Watkins dataset '
                    'from pre-built windows_df.pkl.'
    )
    parser.add_argument(
        '--pkl', required=True,
        help='Path to windows_df.pkl produced by the Watkins embedding pipeline.'
    )
    parser.add_argument(
        '--out_dir', required=True,
        help='Directory where .png spectrogram files will be saved.'
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    counts = process_pickle(args.pkl, out_dir)

    logging.info(
        f"\n{'='*50}\n"
        f"TOTAL : {counts['total']} windows\n"
        f"Done  : {counts['done']} computed\n"
        f"Skip  : {counts['skipped']} already existed\n"
        f"Fail  : {counts['failed']} errors\n"
        f"Output: {out_dir}\n"
        f"{'='*50}"
    )

    if counts['failed'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
    