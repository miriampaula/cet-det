"""
extract_spectrograms_fremantle.py
----------------------------------
Rebuilds the Fremantle dolphin whistle windows using the exact same padding
logic that was used during embedding, then computes the Perch V2 log-mel
spectrogram for each window and saves it as a .png image.

Since the original windows_data list was not saved to disk, this script
reconstructs it deterministically from the source WAV files using the same:
  - file sorting order (sorted glob)
  - minimum duration threshold (3.75s)
  - padding strategy (append noise from signal's own quiet frames, no crossfade)
  - trimming strategy (first 5s only for files >= 5s)

Output filename convention:
    FREMANTLE_{embedding_idx:05d}_{wav_stem}_{strategy}_{label}.png

    e.g.  FREMANTLE_00000_whistle_001_padded_Tursiops_truncatus.png

The embedding_idx in the filename matches the row index in metadata.csv,
ensuring a 1-to-1 correspondence between spectrograms and embeddings.

Usage
-----
    python extract_spectrograms_fremantle.py \\
        --wav_folder /data2/.../FREMANTLE/audio \\
        --out_dir    /data2/.../FREMANTLE/spectrograms

    # Resume-safe: skips any .png that already exists.
    # Failed files are logged to {out_dir}/failed_FREMANTLE.tsv

Notes
-----
- Native SR is 96 kHz; resampled to 32 kHz → full 16 kHz Perch V2 range.
- Files < 3.75s are excluded (4 files, 1.2% of dataset) — same as embedding.
- Files >= 5s are trimmed to the first 5-second window only.
- Files between 3.75s and 5s are padded by appending tiled quiet frames.
- The Perch V2 frontend parameters match extract_spectrograms.py exactly:
    N_FFT=1024, HOP=320, WIN_LEN=640, N_MELS=128, FMIN=60, FMAX=16000 Hz,
    HTK mel scale, no area normalization, uncentered STFT.
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
import soxr
from tqdm import tqdm

# ── Pipeline constants ────────────────────────────────────────────────────────
TARGET_SR      = 32_000
WINDOW_S       = 5.0
TARGET_SAMPLES = int(WINDOW_S * TARGET_SR)   # 160,000
MIN_DURATION_S = 3.75                         # files below this are dropped
LABEL          = 'Tursiops_truncatus'         # all Fremantle files are this class

# ── Perch V2 frontend constants ───────────────────────────────────────────────
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


# ── Padding logic (exact replica of notebook) ─────────────────────────────────

def sample_background_noise(signal: np.ndarray,
                             n_samples: int,
                             noise_floor_percentile: int = 10) -> np.ndarray:
    """
    Estimate background noise from the quietest frames of the signal itself,
    then tile to n_samples.

    This is the exact function used in the Fremantle embedding notebook.
    """
    frame_len = TARGET_SR // 10  # 100ms frames @ 32 kHz = 3200 samples
    frames = librosa.util.frame(signal,
                                frame_length=frame_len,
                                hop_length=frame_len)
    energies = np.mean(frames ** 2, axis=0)
    threshold = np.percentile(energies, noise_floor_percentile)
    quiet_frames = frames[:, energies <= threshold]

    if quiet_frames.shape[1] == 0:
        # fallback: gaussian noise matched to signal std
        return np.random.normal(0, signal.std() * 0.1,
                                n_samples).astype(np.float32)

    noise_template = quiet_frames.flatten(order='F')
    repeats = int(np.ceil(n_samples / len(noise_template)))
    noise = np.tile(noise_template, repeats)[:n_samples]
    return noise.astype(np.float32)


def pad_to_window(signal: np.ndarray,
                  target_samples: int = TARGET_SAMPLES) -> np.ndarray:
    """
    Pad signal to target_samples by appending background noise estimated
    from the signal's own noise floor.

    Signal is placed at the START (offset=0), noise appended at the end.
    No crossfade — this matches the Fremantle notebook exactly.
    """
    n_pad = target_samples - len(signal)
    if n_pad <= 0:
        return signal[:target_samples]
    noise_pad = sample_background_noise(signal, n_pad)
    return np.concatenate([signal, noise_pad]).astype(np.float32)


def build_windows(wav_folder: Path) -> list[dict]:
    """
    Reconstruct windows_data deterministically from WAV files,
    using the exact same logic as the embedding notebook.
    """
    wav_files = sorted(wav_folder.glob('*.wav'))
    logging.info(f'Found {len(wav_files)} WAV files in {wav_folder}')

    windows = []
    dropped = []

    for f in tqdm(wav_files, desc='Building windows', unit='file'):
        try:
            audio, native_sr = librosa.load(str(f), sr=None, mono=True)
            if native_sr != TARGET_SR:
                audio = soxr.resample(audio.astype(np.float64),
                                      native_sr, TARGET_SR).astype(np.float32)
            else:
                audio = audio.astype(np.float32)

            duration_s = len(audio) / TARGET_SR

            # ── Drop files below minimum duration ─────────────────────────
            if duration_s < MIN_DURATION_S:
                dropped.append(f.name)
                continue

            # ── Pad or trim to exactly 5s ──────────────────────────────────
            if len(audio) < TARGET_SAMPLES:
                audio_out = pad_to_window(audio)
                strategy  = 'padded'
            else:
                audio_out = audio[:TARGET_SAMPLES]
                strategy  = 'trimmed'

            windows.append({
                'file':                f.name,
                'stem':                f.stem,
                'audio':               audio_out,
                'duration_original_s': round(duration_s, 3),
                'strategy':            strategy,
                'label':               LABEL,
                'native_sr':           native_sr,
            })

        except Exception as e:
            logging.warning(f'FAILED loading {f.name}: {e}')

    logging.info(f'Windows built: {len(windows)} kept, {len(dropped)} dropped '
                 f'(< {MIN_DURATION_S}s): {dropped}')
    return windows


# ── Perch V2 spectrogram ──────────────────────────────────────────────────────

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
        f'Expected ({TARGET_SAMPLES},), got {audio.shape}'
    )

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN_LEN,
        window='hann',
        center=False,       # uncentered — matches Perch V2
        dtype=np.complex64,
    )

    magnitude = np.abs(stft) / _window_sum    # (513, 500)
    mel_spec  = _mel_fb @ magnitude            # (128, 500)
    log_mel   = np.log(np.maximum(mel_spec, 1e-5)) * 0.1

    return log_mel.astype(np.float32)


# ── Filename builder ──────────────────────────────────────────────────────────

def make_filename(embedding_idx: int, stem: str,
                  strategy: str, label: str) -> str:
    """
    Format: FREMANTLE_{embedding_idx:05d}_{wav_stem}_{strategy}_{label}.png
    """
    safe_label = label.replace(' ', '_').replace('/', '+')
    return f'FREMANTLE_{embedding_idx:05d}_{stem}_{strategy}_{safe_label}.png'


# ── Main processing ───────────────────────────────────────────────────────────

def process(wav_folder: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_log = out_dir / 'failed_FREMANTLE.tsv'

    windows = build_windows(wav_folder)
    counts  = {'total': len(windows), 'done': 0, 'skipped': 0, 'failed': 0}
    failed_rows = []

    for embedding_idx, w in enumerate(tqdm(windows, desc='FREMANTLE', unit='win')):
        out_name = make_filename(embedding_idx, w['stem'], w['strategy'], w['label'])
        out_path = out_dir / out_name

        if out_path.exists():
            counts['skipped'] += 1
            continue

        try:
            audio   = w['audio']
            log_mel = perch_v2_spectrogram(audio)

            fig, ax = plt.subplots(figsize=(5, 1.28), dpi=100)
            ax.imshow(log_mel, origin='lower', aspect='auto', cmap=cc.cm.CET_L20)
            ax.axis('off')

            # ── Mark real/pad boundary for padded windows ─────────────────
            if w['strategy'] == 'padded':
                n_frames   = log_mel.shape[1]  # 500
                real_end_s = w['duration_original_s']
                x_boundary = (real_end_s / WINDOW_S) * n_frames - 0.5
                ax.axvline(x_boundary, color='cyan',
                           linewidth=1.0, linestyle='--', alpha=0.7)

            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            counts['done'] += 1

        except Exception as e:
            counts['failed'] += 1
            failed_rows.append({
                'embedding_idx': embedding_idx,
                'file':          w['file'],
                'strategy':      w['strategy'],
                'error':         str(e),
            })
            logging.warning(f'FAILED [{embedding_idx}] {w["file"]} — {e}')

    if failed_rows:
        with open(failed_log, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=list(failed_rows[0].keys()), delimiter='\t'
            )
            writer.writeheader()
            writer.writerows(failed_rows)
        logging.info(f'Failed rows → {failed_log}')

    return counts


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )

    parser = argparse.ArgumentParser(
        description='Extract Perch V2 spectrograms for the Fremantle dataset '
                    'by rebuilding windows from source WAVs.'
    )
    parser.add_argument(
        '--wav_folder', required=True,
        help='Folder containing the Fremantle .wav files.'
    )
    parser.add_argument(
        '--out_dir', required=True,
        help='Directory where .png spectrogram files will be saved.'
    )
    args   = parser.parse_args()
    counts = process(Path(args.wav_folder), Path(args.out_dir))

    logging.info(
        f"\n{'='*50}\n"
        f"TOTAL : {counts['total']} windows\n"
        f"Done  : {counts['done']} computed\n"
        f"Skip  : {counts['skipped']} already existed\n"
        f"Fail  : {counts['failed']} errors\n"
        f"{'='*50}"
    )

    if counts['failed'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()