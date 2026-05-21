"""
perch_inference.py
------------------
Runs Perch V2 inference on a folder of WAV files using multiple workers.
Splits each WAV into 5-second windows, resamples to 32 kHz with soxr,
extracts embeddings + logits, and saves:

  <save_dir>/
    embeddings.npy      – float32, shape (N, embedding_dim)
    logits.npy          – float32, shape (N, n_classes)
    metadata.csv        – wav_path, segment_index, offset_s,
                          top_predicted_class, top_logit_score
    processing_log.csv  – one row per file: status, native_sr_hz, duration_s,
                          n_segments, resampled, error_message
    inference.log       – full timestamped text log
    checkpoints/
      worker_0/         – per-worker checkpoint .npy and .csv files
      worker_1/
      ...

Usage
-----
    python perch_inference.py \
        --wav_folder  /path/to/wavs \
        --save_dir    /path/to/output \
        [--window_size 5.0] \
        [--batch_size 64] \
        [--workers 4] \
        [--checkpoint_every 5]

Resume
------
    Rerun the exact same command — already-processed files in
    processing_log.csv are skipped automatically.

Install
-------
    pip install tensorflow~=2.20.0 librosa soxr pandas numpy perch_hoplite
"""

import argparse
import fcntl
import logging
import multiprocessing as mp
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soxr

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path, worker_id: int | None = None) -> logging.Logger:
    name = "perch_inference" if worker_id is None else f"worker_{worker_id}"
    log  = logging.getLogger(name)
    if log.handlers:          # avoid duplicate handlers on re-entry
        return log
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Perch V2 batch inference (multiprocessing)")
    p.add_argument("--wav_folder",       required=True)
    p.add_argument("--save_dir",         required=True)
    p.add_argument("--window_size",      type=float, default=5.0)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--workers",          type=int,   default=4)
    p.add_argument("--checkpoint_every", type=int,   default=5)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_and_resample(wav_path: Path, target_sr: int) -> tuple[np.ndarray, int, bool]:
    audio, native_sr = librosa.load(str(wav_path), sr=None, mono=True)
    resampled = False
    if native_sr != target_sr:
        audio = soxr.resample(audio, native_sr, target_sr, quality="HQ")
        resampled = True
    return audio.astype(np.float32), native_sr, resampled


def make_windows(audio: np.ndarray, sr: int, window_size_s: float):
    samples_per_window = int(window_size_s * sr)
    n_windows = len(audio) // samples_per_window
    for i in range(n_windows):
        start = i * samples_per_window
        yield i, i * window_size_s, audio[start : start + samples_per_window]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_batch(chunks: list, model) -> tuple[np.ndarray, np.ndarray]:
    batch   = np.stack(chunks, axis=0)
    outputs = model.embed(batch)
    embeddings = np.array(outputs.embeddings, dtype=np.float32)
    if isinstance(outputs.logits, dict):
        logits = np.array(list(outputs.logits.values())[0], dtype=np.float32)
    else:
        logits = np.array(outputs.logits, dtype=np.float32)
    return embeddings, logits


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker_fn(worker_id: int, wav_paths: list[str], save_dir: str,
              window_size: float, batch_size: int, checkpoint_every: int,
              proc_log_path: str):
    """Runs in a separate process. Each worker has its own model instance."""

    # Import TF inside worker so it initialises fresh per process
    import tensorflow as tf  # noqa: F401
    from perch_hoplite.zoo import model_configs

    save_dir      = Path(save_dir)
    proc_log_path = Path(proc_log_path)
    ckpt_dir      = save_dir / "checkpoints" / f"worker_{worker_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(save_dir / "inference.log", worker_id)
    log.info(f"Starting — {len(wav_paths)} file(s) assigned")

    # Load model
    model       = model_configs.load_model_by_name("perch_v2")
    sr          = model.sample_rate
    class_names = list(model.class_list["labels"].classes)
    log.info(f"Model loaded (SR={sr}, classes={len(class_names)})")

    # Checkpoint index — continue from last saved
    def next_ckpt_index():
        existing = sorted(ckpt_dir.glob("embeddings_ckpt_*.npy"))
        return 0 if not existing else int(existing[-1].stem.split("_")[-1]) + 1

    def save_checkpoint(emb_list, lgt_list, meta_list, log_list, idx):
        if not emb_list:
            return
        tag = f"{idx:04d}"
        np.save(ckpt_dir / f"embeddings_ckpt_{tag}.npy", np.concatenate(emb_list))
        np.save(ckpt_dir / f"logits_ckpt_{tag}.npy",     np.concatenate(lgt_list))
        pd.DataFrame(meta_list).to_csv(ckpt_dir / f"metadata_ckpt_{tag}.csv", index=False)

        # Append to shared processing_log.csv with file lock to avoid races
        log_df = pd.DataFrame(log_list, columns=[
            "wav_path", "status", "native_sr_hz", "duration_s",
            "n_segments", "resampled", "error_message",
        ])
        with open(proc_log_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            log_df.to_csv(f, header=False, index=False)
            fcntl.flock(f, fcntl.LOCK_UN)

        log.info(f"✓ checkpoint {tag} saved — {np.concatenate(emb_list).shape[0]} segments")

    # Accumulators
    all_embeddings:  list[np.ndarray] = []
    all_logits:      list[np.ndarray] = []
    metadata_rows:   list[dict]       = []
    file_log_rows:   list[dict]       = []
    pending_chunks:  list[np.ndarray] = []
    pending_meta:    list[dict]       = []

    ckpt_idx         = next_ckpt_index()
    files_since_ckpt = 0

    def flush_batch():
        if not pending_chunks:
            return
        emb, lgt = run_batch(pending_chunks, model)
        all_embeddings.append(emb)
        all_logits.append(lgt)
        metadata_rows.extend(pending_meta)
        pending_chunks.clear()
        pending_meta.clear()

    for wav_str in wav_paths:
        wav_path = Path(wav_str)
        log.info(f"Processing: {wav_path.name}")
        file_entry = {
            "wav_path":      wav_str,
            "status":        "ok",
            "native_sr_hz":  None,
            "duration_s":    None,
            "n_segments":    0,
            "resampled":     False,
            "error_message": "",
        }

        try:
            audio, native_sr, resampled = load_and_resample(wav_path, sr)
            duration_s = len(audio) / sr
            n_windows  = int(duration_s // window_size)

            file_entry.update({
                "native_sr_hz": native_sr,
                "duration_s":   round(duration_s, 2),
                "resampled":    resampled,
                "n_segments":   n_windows,
            })

            if resampled:
                log.info(f"  native SR {native_sr} Hz → resampled to {sr} Hz")
            else:
                log.info(f"  native SR {native_sr} Hz (no resampling needed)")
            log.info(f"  duration {duration_s:.1f} s → {n_windows} segments")

        except Exception as e:
            log.warning(f"  FAILED: {e}")
            file_entry["status"]        = "error"
            file_entry["error_message"] = str(e)
            file_log_rows.append(file_entry)
            continue

        for seg_idx, offset_s, chunk in make_windows(audio, sr, window_size):
            pending_chunks.append(chunk)
            pending_meta.append({
                "wav_path":      wav_str,
                "segment_index": seg_idx,
                "offset_s":      round(offset_s, 3),
            })
            if len(pending_chunks) >= batch_size:
                flush_batch()

        file_log_rows.append(file_entry)
        files_since_ckpt += 1

        if files_since_ckpt >= checkpoint_every:
            flush_batch()
            save_checkpoint(all_embeddings, all_logits, metadata_rows, file_log_rows, ckpt_idx)
            all_embeddings.clear();  all_logits.clear()
            metadata_rows.clear();   file_log_rows.clear()
            ckpt_idx += 1;           files_since_ckpt = 0

    # Final tail checkpoint
    flush_batch()
    if all_embeddings:
        save_checkpoint(all_embeddings, all_logits, metadata_rows, file_log_rows, ckpt_idx)

    log.info("Worker done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    wav_folder = Path(args.wav_folder)
    save_dir   = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log           = setup_logging(save_dir / "inference.log")
    proc_log_path = save_dir / "processing_log.csv"

    log.info("=" * 60)
    log.info("Perch V2 inference run started")
    log.info(f"  wav_folder       : {wav_folder}")
    log.info(f"  save_dir         : {save_dir}")
    log.info(f"  window_size      : {args.window_size} s")
    log.info(f"  batch_size       : {args.batch_size}")
    log.info(f"  workers          : {args.workers}")
    log.info(f"  checkpoint_every : {args.checkpoint_every} files")
    log.info("=" * 60)

    # Discover all WAV files
    all_wav_files = sorted(wav_folder.rglob("*.wav"))
    if not all_wav_files:
        log.error(f"No .wav files found under {wav_folder}")
        raise FileNotFoundError(f"No .wav files found under {wav_folder}")
    log.info(f"Found {len(all_wav_files)} WAV file(s) total")

    # Resume: skip already-done files
    already_done: set[str] = set()
    if proc_log_path.exists():
        prev = pd.read_csv(proc_log_path)
        already_done = set(prev.loc[prev["status"] == "ok", "wav_path"].tolist())
        if already_done:
            log.info(f"Resuming — skipping {len(already_done)} already-processed file(s)")

    remaining = [str(w) for w in all_wav_files if str(w) not in already_done]
    log.info(f"Files to process this run: {len(remaining)}")

    if not remaining:
        log.info("Nothing left to process — jumping straight to merge.")
    else:
        # Write processing_log.csv header if starting fresh
        if not proc_log_path.exists():
            pd.DataFrame(columns=[
                "wav_path", "status", "native_sr_hz", "duration_s",
                "n_segments", "resampled", "error_message",
            ]).to_csv(proc_log_path, index=False)

        # Distribute files round-robin across workers so each gets a mix
        # of folders rather than one worker getting all of folder A
        n_workers = min(args.workers, len(remaining))
        chunks    = [remaining[i::n_workers] for i in range(n_workers)]

        log.info(f"Launching {n_workers} worker(s) …")
        for i, chunk in enumerate(chunks):
            log.info(f"  worker {i}: {len(chunk)} files")

        # Use spawn to avoid TF/fork conflicts
        ctx = mp.get_context("spawn")
        processes = []
        for worker_id, chunk in enumerate(chunks):
            p = ctx.Process(
                target=worker_fn,
                args=(worker_id, chunk, str(save_dir), args.window_size,
                      args.batch_size, args.checkpoint_every, str(proc_log_path)),
                name=f"worker_{worker_id}",
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                log.warning(f"{p.name} exited with code {p.exitcode}")

    # ------------------------------------------------------------------
    # Merge all worker checkpoints into final outputs
    # ------------------------------------------------------------------
    log.info("Merging all checkpoints into final outputs …")

    from perch_hoplite.zoo import model_configs
    _model      = model_configs.load_model_by_name("perch_v2")
    class_names = list(_model.class_list["labels"].classes)
    del _model

    ckpt_root  = save_dir / "checkpoints"
    emb_files  = sorted(ckpt_root.rglob("embeddings_ckpt_*.npy"))
    lgt_files  = sorted(ckpt_root.rglob("logits_ckpt_*.npy"))
    meta_files = sorted(ckpt_root.rglob("metadata_ckpt_*.csv"))

    if not emb_files:
        log.error("No checkpoint files found — nothing to merge.")
        raise RuntimeError("No checkpoint files found.")

    log.info(f"  merging {len(emb_files)} checkpoint file(s) …")
    embeddings = np.concatenate([np.load(f) for f in emb_files],  axis=0)
    logits     = np.concatenate([np.load(f) for f in lgt_files],  axis=0)
    meta_df    = pd.concat([pd.read_csv(f) for f in meta_files], ignore_index=True)

    top_idx   = np.argmax(logits, axis=1)
    top_class = [class_names[i] for i in top_idx]
    top_score = logits[np.arange(len(logits)), top_idx]

    meta_df["top_predicted_class"] = top_class
    meta_df["top_logit_score"]     = top_score.round(4)

    # Sort cleanly by file and segment
    meta_df = meta_df.sort_values(["wav_path", "segment_index"]).reset_index(drop=True)

    np.save(save_dir / "embeddings.npy", embeddings)
    np.save(save_dir / "logits.npy",     logits)
    meta_df.to_csv(save_dir / "metadata.csv", index=False)

    log_df    = pd.read_csv(proc_log_path)
    n_ok      = (log_df["status"] == "ok").sum()
    n_err     = (log_df["status"] == "error").sum()
    n_resamp  = log_df["resampled"].sum()
    sr_counts = log_df["native_sr_hz"].value_counts().to_dict()

    log.info("=" * 60)
    log.info("Run complete")
    log.info(f"  files processed  : {n_ok} ok / {n_err} failed")
    log.info(f"  resampled        : {n_resamp}")
    log.info(f"  native SR dist.  : {sr_counts}")
    log.info(f"  total segments   : {len(meta_df)}")
    log.info(f"  embeddings       : {embeddings.shape}  → embeddings.npy")
    log.info(f"  logits           : {logits.shape}  → logits.npy")
    log.info(f"  metadata         : {len(meta_df)} rows  → metadata.csv")
    log.info(f"  processing log   : {len(log_df)} rows  → processing_log.csv")
    log.info(f"  checkpoints      : {len(emb_files)}  → checkpoints/")
    log.info("Top-5 predicted classes:")
    for cls, cnt in meta_df["top_predicted_class"].value_counts().head(5).items():
        log.info(f"    {cls:<30s} {cnt}")
    if n_err:
        log.warning(f"{n_err} file(s) failed — see processing_log.csv for details")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
