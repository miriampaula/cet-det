#!/usr/bin/env python3
"""
Precompute Perch V2 embeddings for the Tursiops binary pipeline.

Reads:  /data2/mromaniuc/cet-det/tursiops_perch/X_audio.npy   (N, 160000)
Writes: /data2/mromaniuc/cet-det/tursiops_perch/student_teacher/X_teacher_emb.npy  (N, 1536)

Row i in X_teacher_emb corresponds to row i in X_audio / meta_train.parquet.

Usage:
  # Needs TF + Perch — run on GPU1:
  CUDA_VISIBLE_DEVICES=1 python precompute_teacher_embeddings_tursiops.py

  # Or background:
  nohup CUDA_VISIBLE_DEVICES=1 python precompute_teacher_embeddings_tursiops.py \
      > logs/teacher_emb.log 2>&1 &
  echo $! > logs/teacher_emb.pid
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from perch_hoplite.zoo import model_configs

# ── Paths ─────────────────────────────────────────────────────────────────────
X_AUDIO_PATH = Path('/data2/mromaniuc/cet-det/tursiops_perch/X_audio.npy')
OUT_PATH     = Path('/data2/mromaniuc/cet-det/tursiops_perch/student_teacher/X_teacher_emb.npy')
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32

# ── Load Perch V2 ─────────────────────────────────────────────────────────────
print("Loading Perch V2...")
perch_wrapper = model_configs.load_model_by_name('perch_v2')
raw_model     = perch_wrapper.model
FORWARD_FN    = raw_model.signatures['serving_default']
print("Perch loaded.")

# ── Warmup ────────────────────────────────────────────────────────────────────
print("Warming up XLA...")
_ = FORWARD_FN(inputs=tf.zeros((BATCH_SIZE, 160000), tf.float32))
print("Warmup done.")

# ── Load audio ────────────────────────────────────────────────────────────────
print(f"Loading {X_AUDIO_PATH}...")
X_audio = np.load(X_AUDIO_PATH, mmap_mode='r')
print(f"  Shape: {X_audio.shape}  dtype: {X_audio.dtype}")
n_samples = X_audio.shape[0]

# Sanity: Perch expects 5s @ 32kHz = 160000 samples
assert X_audio.shape[1] == 160000, \
    f"Expected 160000 samples per window, got {X_audio.shape[1]}"

# ── Extract embeddings ────────────────────────────────────────────────────────
print(f"Extracting embeddings ({n_samples:,} samples, batch_size={BATCH_SIZE})...")
embeddings = np.zeros((n_samples, 1536), dtype=np.float32)

n_batches = int(np.ceil(n_samples / BATCH_SIZE))
for i in tqdm(range(n_batches)):
    start = i * BATCH_SIZE
    end   = min(start + BATCH_SIZE, n_samples)
    batch = tf.constant(X_audio[start:end], dtype=tf.float32)
    out   = FORWARD_FN(inputs=batch)
    embeddings[start:end] = out['embedding'].numpy()

# ── Save ──────────────────────────────────────────────────────────────────────
np.save(OUT_PATH, embeddings)
print(f"\nSaved → {OUT_PATH}")
print(f"  Shape: {embeddings.shape}  dtype: {embeddings.dtype}")

# ── Sanity checks ─────────────────────────────────────────────────────────────
norms = np.linalg.norm(embeddings, axis=1)
print(f"\nSanity checks:")
print(f"  Mean norm : {norms.mean():.4f}")
print(f"  Std norm  : {norms.std():.4f}")
print(f"  Any NaN   : {np.isnan(embeddings).any()}")
print(f"  Any Inf   : {np.isinf(embeddings).any()}")

# Cross-check row count against metadata if it exists
meta_path = OUT_PATH.parent / '../meta_train.parquet'
if meta_path.exists():
    import pandas as pd
    meta = pd.read_parquet(meta_path.resolve())
    if len(meta) != n_samples:
        print(f"\n  ⚠️  WARNING: meta_train has {len(meta):,} rows but "
              f"X_audio has {n_samples:,} — they are misaligned!")
    else:
        print(f"  Row count matches meta_train.parquet ✓  ({n_samples:,})")