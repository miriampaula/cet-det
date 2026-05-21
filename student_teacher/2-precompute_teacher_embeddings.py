# precompute_teacher_embeddings.py
"""
Run Perch V2 once over all 11,769 audio windows and cache embeddings.
Output: X_teacher_emb.npy of shape (11769, 1536), row-aligned with meta_train.
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from perch_hoplite.zoo import model_configs

# ── Paths ────────────────────────────────────────────────────────────────────
X_AUDIO_PATH   = Path('/data2/mromaniuc/cet-det/cet_perchv2/X_audio.npy')
OUT_PATH       = Path('/data2/mromaniuc/cet-det/student_teacher/X_teacher_emb.npy')
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32

# ── Load Perch ───────────────────────────────────────────────────────────────
print("Loading Perch V2...")
perch_wrapper = model_configs.load_model_by_name('perch_v2')
raw_model     = perch_wrapper.model
FORWARD_FN    = raw_model.signatures['serving_default']
print("Perch loaded.")

# ── Warmup ───────────────────────────────────────────────────────────────────
print("Warming up XLA...")
_ = FORWARD_FN(inputs=tf.zeros((BATCH_SIZE, 160000), tf.float32))
print("Warmup done.")

# ── Load audio ───────────────────────────────────────────────────────────────
print(f"Loading {X_AUDIO_PATH}...")
X_audio = np.load(X_AUDIO_PATH, mmap_mode='r')  # mmap — don't load all into RAM
print(f"  Shape: {X_audio.shape}, dtype: {X_audio.dtype}")
n_samples = X_audio.shape[0]

# ── Extract embeddings ───────────────────────────────────────────────────────
print(f"Extracting embeddings ({n_samples} samples, batch_size={BATCH_SIZE})...")
embeddings = np.zeros((n_samples, 1536), dtype=np.float32)

n_batches = int(np.ceil(n_samples / BATCH_SIZE))
for i in tqdm(range(n_batches)):
    start = i * BATCH_SIZE
    end   = min(start + BATCH_SIZE, n_samples)
    
    batch = tf.constant(X_audio[start:end], dtype=tf.float32)
    out   = FORWARD_FN(inputs=batch)
    embeddings[start:end] = out['embedding'].numpy()

# ── Save ─────────────────────────────────────────────────────────────────────
np.save(OUT_PATH, embeddings)
print(f"\nSaved → {OUT_PATH}")
print(f"Shape: {embeddings.shape}, dtype: {embeddings.dtype}")

# Quick sanity check
print(f"\nSanity check:")
print(f"  Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
print(f"  Any NaN:   {np.isnan(embeddings).any()}")
print(f"  Any Inf:   {np.isinf(embeddings).any()}")