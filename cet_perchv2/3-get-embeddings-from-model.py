#!/usr/bin/env python3
"""
Extract and visualise Stage 2 embeddings for s2d (proj_dim=256).
Run standalone — no need to rerun the training notebook.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import tensorflow as tf
from pathlib import Path
from perch_hoplite.zoo import model_configs

# ── Constants (must match s2d training) ──────────────────────────────────────
N_CLASSES  = 10
N_DATASETS = 11
PROJ_DIM   = 256
HEAD_UNITS = 256
HEAD_DROP  = 0.25

WEIGHTS    = Path('/data2/mromaniuc/cet-det/cet_perchv2/runs_stage2/s2d_bottleneck256_adv_0.3.weights.h5')
AUDIO_PATH = Path('/data2/mromaniuc/cet-det/cet_perchv2/X_audio.npy')
META_PATH  = Path('/data2/mromaniuc/cet-det/cet_perchv2/meta_train.parquet')
OUT_PATH   = Path('/data2/mromaniuc/cet-det/cet_perchv2/X_emb_s2d.npy')

# ── Model (must match training exactly) ──────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, lambda_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_value = float(lambda_value)
    def call(self, x):
        @tf.custom_gradient
        def _reverse(x):
            return x, lambda dy: -self.lambda_value * dy
        return _reverse(x)
    def get_config(self):
        cfg = super().get_config()
        cfg['lambda_value'] = float(self.lambda_value)
        return cfg

class CetPerchStage2(tf.keras.Model):
    def __init__(self, forward_fn, n_classes=N_CLASSES, n_datasets=N_DATASETS,
                 proj_dim=PROJ_DIM, head_units=HEAD_UNITS, head_dropout=HEAD_DROP):
        super().__init__()
        self.forward_fn = forward_fn
        self.projection = tf.keras.Sequential([
            tf.keras.layers.Dense(proj_dim, use_bias=False,
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Activation('tanh'),
        ], name='projection')
        self.class_head = tf.keras.Sequential([
            tf.keras.layers.Dropout(head_dropout),
            tf.keras.layers.Dense(head_units, activation='relu'),
            tf.keras.layers.Dropout(head_dropout),
            tf.keras.layers.Dense(n_classes),
        ], name='class_head')
        self.grl = GradientReversalLayer(lambda_value=0.0)
        self.domain_head = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(head_dropout),
            tf.keras.layers.Dense(n_datasets),
        ], name='domain_head')

    def call(self, audio, training=False):
        out  = self.forward_fn(inputs=tf.cast(audio, tf.float32))
        emb  = tf.stop_gradient(out['embedding'])
        proj = self.projection(emb, training=training)
        return {
            'embedding':     proj,
            'class_logits':  self.class_head(proj, training=training),
            'domain_logits': self.domain_head(self.grl(proj), training=training),
        }

# ── Load & run ────────────────────────────────────────────────────────────────
print("Loading Perch backbone...")
perch_wrapper = model_configs.load_model_by_name('perch_v2')
forward_fn    = perch_wrapper.model.signatures['serving_default']

print("Building model...")
model = CetPerchStage2(forward_fn)
_     = model(tf.zeros((1, 160000), tf.float32), training=False)

print(f"Loading weights from {WEIGHTS}...")
model.load_weights(str(WEIGHTS))
print("Weights loaded.")

print("Loading audio...")
X = np.load(AUDIO_PATH, mmap_mode='r')
print(f"  Shape: {X.shape}")

print("Extracting embeddings...")
BATCH = 64
embeddings = []
for i in range(0, len(X), BATCH):
    batch = tf.constant(X[i:i+BATCH], dtype=tf.float32)
    out   = model(batch, training=False)
    embeddings.append(out['embedding'].numpy())
    if i % 2000 == 0:
        print(f"  {i}/{len(X)}")

emb = np.concatenate(embeddings, axis=0)
np.save(OUT_PATH, emb)
print(f"Saved → {OUT_PATH}  shape={emb.shape}")