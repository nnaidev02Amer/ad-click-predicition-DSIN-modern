"""Training pipeline: time-based train/test split (as in the original), a time-ordered validation tail,
early stopping, checkpointing, history and configuration saving."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .config import ModelParams, Paths
from .model import build_dsin, params_to_dict

log = logging.getLogger('dsin.training')


def load_processed(paths: Paths):
    p = paths.processed
    for n in ('dsin_input.pkl', 'dsin_label.pkl', 'dsin_time_stamp.pkl', 'schema.json'):
        if not (p / n).exists():
            raise FileNotFoundError(f'{p / n} not found - run run_preprocessing.py for mode "{paths.mode}" first.')
    schema = json.loads((p / 'schema.json').read_text())
    return (schema, pd.read_pickle(p / 'dsin_input.pkl'), pd.read_pickle(p / 'dsin_label.pkl'),
            pd.read_pickle(p / 'dsin_time_stamp.pkl'))


def split_indices(time_stamp: np.ndarray, val_fraction=0.1, split_ts=C.SPLIT_TS):
    """train = before split_ts (minus the latest `val_fraction` -> validation); test = split_ts onwards."""
    train_all = np.flatnonzero(time_stamp < split_ts)
    test = np.flatnonzero(time_stamp >= split_ts)
    order = train_all[np.argsort(time_stamp[train_all], kind='stable')]
    n_val = int(len(order) * val_fraction)
    return order[:len(order) - n_val], (order[len(order) - n_val:] if n_val else order[:0]), test


def take(inputs: dict, idx):
    return {k: v[idx] for k, v in inputs.items()}


def train(paths: Paths, epochs=3, batch_size=4096, learning_rate=0.01, val_fraction=0.1, patience=2,
          params: ModelParams = ModelParams(), seed=1024, max_train_rows=None):
    C.cpu_only()
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
    paths.ensure()
    tf.keras.utils.set_random_seed(seed)

    schema, inputs, label, ts = load_processed(paths)
    tr, va, te = split_indices(ts, val_fraction)
    if max_train_rows and len(tr) > max_train_rows:
        tr = np.sort(np.random.default_rng(seed).choice(tr, max_train_rows, replace=False))
    if len(tr) == 0:
        raise ValueError('Training split is empty - check time stamps / SPLIT_TS.')
    log.info('train=%d validation=%d test=%d', len(tr), len(va), len(te))

    model = build_dsin(schema, params, seed)
    model.compile(tf.keras.optimizers.Adagrad(learning_rate=learning_rate), 'binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc'), 'binary_crossentropy'])

    ckpt = paths.models / 'dsin.weights.h5'
    monitor = 'val_loss' if len(va) else 'loss'
    callbacks = [ModelCheckpoint(str(ckpt), monitor=monitor, save_best_only=True, save_weights_only=True),
                 EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True),
                 CSVLogger(str(paths.results / 'training_log.csv'))]
    t0 = time.time()
    hist = model.fit(take(inputs, tr), label[tr], batch_size=batch_size, epochs=epochs, verbose=2,
                     validation_data=(take(inputs, va), label[va]) if len(va) else None, callbacks=callbacks)
    secs = time.time() - t0

    model.save_weights(str(ckpt))        # = best weights (restore_best_weights=True)
    history = {k: [float(x) for x in v] for k, v in hist.history.items()}
    (paths.results / 'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    cfg = {'mode': paths.mode, 'trained_on_synthetic_data': paths.mode == 'demo',
           'model_params': params_to_dict(params),
           'training': {'epochs_requested': epochs, 'epochs_run': len(history['loss']), 'batch_size': batch_size,
                        'learning_rate': learning_rate, 'optimizer': 'Adagrad', 'loss': 'binary_crossentropy',
                        'val_fraction': val_fraction, 'early_stopping_patience': patience, 'seed': seed,
                        'n_train': int(len(tr)), 'n_validation': int(len(va)), 'n_test': int(len(te)),
                        'seconds': round(secs, 1), 'tensorflow': tf.__version__},
           'data_schema': {'frac': schema.get('frac'), 'sess_count': schema['sess_count'],
                           'sess_max_len': schema['sess_max_len']},
           'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    (paths.models / 'model_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    return model, history, cfg
