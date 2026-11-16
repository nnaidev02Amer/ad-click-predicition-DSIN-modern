"""Evaluation on the held-out (time-based) test split. Only measured numbers are ever written."""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score, roc_curve

from .config import Paths
from .model import load_trained
from .training import load_processed, split_indices, take

log = logging.getLogger('dsin.evaluation')

THRESHOLD = 0.5   # documented decision threshold used for the confusion matrix


def compute_metrics(y_true, y_prob, threshold=THRESHOLD) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), 1e-7, 1 - 1e-7)
    out = {'n': int(len(y_true)), 'positive_rate': float(y_true.mean()), 'logloss': float(log_loss(y_true, y_prob, labels=[0, 1]))}
    if len(np.unique(y_true)) == 2:
        out['auc'] = float(roc_auc_score(y_true, y_prob))
    else:
        out['auc'] = None          # undefined when only one class is present
    tn, fp, fn, tp = confusion_matrix(y_true, (y_prob >= threshold).astype(int), labels=[0, 1]).ravel()
    out.update({'threshold': threshold, 'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}})
    return out


def evaluate(paths: Paths, batch_size=2 ** 14):
    from .config import cpu_only
    cpu_only()
    schema, inputs, label, ts = load_processed(paths)
    model, cfg = load_trained(paths.models, schema)
    _, _, te = split_indices(ts, cfg['training']['val_fraction'])
    if len(te) == 0:
        raise ValueError('Test split is empty.')
    t0 = time.time()
    prob = model.predict(take(inputs, te), batch_size=batch_size, verbose=0).ravel()
    m = compute_metrics(label[te], prob)
    m.update({'mode': paths.mode, 'synthetic_data': paths.mode == 'demo', 'evaluated_on': 'time-based held-out test split',
              'predict_seconds': round(time.time() - t0, 2), 'evaluated_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    if paths.mode == 'demo':
        m['warning'] = 'DEMO MODE - SYNTHETIC DATA. These numbers are NOT Taobao results.'
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / 'evaluation.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
    if m['auc'] is not None:
        fpr, tpr, thr = roc_curve(label[te], prob)
        pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thr}).to_csv(paths.results / 'roc_curve.csv', index=False)
    k = min(1000, len(te))
    pd.DataFrame({'label': label[te][:k], 'predicted_ctr': prob[:k]}).to_csv(paths.results / 'prediction_samples.csv', index=False)
    pd.DataFrame({'label': label[te], 'predicted_ctr': prob}).to_pickle(paths.results / 'test_predictions.pkl')
    return m
