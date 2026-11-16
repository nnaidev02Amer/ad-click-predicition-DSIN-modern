"""Inference helper shared by the CLI and the dashboard: raw form values -> model tensors -> CTR."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .config import Paths


class Predictor:
    def __init__(self, paths: Paths):
        C.cpu_only()
        from .model import load_trained
        self.paths = paths
        self.schema = json.loads((paths.processed / 'schema.json').read_text())
        self.model, self.cfg = load_trained(paths.models, self.schema)
        lk = pd.read_pickle(paths.processed / 'lookup.pkl')
        self.ads = lk['ads']
        self.cate_classes, self.brand_classes = lk['cate_classes'], lk['brand_classes']
        self.sess_count, self.sess_len = self.schema['sess_count'], self.schema['sess_max_len']

    # -- helpers for the UI --------------------------------------------------------------------
    @property
    def synthetic(self) -> bool:
        return bool(self.cfg.get('trained_on_synthetic_data'))

    def cate_enc(self, raw):
        i = np.flatnonzero(self.cate_classes == raw)
        if not len(i):
            raise ValueError(f'Unknown category id {raw}')
        return int(i[0]) + 1

    def brand_enc(self, raw):
        i = np.flatnonzero(self.brand_classes == raw)
        if not len(i):
            raise ValueError(f'Unknown brand id {raw}')
        return int(i[0]) + 1

    def build_inputs(self, profile: dict, ad_row: dict, pid: str, sessions: list, user_index: int | None = None):
        """profile: raw values per USER_PROFILE_FEATURES; ad_row: one row of `self.ads` (optionally with cate/brand
        overridden); sessions: list (<= sess_count) of lists of (cate_raw, brand_raw), most recent session first."""
        s = self.schema
        x = {}
        for f in C.USER_PROFILE_FEATURES:
            choices = s['profile_choices'][f]
            if profile[f] not in choices:
                raise ValueError(f'{f}={profile[f]!r} not a known value: {choices}')
            x[f] = choices.index(profile[f])
        if pid not in s['pid_choices']:
            raise ValueError(f'unknown pid {pid!r}')
        x['pid'] = s['pid_choices'].index(pid)
        x['userid'] = s['vocab_sizes']['userid'] - 1 if user_index is None else int(user_index)   # cold-start slot
        x['adgroup_id'] = int(ad_row['adgroup_enc'])
        x['campaign_id'] = int(ad_row['campaign_enc'])
        x['customer'] = int(ad_row['customer_enc'])
        x['cate_id'] = int(ad_row['cate_id'])
        x['brand'] = int(ad_row['brand'])
        x['price'] = float(ad_row['price_scaled'])
        if len(sessions) > self.sess_count:
            raise ValueError(f'at most {self.sess_count} sessions are supported')
        valid = 0
        for i in range(self.sess_count):
            seq = sessions[i] if i < len(sessions) else []
            seq = seq[-self.sess_len:]
            c = np.zeros(self.sess_len, np.int32)
            b = np.zeros(self.sess_len, np.int32)
            for j, (cr, br) in enumerate(seq):
                c[j], b[j] = self.cate_enc(cr), self.brand_enc(br)
            x[f'sess_{i}_cate_id'], x[f'sess_{i}_brand'] = c[None], b[None]
            valid += int(len(seq) > 0)
        x['sess_length'] = np.array([valid], np.int32)
        for k, v in list(x.items()):
            if not isinstance(v, np.ndarray):
                x[k] = np.array([v], np.float32 if k == 'price' else np.int32)
        return x

    def predict_one(self, x: dict) -> float:
        p = float(self.model.predict(x, verbose=0).ravel()[0])
        if not (0.0 <= p <= 1.0) or np.isnan(p):
            raise RuntimeError(f'Model produced an invalid probability: {p}')
        return p

    def predict_batch(self, x: dict, batch_size=4096):
        return self.model.predict(x, batch_size=batch_size, verbose=0).ravel()
