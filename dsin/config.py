"""Central configuration: hyper-parameters and the directory layout.

Two data "modes" are kept strictly apart on disk so synthetic results can never be
mistaken for results on the genuine Taobao data:

    real : raw_data/            -> processed_data/real  -> models/real  -> results/real
    demo : demo_data/raw/       -> processed_data/demo  -> models/demo  -> results/demo
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- values taken unchanged from the original code/config.py ---------------------
FRAC = 0.25                  # fraction of users sampled
DSIN_SESS_COUNT = 5          # max number of sessions per sample
DSIN_SESS_MAX_LEN = 10       # max behaviours kept per session
ID_OFFSET = 1000             # head-room added to every vocabulary size
SESSION_GAP_SECONDS = 30 * 60           # new session when gap > 30 min
BEHAVIOR_START_TS = 1493769600          # 2017-05-03 00:00:00 +08:00 - history window start
SPLIT_TS = 1494633600                   # 2017-05-13 00:00:00 +08:00 - train < SPLIT_TS <= test
FIRST_LAST_TIME = 1483574401            # initial "last_time" in the original session code

SPARSE_FEATURES = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code',
                   'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
                   'campaign_id', 'customer']
DENSE_FEATURES = ['price']
SESS_FEATURES = ['cate_id', 'brand']
USER_PROFILE_FEATURES = ['cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level',
                         'shopping_level', 'occupation', 'new_user_class_level']

# Accepted file names for the raw behaviour log (the Tianchi archive ships `behavior_log.csv`,
# the original README says `raw_behavior_log.csv`; the original code reads `behavior_log.csv`).
BEHAVIOR_LOG_NAMES = ['behavior_log.csv', 'raw_behavior_log.csv']

REQUIRED_RAW = {
    'raw_sample.csv': ['user', 'time_stamp', 'adgroup_id', 'pid', 'nonclk', 'clk'],
    'ad_feature.csv': ['adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'price'],
    'user_profile.csv': ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
                         'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level'],
    'behavior_log': ['user', 'time_stamp', 'btag', 'cate', 'brand'],
}

MODES = ('real', 'demo')


@dataclass(frozen=True)
class ModelParams:
    att_embedding_size: int = 1
    att_head_num: int = 8
    dnn_hidden_units: tuple = (200, 80)
    dnn_activation: str = 'relu'
    bias_encoding: bool = False      # README of the original repo: True may give NaN loss
    embedding_dim: int = 4           # per sparse feature (cate_id+brand = 8 = 1*8 heads)


@dataclass(frozen=True)
class Paths:
    mode: str
    root: Path = ROOT   # overridable (tests use a temp directory)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    @property
    def raw(self) -> Path:
        return self.root / 'raw_data' if self.mode == 'real' else self.root / 'demo_data' / 'raw'

    @property
    def processed(self) -> Path:
        return self.root / 'processed_data' / self.mode

    @property
    def models(self) -> Path:
        return self.root / 'models' / self.mode

    @property
    def results(self) -> Path:
        return self.root / 'results' / self.mode

    def ensure(self):
        for p in (self.processed, self.models, self.results):
            p.mkdir(parents=True, exist_ok=True)

    def behavior_log(self):
        for n in BEHAVIOR_LOG_NAMES:
            if (self.raw / n).exists():
                return self.raw / n
        return None


def cpu_only():
    """Force CPU execution (must be called before TensorFlow initialises)."""
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
