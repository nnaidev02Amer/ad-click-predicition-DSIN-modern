"""Tests for environment, validation, preprocessing, sessions and feature dimensions (synthetic data only)."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from dsin import config as C
from dsin.config import Paths
from dsin.data_validation import check_files, validate_dataset
from dsin.preprocessing import gen_sess_feature_dsin, session_list_dsin


def test_python_environment():
    assert sys.version_info[:2] == (3, 9)


@pytest.mark.parametrize('mod', ['tensorflow', 'deepctr', 'numpy', 'pandas', 'sklearn', 'joblib', 'requests',
                                 'tqdm', 'streamlit', 'plotly'])
def test_imports(mod):
    __import__(mod)


def test_tensorflow_and_deepctr_versions():
    import tensorflow as tf
    import deepctr
    assert tf.__version__.startswith('2.15')
    assert deepctr.__version__ == '0.9.4'


def test_missing_real_dataset_is_reported(tmp_path):
    rep = check_files(Paths('real', root=tmp_path))
    assert not rep['ok'] and len(rep['missing']) == 4


def test_dataset_validation(demo_paths):
    rep = validate_dataset(demo_paths)
    assert rep['ok'], rep.get('problems')
    s = rep['stats']['raw_sample']
    assert s['clk_values_valid'] and s['rows'] == 3000
    assert 'new_user_class_level' in rep['stats']['user_profile']['missing']      # trailing space stripped


def test_session_splitting():
    t0 = C.BEHAVIOR_START_TS + 100
    ts = np.array([t0, t0 + 60, t0 + 120, t0 + 5000, t0 + 5060, t0 + 5120, t0 + 5180, t0 + 20000, t0 + 20001])
    cate = np.arange(1, 10)
    sess = session_list_dsin(ts, cate, cate)
    assert [len(s) for s in sess] == [3, 4]          # 3-gap >30 min; last 2-behaviour session dropped (needs >2)


def test_session_features_no_future_leak():
    t0 = C.BEHAVIOR_START_TS + 100
    ts = np.array([t0, t0 + 60, t0 + 120, t0 + 5000, t0 + 5060, t0 + 5120, t0 + 5180])
    sess = session_list_dsin(ts, np.arange(1, 8), np.arange(1, 8))
    cates, brands, lens, valid = gen_sess_feature_dsin(sess, t0 + 5100)     # inside 2nd session, 2 behaviours before
    assert valid == 1 and lens[0] == 3 and cates[0] == [1, 2, 3] and lens[1] == 0
    cates, _, lens, valid = gen_sess_feature_dsin(sess, t0 + 60)             # before any complete history
    assert valid == 0 and lens == [0] * 5
    cates, _, lens, valid = gen_sess_feature_dsin(sess, t0 + 99999)
    assert valid == 2 and cates[0] == [4, 5, 6, 7] and cates[1] == [1, 2, 3]


def test_preprocessing_outputs_and_dimensions(demo_paths):
    p = demo_paths.processed
    schema = json.loads((p / 'schema.json').read_text())
    x = pd.read_pickle(p / 'dsin_input.pkl')
    y = pd.read_pickle(p / 'dsin_label.pkl')
    n = len(y)
    assert set(np.unique(y)) <= {0, 1}
    for f in C.SPARSE_FEATURES + C.SESS_FEATURES:
        assert x[f].shape == (n,) and x[f].min() >= 0 and x[f].max() < schema['vocab_sizes'][f], f
    assert x['price'].shape == (n,)
    for i in range(schema['sess_count']):
        for f in C.SESS_FEATURES:
            a = x[f'sess_{i}_{f}']
            assert a.shape == (n, schema['sess_max_len']) and a.max() < schema['vocab_sizes'][f]
    assert x['sess_length'].shape == (n,) and x['sess_length'].max() <= schema['sess_count']
    assert (p / 'lookup.pkl').exists() and (p / 'sample_records.pkl').exists()
    assert not np.isnan(x['price']).any()


def test_time_split_has_both_sides(demo_paths):
    from dsin.training import split_indices
    ts = pd.read_pickle(demo_paths.processed / 'dsin_time_stamp.pkl')
    tr, va, te = split_indices(ts, 0.1)
    assert len(tr) and len(va) and len(te)
    assert ts[tr].max() <= ts[va].min() and ts[te].min() >= C.SPLIT_TS
