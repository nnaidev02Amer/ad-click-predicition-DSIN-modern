"""DSIN preprocessing pipeline (modernised port of 0_gen_sampled_data / 1_gen_sessions / 2_gen_dsin_input).

STEP 0  gen_sampled_data   : sample users, keep 'pv' behaviours, label-encode category/brand
STEP 1  gen_sessions       : split every user's behaviour history into sessions (gap > 30 min)
STEP 2  gen_dsin_input     : build, for every ad impression, the (<=5 sessions x <=10 behaviours) history
                             that existed *before* the impression, plus profile/ad features

Differences from the original code (all documented in README-MODERN.md):
  * file names are consistent (the original wrote `*_0.25.pkl` but looked for `*_0.25_.pkl`);
  * the behaviour log is read in chunks and filtered while reading (the original loaded all of it);
  * the session builder never uses sessions that start *after* the impression time (the original fell back
    to `len(sessions)-1` when no valid session existed, which leaks future behaviour);
  * vocabulary sizes of the shared cate_id/brand embeddings cover the behaviour-log encoders, not only the ads.
"""
from __future__ import annotations

import bisect
import ctypes
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from . import config as C
from .config import Paths

log = logging.getLogger('dsin.preprocessing')


# ----------------------------------------------------------------------------------- helpers
def available_ram_gb() -> float:
    """Free physical RAM in GB (Windows via ctypes, otherwise /proc/meminfo)."""
    try:
        class MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('sullAvailExtendedVirtual', ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullAvailPhys / 2 ** 30
    except Exception:  # pragma: no cover
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable'):
                return int(line.split()[1]) / 2 ** 20
    return float('nan')


def estimate_input_gb(n_samples: int, sess_count=C.DSIN_SESS_COUNT, sess_len=C.DSIN_SESS_MAX_LEN) -> float:
    """Approximate size of the model-input arrays (int32 session ids + scalar features)."""
    per_row = (sess_count * sess_len * len(C.SESS_FEATURES) + len(C.SPARSE_FEATURES) + 3) * 4
    return n_samples * per_row / 2 ** 30


def _require(paths_: list):
    missing = [str(p) for p in paths_ if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            'Missing required file(s): ' + ', '.join(missing) +
            '. Run the previous preprocessing step first (or place the dataset in raw_data/ - see README-MODERN.md).')


# ----------------------------------------------------------------------------------- STEP 0
def _read_behavior_pv(path: Path, userset: set, chunksize=5_000_000) -> pd.DataFrame:
    parts = []
    usecols = ['user', 'time_stamp', 'btag', 'cate', 'brand']
    for chunk in tqdm(pd.read_csv(path, usecols=usecols, chunksize=chunksize), desc='reading behaviour log',
                      unit='chunk'):
        chunk = chunk.loc[(chunk['btag'] == 'pv') & chunk['user'].isin(userset) & (chunk['time_stamp'] > 0)]
        parts.append(chunk.drop(columns=['btag']))
    if not parts:
        return pd.DataFrame(columns=['user', 'time_stamp', 'cate', 'brand'])
    return pd.concat(parts, ignore_index=True)


def gen_sampled_data(paths: Paths, frac: float = C.FRAC, force=False, chunksize=5_000_000):
    paths.ensure()
    out = paths.processed
    done = [out / n for n in ('user_profile_sub.pkl', 'raw_sample_sub.pkl', 'ad_feature_enc.pkl',
                              'behavior_log_pv_enc.pkl', 'cate_brand_encoders.pkl')]
    if not force and all(p.exists() for p in done):
        log.info('step 0 already done (use --force to redo)')
        return
    beh_path = paths.behavior_log()
    if beh_path is None:
        raise FileNotFoundError(f'No behaviour log found in {paths.raw}; expected one of {C.BEHAVIOR_LOG_NAMES}')
    _require([paths.raw / 'user_profile.csv', paths.raw / 'raw_sample.csv', paths.raw / 'ad_feature.csv'])

    user = pd.read_csv(paths.raw / 'user_profile.csv')
    user.columns = [c.strip() for c in user.columns]       # 'new_user_class_level ' has a trailing space
    sample = pd.read_csv(paths.raw / 'raw_sample.csv')
    log.info('users=%d samples=%d', len(user), len(sample))

    user_sub = user.sample(frac=frac, random_state=1024) if frac < 1.0 else user
    sample_sub = sample.loc[sample['user'].isin(user_sub['userid'].unique())].copy()
    userset = set(sample_sub['user'].unique())
    if sample_sub.empty:
        raise ValueError('No raw_sample rows belong to the sampled users; check the data and FRAC.')

    beh = _read_behavior_pv(beh_path, userset, chunksize)
    log.info('pv behaviour rows kept: %d', len(beh))

    ad = pd.read_csv(paths.raw / 'ad_feature.csv')
    ad['brand'] = ad['brand'].fillna(-1)
    beh['brand'] = beh['brand'].fillna(-1)

    cate_enc = LabelEncoder().fit(np.concatenate((ad['cate_id'].unique(), beh['cate'].unique())))
    ad['cate_id'] = cate_enc.transform(ad['cate_id']) + 1
    beh['cate'] = cate_enc.transform(beh['cate']) + 1
    brand_enc = LabelEncoder().fit(np.concatenate((ad['brand'].unique(), beh['brand'].unique())))
    ad['brand'] = brand_enc.transform(ad['brand']) + 1
    beh['brand'] = brand_enc.transform(beh['brand']) + 1

    pd.to_pickle(user_sub, out / 'user_profile_sub.pkl')
    pd.to_pickle(sample_sub, out / 'raw_sample_sub.pkl')
    pd.to_pickle(ad, out / 'ad_feature_enc.pkl')
    pd.to_pickle(beh, out / 'behavior_log_pv_enc.pkl')
    pd.to_pickle({'cate_classes': cate_enc.classes_, 'brand_classes': brand_enc.classes_},
                 out / 'cate_brand_encoders.pkl')
    log.info('step 0 done')


# ----------------------------------------------------------------------------------- STEP 1
def session_list_dsin(ts: np.ndarray, cate: np.ndarray, brand: np.ndarray):
    """Sessions for ONE user (arrays sorted by time). Same rules as the original gen_session_list_dsin."""
    last_time = C.FIRST_LAST_TIME
    sessions, cur = [], []
    for t, c, b in zip(ts.tolist(), cate.tolist(), brand.tolist()):
        if t - last_time > C.SESSION_GAP_SECONDS:
            if len(cur) > 2:          # only sessions with > 2 behaviours are kept
                sessions.append(cur)
            cur = []
        cur.append((c, b, t))
        last_time = t
    if len(cur) > 2:
        sessions.append(cur)
    return sessions


def gen_sessions(paths: Paths, force=False, batch_users=150_000):
    out_file = paths.processed / 'user_hist_session_dsin.pkl'
    if out_file.exists() and not force:
        log.info('step 1 already done (use --force to redo)')
        return
    _require([paths.processed / 'behavior_log_pv_enc.pkl', paths.processed / 'user_profile_sub.pkl'])
    data = pd.read_pickle(paths.processed / 'behavior_log_pv_enc.pkl')
    data = data.loc[data['time_stamp'] >= C.BEHAVIOR_START_TS].sort_values(['user', 'time_stamp'], kind='stable')
    users = data['user'].to_numpy()
    ts, cate, brand = data['time_stamp'].to_numpy(), data['cate'].to_numpy(), data['brand'].to_numpy()
    bounds = np.flatnonzero(np.r_[True, users[1:] != users[:-1], True])
    result = {}
    for s, e in tqdm(zip(bounds[:-1], bounds[1:]), total=len(bounds) - 1, desc='sessions', unit='user'):
        sess = session_list_dsin(ts[s:e], cate[s:e], brand[s:e])
        result[int(users[s])] = sess
    pd.to_pickle(result, out_file)
    n_sess = sum(len(v) for v in result.values())
    log.info('step 1 done: %d users, %d sessions (%d users have no session with >2 behaviours)',
             len(result), n_sess, sum(1 for v in result.values() if not v))


# ----------------------------------------------------------------------------------- STEP 2
def gen_sess_feature_dsin(sessions, time_stamp, sess_count=C.DSIN_SESS_COUNT, sess_max_len=C.DSIN_SESS_MAX_LEN):
    """History of one impression. Returns (cates[sess_count], brands[sess_count], lengths[sess_count], n_valid).

    sess_0 : the most recent session that started before `time_stamp` and has > 2 behaviours before it
             (truncated at `time_stamp`, last `sess_max_len` behaviours);
    sess_i : the i-th previous complete session (last `sess_max_len` behaviours).
    Missing sessions are [0] with length 0 (padding index 0 is masked inside the model).
    """
    cates = [[0] for _ in range(sess_count)]
    brands = [[0] for _ in range(sess_count)]
    lens = [0] * sess_count
    if not sessions:
        return cates, brands, lens, 0
    starts = [s[0][2] for s in sessions]
    k = bisect.bisect_left(starts, time_stamp) - 1          # last session starting before time_stamp
    base = None
    while k >= 0:
        cur = sessions[k]
        n_in = sum(1 for e in cur if e[2] < time_stamp)
        if n_in > 2:
            sel = cur[max(0, n_in - sess_max_len):n_in]
            cates[0] = [e[0] for e in sel]
            brands[0] = [e[1] for e in sel]
            lens[0] = min(sess_max_len, n_in)
            base = k
            break
        k -= 1
    if base is None:
        return cates, brands, lens, 0
    valid = 1
    for i in range(1, sess_count):
        if base - i >= 0:
            cur = sessions[base - i][-sess_max_len:]
            cates[i] = [e[0] for e in cur]
            brands[i] = [e[1] for e in cur]
            lens[i] = len(cur)
            valid += 1
    return cates, brands, lens, valid


def _pad(list_of_lists, maxlen):
    arr = np.zeros((len(list_of_lists), maxlen), dtype=np.int32)
    for i, seq in enumerate(list_of_lists):
        seq = seq[:maxlen]
        arr[i, :len(seq)] = seq            # padding='post'
    return arr


def gen_dsin_input(paths: Paths, sess_count=C.DSIN_SESS_COUNT, sess_max_len=C.DSIN_SESS_MAX_LEN,
                   force=False, max_ram_fraction=0.7):
    out = paths.processed
    if (out / 'dsin_input.pkl').exists() and not force:
        log.info('step 2 already done (use --force to redo)')
        return
    _require([out / n for n in ('user_hist_session_dsin.pkl', 'raw_sample_sub.pkl', 'user_profile_sub.pkl',
                                'ad_feature_enc.pkl', 'cate_brand_encoders.pkl')])
    user_hist = pd.read_pickle(out / 'user_hist_session_dsin.pkl')
    sample_sub = pd.read_pickle(out / 'raw_sample_sub.pkl').reset_index(drop=True)
    n = len(sample_sub)
    need, have = estimate_input_gb(n, sess_count, sess_max_len), available_ram_gb()
    log.info('%d impressions; model input needs ~%.2f GB, free RAM %.2f GB', n, need, have)
    if need > have * max_ram_fraction:
        raise MemoryError(
            f'Model input would need ~{need:.1f} GB but only {have:.1f} GB RAM is free. '
            f'Lower FRAC (e.g. --frac 0.05) and re-run from step 0.')

    cate_s = np.zeros((n, sess_count, sess_max_len), dtype=np.int32)
    brand_s = np.zeros((n, sess_count, sess_max_len), dtype=np.int32)
    len_s = np.zeros((n, sess_count), dtype=np.int32)
    sess_length = np.zeros(n, dtype=np.int32)
    for r, (u, t) in enumerate(tqdm(zip(sample_sub['user'].to_numpy(), sample_sub['time_stamp'].to_numpy()),
                                    total=n, desc='session features', unit='row')):
        c, b, l, v = gen_sess_feature_dsin(user_hist.get(int(u), []), int(t), sess_count, sess_max_len)
        for i in range(sess_count):
            cate_s[r, i, :len(c[i])] = c[i][:sess_max_len]
            brand_s[r, i, :len(b[i])] = b[i][:sess_max_len]
        len_s[r] = l
        sess_length[r] = v

    user = pd.read_pickle(out / 'user_profile_sub.pkl')
    user.columns = [c.strip() for c in user.columns]
    user = user.fillna(-1)
    ad = pd.read_pickle(out / 'ad_feature_enc.pkl')
    enc = pd.read_pickle(out / 'cate_brand_encoders.pkl')

    sample_sub = sample_sub.rename(columns={'user': 'userid'})
    data = sample_sub.merge(user, how='left', on='userid').merge(ad, how='left', on='adgroup_id')
    if data[C.SPARSE_FEATURES + C.DENSE_FEATURES].isna().any().any():
        bad = data[C.SPARSE_FEATURES + C.DENSE_FEATURES].isna().sum()
        raise ValueError(f'Join produced missing values (impressions with unknown user/ad?):\n{bad[bad > 0]}')
    data = data.reset_index(drop=True)

    # ---- statistics for the dashboard's Data Explorer (raw values, before encoding; no user ids) -----
    explorer = data.drop(columns=['userid']).copy()
    explorer['sess_length'] = sess_length
    explorer['cate_raw'] = enc['cate_classes'][explorer['cate_id'].to_numpy() - 1]
    explorer['brand_raw'] = enc['brand_classes'][explorer['brand'].to_numpy() - 1]
    if len(explorer) > 50000:
        explorer = explorer.sample(50000, random_state=0)
    explorer.to_pickle(out / 'explorer_sample.pkl')
    stats = {
        'n_impressions': int(n), 'n_users': int(data['userid'].nunique()), 'n_ads': int(data['adgroup_id'].nunique()),
        'click_rate': float(data['clk'].mean()), 'clk_counts': {str(k): int(v) for k, v in data['clk'].value_counts().items()},
        'time_min': int(data['time_stamp'].min()), 'time_max': int(data['time_stamp'].max()),
        'users_with_no_history': int((sess_length == 0).sum()),
        'sess_length_hist': {str(k): int(v) for k, v in pd.Series(sess_length).value_counts().sort_index().items()},
        'missing_after_join': {k: int(v) for k, v in data.isna().sum().items()},
        'duplicates_in_samples': int(data.duplicated().sum()),
        'profile_missing_filled_with_-1': {c: int((user[c] == -1).sum()) for c in C.USER_PROFILE_FEATURES},
    }
    (out / 'data_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')

    # ---- encode ---------------------------------------------------------------------------------
    classes = {}
    for feat in tqdm(C.SPARSE_FEATURES, desc='label-encoding'):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        classes[feat] = lbe.classes_
    scaler = StandardScaler()
    data[C.DENSE_FEATURES] = scaler.fit_transform(data[C.DENSE_FEATURES])

    vocab = {f: int(data[f].max()) + C.ID_OFFSET for f in C.SPARSE_FEATURES}
    vocab['cate_id'] = len(enc['cate_classes']) + 1 + C.ID_OFFSET
    vocab['brand'] = len(enc['brand_classes']) + 1 + C.ID_OFFSET

    model_input = {f: data[f].to_numpy().astype(np.int32) for f in C.SPARSE_FEATURES + C.SESS_FEATURES}
    model_input['price'] = data['price'].to_numpy().astype(np.float32)
    for i in range(sess_count):
        for feat, arr in (('cate_id', cate_s), ('brand', brand_s)):
            model_input[f'sess_{i}_{feat}'] = arr[:, i, :]
    model_input['sess_length'] = sess_length.astype(np.int32)
    label = data['clk'].to_numpy().astype(np.int32)
    time_stamp = data['time_stamp'].to_numpy()

    # ---- lookup tables for the dashboard ----------------------------------------------------------
    ad_tab = ad[['adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'price']].copy()
    ad_tab['cate_raw'] = enc['cate_classes'][ad_tab['cate_id'].to_numpy() - 1]
    ad_tab['brand_raw'] = enc['brand_classes'][ad_tab['brand'].to_numpy() - 1]
    ad_tab = ad_tab[ad_tab['adgroup_id'].isin(classes['adgroup_id'])]
    adg_idx = {int(v): i for i, v in enumerate(classes['adgroup_id'])}
    camp_idx = {int(v): i for i, v in enumerate(classes['campaign_id'])}
    cust_idx = {int(v): i for i, v in enumerate(classes['customer'])}
    ad_tab['adgroup_enc'] = ad_tab['adgroup_id'].map(adg_idx)
    ad_tab['campaign_enc'] = ad_tab['campaign_id'].map(camp_idx)
    ad_tab['customer_enc'] = ad_tab['customer'].map(cust_idx)
    ad_tab['price_scaled'] = (ad_tab['price'] - scaler.mean_[0]) / scaler.scale_[0]

    meta = {
        'sess_count': sess_count, 'sess_max_len': sess_max_len, 'frac': None,
        'vocab_sizes': vocab, 'dense_features': C.DENSE_FEATURES, 'sparse_features': C.SPARSE_FEATURES,
        'sess_features': C.SESS_FEATURES, 'price_mean': float(scaler.mean_[0]), 'price_scale': float(scaler.scale_[0]),
        'split_ts': C.SPLIT_TS, 'n_samples': int(n),
        'profile_choices': {f: [float(x) for x in classes[f]] for f in C.USER_PROFILE_FEATURES},
        'pid_choices': [str(x) for x in classes['pid']],
        'n_cate': int(len(enc['cate_classes'])), 'n_brand': int(len(enc['brand_classes'])),
    }
    pd.to_pickle({'ads': ad_tab.reset_index(drop=True),
                  'cate_classes': enc['cate_classes'], 'brand_classes': enc['brand_classes']}, out / 'lookup.pkl')

    # a small held-out sample of ready-to-use records for the dashboard
    test_idx = np.flatnonzero(time_stamp >= C.SPLIT_TS)
    rng = np.random.default_rng(0)
    pick = rng.choice(test_idx, min(500, len(test_idx)), replace=False) if len(test_idx) else np.array([], int)
    pd.to_pickle({'inputs': {k: v[pick] for k, v in model_input.items()}, 'label': label[pick]},
                 out / 'sample_records.pkl')

    (out / 'schema.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    pd.to_pickle(model_input, out / 'dsin_input.pkl')
    pd.to_pickle(label, out / 'dsin_label.pkl')
    pd.to_pickle(time_stamp, out / 'dsin_time_stamp.pkl')
    log.info('step 2 done: %d rows; train=%d test=%d', n, int((time_stamp < C.SPLIT_TS).sum()), len(test_idx))


def run_all(paths: Paths, frac=C.FRAC, force=False, chunksize=5_000_000):
    gen_sampled_data(paths, frac, force, chunksize)
    gen_sessions(paths, force)
    gen_dsin_input(paths, force=force)
    # record FRAC in the schema
    sp = paths.processed / 'schema.json'
    meta = json.loads(sp.read_text())
    meta['frac'] = frac
    sp.write_text(json.dumps(meta, indent=2))
