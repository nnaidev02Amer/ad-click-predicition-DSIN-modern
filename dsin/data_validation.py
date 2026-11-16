"""Validation of the raw Taobao CSV files (schema, types, missing values, duplicates, labels)."""
from __future__ import annotations

import json

import pandas as pd

from .config import Paths, REQUIRED_RAW


def _header(path):
    return [c.strip() for c in pd.read_csv(path, nrows=0).columns]


def check_files(paths: Paths) -> dict:
    """Cheap check: do the files exist and do their headers match the expected schema?"""
    report = {'raw_dir': str(paths.raw), 'files': {}, 'missing': [], 'ok': True}
    for name, cols in REQUIRED_RAW.items():
        if name == 'behavior_log':
            f = paths.behavior_log()
            label = 'behavior_log.csv (or raw_behavior_log.csv)'
        else:
            f = paths.raw / name
            f = f if f.exists() else None
            label = name
        if f is None:
            report['missing'].append(label)
            report['files'][label] = {'exists': False}
            report['ok'] = False
            continue
        header = _header(f)
        missing_cols = [c for c in cols if c not in header]
        report['files'][label] = {'exists': True, 'path': str(f), 'size_mb': round(f.stat().st_size / 2**20, 2),
                                  'columns': header, 'missing_columns': missing_cols}
        if missing_cols:
            report['ok'] = False
    return report


def validate_dataset(paths: Paths, max_rows_behavior: int = 2_000_000) -> dict:
    """Deeper validation. The (huge) behaviour log is only inspected on its first rows."""
    rep = check_files(paths)
    if not rep['ok']:
        return rep
    stats = {}
    sample = pd.read_csv(paths.raw / 'raw_sample.csv')
    stats['raw_sample'] = {
        'rows': len(sample), 'duplicates': int(sample.duplicated().sum()),
        'missing': sample.isna().sum().to_dict(),
        'clk_distribution': {str(k): int(v) for k, v in sample['clk'].value_counts().items()},
        'clk_values_valid': bool(sample['clk'].isin([0, 1]).all()),
        'users': int(sample['user'].nunique()), 'ads': int(sample['adgroup_id'].nunique()),
        'time_min': int(sample['time_stamp'].min()), 'time_max': int(sample['time_stamp'].max()),
        'dtypes': {k: str(v) for k, v in sample.dtypes.items()}}
    ad = pd.read_csv(paths.raw / 'ad_feature.csv')
    stats['ad_feature'] = {'rows': len(ad), 'duplicate_adgroup_id': int(ad['adgroup_id'].duplicated().sum()),
                           'missing': ad.isna().sum().to_dict(),
                           'sample_ads_without_feature': int((~sample['adgroup_id'].isin(ad['adgroup_id'])).sum())}
    user = pd.read_csv(paths.raw / 'user_profile.csv')
    user.columns = [c.strip() for c in user.columns]
    stats['user_profile'] = {'rows': len(user), 'duplicate_userid': int(user['userid'].duplicated().sum()),
                             'missing': user.isna().sum().to_dict(),
                             'sample_users_without_profile': int((~sample['user'].isin(user['userid'])).sum())}
    beh = pd.read_csv(paths.behavior_log(), nrows=max_rows_behavior)
    stats['behavior_log_head'] = {'rows_inspected': len(beh), 'btag_counts': beh['btag'].value_counts().to_dict(),
                                  'missing': beh.isna().sum().to_dict()}
    rep['stats'] = stats
    problems = []
    if not stats['raw_sample']['clk_values_valid']:
        problems.append('clk contains values other than 0/1')
    if stats['ad_feature']['sample_ads_without_feature']:
        problems.append('some samples reference ads missing from ad_feature.csv')
    if stats['user_profile']['sample_users_without_profile']:
        problems.append('some samples reference users missing from user_profile.csv')
    rep['problems'] = problems
    rep['ok'] = not problems
    return rep


def save_report(rep: dict, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=2, default=str), encoding='utf-8')
