"""Shared helpers for the Streamlit dashboard: styling, artifact loading, KPI cards."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import pandas as pd
import streamlit as st

from dsin.config import MODES, Paths

THEME_FILE = Path(__file__).with_name("theme.css")

PLOT_COLORS = {'primary': '#4f46e5', 'secondary': '#0d9488', 'neutral': '#a5b4fc'}


def style():
    st.markdown('<style>' + THEME_FILE.read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)


def kpi(col, label, value, sub=''):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


DEMO_TEXT = ('DEMO MODE – SYNTHETIC DATA. This educational demonstration uses generated sample records. '
             'It is not Taobao data and does not establish real-world DSIN performance.')


def mode_banner(mode):
    if mode == 'demo':
        st.markdown(f'<div class="banner-demo">⚠ {DEMO_TEXT}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="banner-real">Real mode – original Taobao dataset</div>', unsafe_allow_html=True)


def status(mode) -> dict:
    p = Paths(mode)
    from dsin.data_validation import check_files
    raw = check_files(p)
    return {'raw_ok': raw['ok'], 'raw_missing': raw['missing'], 'raw_dir': str(p.raw),
            'processed': (p.processed / 'dsin_input.pkl').exists(),
            'trained': (p.models / 'dsin.weights.h5').exists() and (p.models / 'model_config.json').exists(),
            'evaluated': (p.results / 'evaluation.json').exists()}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


@st.cache_resource(show_spinner='Loading DSIN model...')
def get_predictor(mode, _mtime):
    from dsin.predict import Predictor
    return Predictor(Paths(mode))


def predictor_for(mode):
    p = Paths(mode)
    f = p.models / 'dsin.weights.h5'
    return get_predictor(mode, f.stat().st_mtime if f.exists() else 0)


@st.cache_data(show_spinner=False)
def load_pickle(path, _mtime):
    return pd.read_pickle(path)


def verdict(ctr, baseline):
    """(label, comparison text). Compares with the average click rate of the training data when known."""
    if baseline:
        ratio = ctr / baseline
        if ratio >= 1.5:
            return 'Higher predicted click likelihood', f'{ratio:.1f}× the average click rate ({baseline:.1%}) of the training data'
        if ratio <= 0.7:
            return 'Lower predicted click likelihood', f'{ratio:.1f}× the average click rate ({baseline:.1%}) of the training data'
        return 'About average predicted click likelihood', f'close to the average click rate ({baseline:.1%}) of the training data'
    if ctr >= 0.5:
        return 'Higher predicted click likelihood', 'above 50%'
    return 'Lower predicted click likelihood', 'below 50%'


def env_info():
    import tensorflow as tf
    try:
        import deepctr
        dv = deepctr.__version__
    except Exception:
        dv = 'n/a'
    return {'python': platform.python_version(), 'tensorflow': tf.__version__, 'deepctr': dv}


def auto_mode():
    """Use the real-data model if one has been trained, otherwise the synthetic demo model."""
    real = status('real')
    if real['trained']:
        return 'real', real
    return 'demo', status('demo')
