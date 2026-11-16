"""DSIN - Ad Click Prediction demo.   Start with:  streamlit run app.py   ->  http://localhost:8501"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dsin.config import cpu_only  # noqa: E402

cpu_only()

import streamlit as st  # noqa: E402

st.set_page_config(page_title='DSIN Ad Click Prediction', page_icon='📈', layout='wide')

from dashboard import common, page_home, page_predict, page_results, page_about  # noqa: E402

PAGES = {
    'Home': page_home,
    'Predict Advertisement Click': page_predict,
    'Prediction Results': page_results,
    'About DSIN': page_about,
}

common.style()
st.sidebar.markdown('## 📈 DSIN')
st.sidebar.caption('Deep Session Interest Network\nAd click-through-rate prediction')
choice = st.sidebar.radio('Navigation', list(PAGES), key='nav', label_visibility='collapsed')
mode, sts = common.auto_mode()
st.sidebar.divider()
st.sidebar.caption('DEMO MODE – synthetic data' if mode == 'demo' else 'Model trained on real data')

PAGES[choice].render(mode, sts)
