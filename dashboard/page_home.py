import streamlit as st

from dsin.config import Paths
from . import common


def render(mode, sts):
    p = Paths(mode)
    st.title('DSIN – Ad Click Prediction')
    st.caption('Deep Session Interest Network for Click-Through Rate (CTR) prediction')
    common.mode_banner(mode)

    cfg = common.read_json(p.models / 'model_config.json')
    env = common.env_info()
    c = st.columns(4)
    common.kpi(c[0], 'Model', 'DSIN', 'trained and loaded' if sts['trained'] else 'not trained yet')
    common.kpi(c[1], 'Training data', 'Synthetic' if mode == 'demo' else 'Taobao', 'demo records, not real users' if mode == 'demo' else 'real ad logs')
    common.kpi(c[2], 'Training records', f"{cfg['training']['n_train']:,}" if cfg else '—', 'ad impressions')
    common.kpi(c[3], 'Python / TensorFlow', f"{env['python']} / {env['tensorflow']}", 'CPU only')
    st.write('')

    a, b = st.columns(2)
    with a:
        st.subheader('Introduction')
        st.write('Online advertising systems must decide which advertisement to show to which user. '
                 'A key quantity is the **click-through rate (CTR)** – the probability that a user clicks an ad. '
                 'This project implements **DSIN (Deep Session Interest Network)**, a deep-learning model that predicts '
                 'this probability from the user\'s recent browsing history and the advertisement.')
        st.subheader('Main objective')
        st.write('Predict how likely a user is to click a given advertisement, by learning the user\'s interests from '
                 'their browsing **sessions**.')
    with b:
        st.subheader('How the system works')
        st.markdown("""
1. **Input** – user profile, browsing history (grouped into sessions), and the advertisement.
2. **Preprocessing** – ids are encoded as numbers; history is padded to 5 sessions × 10 items.
3. **DSIN model** – finds the user's interest in each session and how it relates to the ad.
4. **Output** – a click probability between 0% and 100%.
""")

    st.subheader('Workflow')
    st.graphviz_chart('''digraph {
        rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#4f46e5", fontname="Helvetica"];
        edge [color="#64748b"];
        "User profile\\n+ browsing history" -> "Preprocessing\\n(encode, sessions, padding)" -> "DSIN model\\n(TensorFlow)" -> "Click probability" -> "Result page";
        "Advertisement\\n(category, brand, price)" -> "Preprocessing\\n(encode, sessions, padding)";
    }''')

    st.subheader('Technology stack')
    t = st.columns(5)
    for col, (name, what) in zip(t, [('Python 3.9', 'language'), ('TensorFlow 2.15', 'deep learning'),
                                     ('DeepCTR 0.9.4', 'DSIN layers'), ('Pandas / scikit-learn', 'data processing'),
                                     ('Streamlit + Plotly', 'web interface')]):
        common.kpi(col, what, name)
    if not sts['trained']:
        st.error('No trained model found. Run: `python run_preprocessing.py --mode demo --make-demo` and '
                 '`python train_model.py --mode demo`.')
