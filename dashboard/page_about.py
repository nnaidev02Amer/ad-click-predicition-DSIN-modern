import streamlit as st

from . import common


def render(mode, sts):
    st.title('About DSIN')
    common.mode_banner(mode)

    st.subheader('What is CTR prediction?')
    st.write('**Click-through rate (CTR)** is the fraction of times an advertisement is clicked after being shown. '
             'CTR prediction estimates this probability for a particular user and ad, so that a platform can show '
             'the ads people are most likely to find relevant.')

    st.subheader('What is DSIN?')
    st.write('**Deep Session Interest Network (DSIN)** is a deep-learning CTR model by Feng et al. (IJCAI 2019, Alibaba). '
             'Instead of treating a user\'s history as one long list, DSIN splits it into **sessions** and models the '
             'interest in each session and how interests evolve between sessions.')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Why does browsing history matter?')
        st.write('What a user looked at recently is a strong hint about what they want now. Someone who browsed running '
                 'shoes this morning is more likely to click a sports-shoe ad than a random user.')
        st.subheader('What is a "session interest"?')
        st.write('Browsing happens in bursts: a person opens the app, looks at a few products, and leaves. Within one burst '
                 '(here: a gap of at most 30 minutes) the interest is usually **focused on one theme**. '
                 'A session interest is a numeric summary of that theme.')
    with c2:
        st.subheader('How does attention work?')
        st.write('Attention lets the model give **different importance to different items**. '
                 '*Self-attention* inside a session lets items in a visit influence each other so that the summary focuses on the '
                 'main theme. *Target attention* then compares each session interest with the advertisement and '
                 'gives more weight to the sessions that are relevant to that ad.')

    st.subheader('How DSIN predicts the click probability')
    st.markdown("""
1. **Session division layer** – splits history into sessions (a new session starts after a 30-minute gap).
2. **Session interest extractor** – multi-head self-attention (with bias encoding / positional encoding) turns each session into one interest vector.
3. **Session interest interacting layer** – a Bi-LSTM reads the sequence of session interests to capture how interests evolve.
4. **Session interest activating layer** – attention between the **advertisement** and each session interest (before and after the Bi-LSTM).
5. **Output layer** – the activated interests are joined with user profile, ad features and price, passed through a neural network (200 → 80 units) and a sigmoid, giving a probability between 0 and 1.
""")
    st.graphviz_chart('''digraph {
        rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#4f46e5", fontname="Helvetica"]; edge [color="#64748b"];
        hist [label="Browsing history"]; sess [label="Session division\\n(30-min gap)"];
        ext [label="Session interest extractor\\n(multi-head self-attention)"]; lstm [label="Interest interacting layer\\n(Bi-LSTM)"];
        act [label="Interest activating layers\\n(attention with the ad)"]; ad [label="Advertisement\\nfeatures", fillcolor="#cffafe", color="#0d9488"];
        prof [label="User profile", fillcolor="#cffafe", color="#0d9488"];
        dnn [label="Concatenate + DNN\\n(200, 80)"]; out [label="Click probability\\n(sigmoid)", fillcolor="#ede9fe", color="#7c3aed"];
        hist -> sess -> ext -> lstm -> act; ext -> act; ad -> act; act -> dnn; ad -> dnn; prof -> dnn; dnn -> out;
    }''')

    st.subheader('About this demonstration')
    st.info('This demonstration uses a small **synthetic** dataset with the same structure as the Taobao ad dataset used in the paper. '
            'The DSIN network and the data pipeline are complete, but the model was trained only on synthetic records, '
            'so the probabilities are for demonstration. Training on the original Taobao data is a matter of placing the dataset '
            'files in `raw_data/` and running the same scripts.')
   
