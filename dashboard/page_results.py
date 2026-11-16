import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dsin import history
from . import common


def render(mode, sts):
    st.title('Prediction Results')
    common.mode_banner(mode)
    r = st.session_state.get('last')
    if not r:
        st.info('No prediction yet. Go to **Predict Advertisement Click**, fill the form and press Predict.')
    else:
        ctr, base = r['ctr'], r.get('baseline')
        label, cmp_text = common.verdict(ctr, base)
        if r['synthetic']:
            st.warning('DEMO MODE – SYNTHETIC DATA: this probability comes from a model trained on fake records. '
                       'It demonstrates how the system works; it is not a real-world prediction.')
        left, right = st.columns([1, 1])
        with left:
            tone = {'Higher': 'high', 'Lower': 'low'}.get(label.split()[0], 'mid')
            st.markdown(f'<div class="result tone-{tone}"><div style="color:#64748b;font-weight:700;letter-spacing:.05em">PREDICTED CLICK PROBABILITY</div>'
                        f'<div class="big">{ctr:.1%}</div><span class="pill pill-{tone}">{label}</span>'
                        f'<div style="color:#64748b;margin-top:8px">{cmp_text}</div></div>', unsafe_allow_html=True)
            st.write(f'**What this means:** out of 100 times an advertisement like this is shown to users like this, '
                     f'the model expects about **{ctr * 100:.0f}** clicks.')
            st.caption('This is an **estimate**, not a guaranteed click. '
                       + ('This is a demonstration using synthetic data.' if r['synthetic'] else ''))
        with right:
            fig = go.Figure(go.Indicator(mode='gauge+number', value=ctr * 100, number={'suffix': '%', 'valueformat': '.1f'},
                                         gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#4f46e5'},
                                                'steps': [{'range': [0, 33], 'color': '#ecfeff'}, {'range': [33, 66], 'color': '#e0e7ff'}, {'range': [66, 100], 'color': '#ede9fe'}],
                                                **({'threshold': {'line': {'color': '#0d9488', 'width': 4}, 'value': base * 100}} if base else {})}))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            if base:
                st.caption(f'Teal line = average click rate in the training data ({base:.1%}).')

        st.subheader('Input summary')
        c = st.columns(4)
        common.kpi(c[0], 'Advertisement', r['ad_cate_name'], r['ad_brand_name'])
        common.kpi(c[1], 'User', f"{r['gender_text']}, {r['age_text']}", 'age shown for information only')
        common.kpi(c[2], 'History', f"{r['n_items']} items", f"in {r['n_sessions']} visit(s)")
        common.kpi(c[3], 'Product price', f"{r['price']:.2f}", 'price value')
        with st.expander('Products the user previously viewed'):
            for i, sess in enumerate(r['sessions']):
                st.write(f"**{['Most Recent Visit', 'Previous Visit', 'Earlier Visit'][i]}:** " + '; '.join(r['sessions_text'][i]))

        st.subheader('Why this prediction?')
        match_txt = (f"{r['n_match']} of the {r['n_items']} viewed items belong to the advertised product category"
                     if r['n_match'] else 'none of the viewed items belong to the advertised product category')
        st.write(f"The DSIN model read the user's {r['n_sessions']} previous visit(s), extracted an interest "
                 f"representation for each session with self-attention, and measured how strongly those interests "
                 f"relate to the advertisement ({r['ad_cate_name']}, {r['ad_brand_name']}). "
                 f"For reference, {match_txt}. This is a simple description of the input, not an exact attribution of the model's decision.")
        st.caption('Age is recorded for display only: the demo model age feature uses a fixed default category because age in years '
                   'cannot be mapped to the dataset age codes.')
        st.caption(f"Model output: {ctr:.6f} · computed {r['time']}")

    st.subheader('Recent predictions')
    df = history.load()
    if df.empty:
        st.caption('Nothing stored yet. Predictions are saved only on this computer (results/prediction_history.sqlite).')
    else:
        df['source'] = df['synthetic'].map({1: 'DEMO (synthetic)', 0: 'real'})
        show = df.head(10).rename(columns={'ts': 'Time', 'summary': 'Input', 'ctr': 'Probability', 'result': 'Result', 'source': 'Model'})
        st.dataframe(show[['Time', 'Model', 'Input', 'Probability', 'Result']], hide_index=True, use_container_width=True,
                     column_config={'Probability': st.column_config.NumberColumn(format='%.3f')})
