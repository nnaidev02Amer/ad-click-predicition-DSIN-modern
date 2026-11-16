import time

import streamlit as st

from dsin import config as C
from dsin import history
from dsin.config import Paths
from dsin.labels import AGE_MAX, AGE_MIN, Labels, age_to_model_category, gender_label
from . import common

N_SESSIONS_UI = 3
VISIT_NAMES = ['Most Recent Visit', 'Previous Visit', 'Earlier Visit']
AGE_NOTE = ('Age is recorded for display. The demo model\'s age feature uses a fixed default category, because age in years '
            'cannot be mapped to the dataset\'s age codes (in the synthetic data they are random labels 1–6 with no age meaning).')


def _labels(pred) -> Labels:
    return Labels(pred.cate_classes, pred.brand_classes, pred.synthetic)


def _pairs(pred):
    """display label -> (raw category id, raw brand id) for every product (category, brand) combination in the data."""
    lb = _labels(pred)
    ads = pred.ads[['cate_raw', 'brand_raw']].drop_duplicates().sort_values(['cate_raw', 'brand_raw'])
    return {lb.pair_label(c, b): (float(c), float(b)) for c, b in ads.itertuples(index=False)}


def _ad_row(pred, ad_id):
    return pred.ads[pred.ads['adgroup_id'] == ad_id].iloc[0]


def _ad_options(pred):
    """adgroup_id -> readable product name (no raw ids); the numbering is the position in the sorted list."""
    lb = _labels(pred)
    ads = pred.ads.sort_values('adgroup_id').reset_index(drop=True)
    return {int(r.adgroup_id): f'{lb.cate_name(r.cate_raw)} – {lb.brand_name(r.brand_raw)} (Ad {i + 1})'
            for i, r in enumerate(ads.itertuples())}


def _sync_ad(pred):
    """When the advertisement changes, refill category, brand, price and the optional advanced fields from its record."""
    r = _ad_row(pred, st.session_state['ad_sel'])
    S = st.session_state
    S['ad_cate'], S['ad_brand'] = int(r['cate_raw']), int(r['brand_raw'])
    S['ad_price'] = round(float(r['price']), 2)
    S['adv_campaign'], S['adv_customer'] = int(r['campaign_id']), int(r['customer'])


def _load_sample(pred, kind):
    """kind 'match': history contains the advertised product category; 'other': history is about different categories."""
    ads = pred.ads.sort_values('adgroup_id').reset_index(drop=True)
    r = ads.iloc[len(ads) // 3]
    st.session_state['ad_sel'] = int(r['adgroup_id'])
    _sync_ad(pred)
    pairs = _pairs(pred)
    same = [l for l, (c, b) in pairs.items() if int(c) == int(r['cate_raw'])]
    other = [l for l, (c, b) in pairs.items() if int(c) != int(r['cate_raw'])]
    st.session_state['age_years'] = 28 if kind == 'match' else 45
    st.session_state['gender_code'] = pred.schema['profile_choices']['final_gender_code'][1 if kind == 'match' else 0]
    pool = same if kind == 'match' else other
    pool = pool or list(pairs)
    st.session_state['sess_0'] = pool[:4]
    st.session_state['sess_1'] = pool[:3] if kind == 'match' else other[4:7]
    st.session_state['sess_2'] = other[7:9] if kind == 'match' else other[9:12]
    st.session_state['pred_error'] = None


def _predict(mode):
    """Button callback: validate -> convert friendly choices to model ids -> DSIN -> store -> jump to Results."""
    try:
        pred = common.predictor_for(mode)
        s = st.session_state
        lb = _labels(pred)
        schema = pred.schema
        profile = {f: s['pf_' + f] for f in C.USER_PROFILE_FEATURES}
        profile['age_level'], _ = age_to_model_category(s['age_years'], schema['profile_choices']['age_level'])
        profile['final_gender_code'] = s['gender_code']
        pairs = _pairs(pred)
        sessions = [[pairs[l] for l in s[f'sess_{i}']] for i in range(N_SESSIONS_UI)]
        if not any(sessions):
            raise ValueError('Please select at least one product in "Most Recent Visit".')
        while sessions and not sessions[-1]:
            sessions.pop()
        sessions = [x for x in sessions if x]
        price = float(s['ad_price'])
        if not price > 0:
            raise ValueError('Product Price must be greater than 0.')
        row = _ad_row(pred, s['ad_sel']).to_dict()
        row['cate_id'], row['brand'] = pred.cate_enc(s['ad_cate']), pred.brand_enc(s['ad_brand'])
        row['price'], row['price_scaled'] = price, (price - schema['price_mean']) / schema['price_scale']
        camp = pred.ads.drop_duplicates('campaign_id').set_index('campaign_id')['campaign_enc']
        cust = pred.ads.drop_duplicates('customer').set_index('customer')['customer_enc']
        row['campaign_enc'], row['customer_enc'] = int(camp[s['adv_campaign']]), int(cust[s['adv_customer']])
        x = pred.build_inputs(profile, row, s['adv_pid'], sessions)
        ctr = pred.predict_one(x)
    except (ValueError, KeyError) as e:
        st.session_state['pred_error'] = str(e)
        return
    n_match = sum(1 for sess in sessions for c, _ in sess if int(c) == int(s['ad_cate']))
    n_items = sum(len(sess) for sess in sessions)
    stats = common.read_json(Paths(mode).processed / 'data_stats.json') or {}
    res = {'ctr': ctr, 'mode': mode, 'synthetic': pred.synthetic, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
           'ad_id': int(s['ad_sel']), 'ad_cate': int(s['ad_cate']), 'ad_brand': int(s['ad_brand']),
           'ad_cate_name': lb.cate_name(s['ad_cate']), 'ad_brand_name': lb.brand_name(s['ad_brand']),
           'price': price, 'pid': s['adv_pid'], 'profile': profile,
           'age_years': int(s['age_years']), 'gender': profile['final_gender_code'],
           'age_text': f"{int(s['age_years'])} years", 'gender_text': gender_label(profile['final_gender_code']),
           'n_sessions': len(sessions), 'n_items': n_items, 'n_match': n_match, 'baseline': stats.get('click_rate'),
           'sessions': [[(int(c), int(b)) for c, b in sess] for sess in sessions],
           'sessions_text': [[lb.pair_label(c, b) for c, b in sess] for sess in sessions]}
    st.session_state['last'] = res
    st.session_state['pred_error'] = None
    label = common.verdict(ctr, res['baseline'])[0]
    history.add(mode, pred.synthetic,
                f"{res['ad_cate_name']} · {res['ad_brand_name']} ad, age {res['age_years']}, {res['gender_text']}, "
                f"{n_items} viewed products in {len(sessions)} visit(s)", {k: v for k, v in res.items() if k != 'sessions_text'}, ctr, label)
    st.session_state['nav'] = 'Prediction Results'


def render(mode, sts):
    st.title('Predict Advertisement Click')
    common.mode_banner(mode)
    if not sts['trained']:
        st.error(f'Prediction is unavailable: no trained DSIN model for **{mode}** mode. '
                 f'Run `python run_preprocessing.py --mode {mode}` and `python train_model.py --mode {mode}` first.')
        return
    pred = common.predictor_for(mode)
    schema, ads, lb = pred.schema, pred.ads, _labels(pred)
    pairs = _pairs(pred)
    ad_opts = _ad_options(pred)
    S = st.session_state

    # defaults (set once, before the widgets are created)
    S.setdefault('ad_sel', next(iter(ad_opts)))
    if 'ad_cate' not in S:
        _sync_ad(pred)
    S.setdefault('age_years', 25)
    S.setdefault('gender_code', schema['profile_choices']['final_gender_code'][0])
    for f in C.USER_PROFILE_FEATURES:                # technical profile fields: first valid value of the data
        S.setdefault('pf_' + f, schema['profile_choices'][f][0])
    for i in range(N_SESSIONS_UI):
        S.setdefault(f'sess_{i}', [])
    S.setdefault('adv_pid', schema['pid_choices'][0])

    st.write('Fill in the form (or load a sample), then press **Predict Click Probability**. '
             'Your choices are converted to the DSIN model\'s input format and the trained model calculates the result.')
    b1, b2, _ = st.columns([1.3, 1.3, 3])
    b1.button('Load sample 1: interested in the ad topic', on_click=_load_sample, args=(pred, 'match'), use_container_width=True)
    b2.button('Load sample 2: unrelated interests', on_click=_load_sample, args=(pred, 'other'), use_container_width=True)
    if pred.synthetic:
        st.caption('Demo labels: product and brand names are illustrative labels attached to synthetic demo data. '
                   'The brands are fictional.')

    st.subheader('1. User Information')
    u1, u2 = st.columns(2)
    u1.number_input('Age', min_value=AGE_MIN, max_value=AGE_MAX, step=1, key='age_years')
    u2.selectbox('Gender', schema['profile_choices']['final_gender_code'], key='gender_code', format_func=gender_label)
    st.caption(AGE_NOTE)

    st.subheader('2. Products the User Previously Viewed')
    st.caption('Select products that the user viewed during previous visits to the shopping website '
               f'(up to {C.DSIN_SESS_MAX_LEN} products per visit).')
    sc = st.columns(N_SESSIONS_UI)
    for i in range(N_SESSIONS_UI):
        sc[i].multiselect(VISIT_NAMES[i], list(pairs), key=f'sess_{i}', max_selections=C.DSIN_SESS_MAX_LEN,
                          placeholder='Choose products')

    st.subheader('3. Advertisement Details')
    st.selectbox('Advertisement / Product Name', list(ad_opts), key='ad_sel', format_func=lambda i: ad_opts[i],
                 on_change=_sync_ad, args=(pred,),
                 help='Choosing an advertisement fills in the product category, brand and price. You can still change them.')
    a1, a2, a3 = st.columns(3)
    a1.selectbox('Product Category', pred.cate_classes.tolist(), key='ad_cate', format_func=lb.cate_name)
    a2.selectbox('Product Brand', pred.brand_classes.tolist(), key='ad_brand', format_func=lb.brand_name)
    a3.number_input('Product Price', min_value=0.01, step=1.0, key='ad_price', format='%.2f',
                    help=f"Price value (currency not specified in the dataset). Prices in the training data range from "
                         f"{ads['price'].min():.0f} to {ads['price'].max():.0f}.")
    if not ads['price'].min() * 0.999 <= S['ad_price'] <= ads['price'].max() * 1.001:
        st.caption('⚠ This price is outside the range seen in the training data, so the estimate is less reliable.')
    with st.expander('Advanced Details (Optional)'):
        st.caption('Filled automatically from the chosen advertisement. Change only if needed.')
        x1, x2, x3 = st.columns(3)
        x1.selectbox('Campaign ID', sorted(ads['campaign_id'].unique().tolist()), key='adv_campaign')
        x2.selectbox('Customer ID', sorted(ads['customer'].unique().tolist()), key='adv_customer')
        x3.selectbox('Ad Placement', schema['pid_choices'], key='adv_pid')

    st.button('Predict Click Probability', type='primary', on_click=_predict, args=(mode,))
    if S.get('pred_error'):
        st.error(f"Invalid input: {S['pred_error']}")
