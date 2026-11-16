"""Dashboard tests: Streamlit's AppTest runs app.py headlessly (uses the trained demo model in models/demo)."""
import pytest

from dsin import history

PAGES = ['Home', 'Predict Advertisement Click', 'Prediction Results', 'About DSIN']


def test_history_roundtrip(tmp_path):
    db = tmp_path / 'h.sqlite'
    history.add('demo', True, 'summary', {'a': 1}, 0.42, 'Moderate', path=db)
    df = history.load(db)
    assert len(df) == 1 and df['ctr'].iloc[0] == pytest.approx(0.42)


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file('app.py', default_timeout=180).run()


def test_app_starts_and_every_page_renders():
    at = _app()
    assert not at.exception, [e.value for e in at.exception]
    assert list(at.sidebar.radio[0].options) == PAGES          # exactly four sections
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
    assert any('DEMO MODE' in m.value for m in at.markdown)


def test_predict_flow_uses_real_model_and_shows_result(monkeypatch, tmp_path):
    monkeypatch.setattr(history, 'DB_PATH', tmp_path / 'h.sqlite')
    monkeypatch.setattr(history, '_conn', history._conn)
    at = _app()
    at.sidebar.radio[0].set_value('Predict Advertisement Click').run()
    next(b for b in at.button if b.label.startswith('Load sample 1')).click().run()
    next(b for b in at.button if b.label == 'Predict Click Probability').click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state['nav'] == 'Prediction Results'
    res = at.session_state['last']
    assert 0.0 <= res['ctr'] <= 1.0 and res['synthetic'] is True and res['n_items'] > 0
    assert any('PREDICTED CLICK PROBABILITY' in m.value for m in at.markdown)

    # same inputs through the model directly give the same number (not a random value)
    from dsin.config import Paths
    from dsin.predict import Predictor
    pred = Predictor(Paths('demo'))
    row = pred.ads[pred.ads['adgroup_id'] == res['ad_id']].iloc[0].to_dict()
    row['cate_id'], row['brand'] = pred.cate_enc(res['ad_cate']), pred.brand_enc(res['ad_brand'])
    x = pred.build_inputs(res['profile'], row, res['pid'], res['sessions'])
    assert pred.predict_one(x) == pytest.approx(res['ctr'], abs=1e-5)


def test_predict_requires_history():
    at = _app()
    at.sidebar.radio[0].set_value('Predict Advertisement Click').run()
    next(b for b in at.button if b.label == 'Predict Click Probability').click().run()
    assert at.session_state['pred_error'] and 'Most Recent Visit' in at.session_state['pred_error']
    assert any('Invalid input' in e.value for e in at.error)


def _predict_page(sample):
    at = _app()
    at.sidebar.radio[0].set_value('Predict Advertisement Click').run()
    if sample:
        next(b for b in at.button if b.label.startswith(sample)).click().run()
    return at


def test_form_fields_and_indian_demo_labels():
    at = _predict_page(None)
    sel = {w.label: w for w in at.selectbox}
    assert [w.label for w in at.number_input if w.label == 'Age'] == ['Age']
    age = next(w for w in at.number_input if w.label == 'Age')
    assert (age.min, age.max) == (13, 100)
    assert sel['Gender'].options == ['Male', 'Female']
    assert {'Most Recent Visit', 'Previous Visit', 'Earlier Visit'} <= {w.label for w in at.multiselect}
    assert {'Advertisement / Product Name', 'Product Category', 'Product Brand'} <= set(sel)
    assert any(w.label == 'Product Price' for w in at.number_input)
    assert any(b.label == 'Predict Click Probability' for b in at.button)
    assert any(b.label.startswith('Load sample 1') for b in at.button) and any(b.label.startswith('Load sample 2') for b in at.button)
    # Indian product examples, no raw ids
    cats = set(sel['Product Category'].options)
    assert {'Sarees', 'Kurtas', 'Smartphones', 'Sports Shoes'} <= cats
    assert not any(o.strip().isdigit() for o in sel['Product Category'].options + sel['Product Brand'].options)
    assert not any(o.strip().isdigit() for o in sel['Advertisement / Product Name'].options)
    # technical fields live only inside the optional expander
    assert any(e.label == 'Advanced Details (Optional)' for e in at.expander)
    adv = next(e for e in at.expander if e.label == 'Advanced Details (Optional)')
    assert {w.label for w in adv.selectbox} == {'Campaign ID', 'Customer ID', 'Ad Placement'}
    assert at.sidebar.radio[0].options == PAGES
    assert any('does not establish real-world DSIN performance' in m.value for m in at.markdown)


def test_age_mapping_is_fixed_and_documented():
    from dsin.labels import age_to_model_category
    choices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert {age_to_model_category(a, choices) for a in (13, 20, 45, 100)} == {(1.0, False)}   # no mapping can be established
    for bad in (12, 101):
        with pytest.raises(ValueError):
            age_to_model_category(bad, choices)


@pytest.mark.parametrize('sample', ['Load sample 1', 'Load sample 2'])
def test_both_samples_predict_and_friendly_choices_map_to_model_ids(sample, monkeypatch, tmp_path):
    monkeypatch.setattr(history, 'DB_PATH', tmp_path / 'h.sqlite')
    at = _predict_page(sample)
    assert not at.exception and at.session_state['sess_0']
    chosen = {i: list(at.session_state[f'sess_{i}']) for i in range(3)}
    next(b for b in at.button if b.label == 'Predict Click Probability').click().run()
    assert not at.exception, [e.value for e in at.exception]
    res = at.session_state['last']

    from dsin.config import Paths
    from dsin.labels import Labels
    from dsin.predict import Predictor
    pred = Predictor(Paths('demo'))
    lb = Labels(pred.cate_classes, pred.brand_classes, True)
    for i, labels in chosen.items():
        got = res['sessions'][i] if i < len(res['sessions']) else []
        want = [(lb.cate_rev[l.split(' · ')[0]], lb.brand_rev[l.split(' · ')[1]]) for l in labels]
        assert got == want
    assert (res['ad_cate'], res['ad_brand']) == (lb.cate_rev[res['ad_cate_name']], lb.brand_rev[res['ad_brand_name']])
    # age: typed age is kept for display; the model gets the fixed default age code; gender Male/Female -> codes 1/2
    assert res['age_years'] in (28, 45) and res['profile']['age_level'] == pred.schema['profile_choices']['age_level'][0]
    assert res['profile']['final_gender_code'] == {'Male': 1.0, 'Female': 2.0}[res['gender_text']]
    ids = {r.adgroup_id for r in pred.ads.itertuples()}
    assert res['ad_id'] in ids and res['pid'] in pred.schema['pid_choices']
    row = pred.ads[pred.ads['adgroup_id'] == res['ad_id']].iloc[0].to_dict()
    row['cate_id'], row['brand'] = pred.cate_enc(res['ad_cate']), pred.brand_enc(res['ad_brand'])
    x = pred.build_inputs(res['profile'], row, res['pid'], res['sessions'])
    assert x['campaign_id'][0] == int(row['campaign_enc']) and x['customer'][0] == int(row['customer_enc'])
    assert pred.predict_one(x) == pytest.approx(res['ctr'], abs=1e-5)           # equals a direct model call
    page = ' '.join(m.value for m in at.markdown)
    assert f"{res['ctr']:.1%}" in page and 'DEMO MODE' in page
    assert any(res['ad_cate_name'] in k.value for k in at.get('metric') or []) or res['ad_cate_name'] in page


def test_manual_price_reaches_the_model(monkeypatch, tmp_path):
    monkeypatch.setattr(history, 'DB_PATH', tmp_path / 'h.sqlite')
    out = {}
    for price in (30.0, 900.0):
        at = _predict_page('Load sample 1')
        next(w for w in at.number_input if w.label == 'Product Price').set_value(price).run()
        next(b for b in at.button if b.label == 'Predict Click Probability').click().run()
        assert not at.exception, [e.value for e in at.exception]
        out[price] = at.session_state['last']
    assert out[30.0]['price'] == 30.0 and out[900.0]['price'] == 900.0
    assert out[30.0]['ctr'] != out[900.0]['ctr']                                  # price is a real model input

    from dsin.config import Paths
    from dsin.predict import Predictor
    pred = Predictor(Paths('demo'))
    res = out[900.0]
    row = pred.ads[pred.ads['adgroup_id'] == res['ad_id']].iloc[0].to_dict()
    row['cate_id'], row['brand'] = pred.cate_enc(res['ad_cate']), pred.brand_enc(res['ad_brand'])
    row['price_scaled'] = (900.0 - pred.schema['price_mean']) / pred.schema['price_scale']
    x = pred.build_inputs(res['profile'], row, res['pid'], res['sessions'])
    assert pred.predict_one(x) == pytest.approx(res['ctr'], abs=1e-5)


def test_non_positive_price_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(history, 'DB_PATH', tmp_path / 'h.sqlite')
    at = _predict_page('Load sample 1')
    at.session_state['ad_price'] = 0.0
    next(b for b in at.button if b.label == 'Predict Click Probability').click().run()
    assert at.session_state['pred_error'] and 'Price' in at.session_state['pred_error']
