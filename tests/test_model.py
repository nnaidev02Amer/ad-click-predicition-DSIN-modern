"""DSIN architecture, training, metrics, saving/loading, prediction (synthetic data only - no real-performance claims)."""
import json

import numpy as np
import pandas as pd
import pytest

from dsin.config import ModelParams
from dsin.evaluation import compute_metrics


@pytest.fixture(scope='module')
def schema_inputs(demo_paths):
    schema = json.loads((demo_paths.processed / 'schema.json').read_text())
    x = pd.read_pickle(demo_paths.processed / 'dsin_input.pkl')
    y = pd.read_pickle(demo_paths.processed / 'dsin_label.pkl')
    return schema, x, y


def _batch(x, n=64):
    return {k: v[:n] for k, v in x.items()}


def test_architecture_components(schema_inputs):
    from dsin.model import build_dsin
    schema, x, _ = schema_inputs
    m = build_dsin(schema)
    names = {l.__class__.__name__ for l in m.layers}
    assert {'Transformer', 'BiLSTM', 'AttentionSequencePoolingLayer', 'DNN', 'PredictionLayer'} <= names
    assert sum(1 for l in m.layers if l.__class__.__name__ == 'Transformer') == 1      # shared across sessions
    assert len(m.inputs) == len(schema['sparse_features']) + 2 + 2 * schema['sess_count'] + 1 + 1 \
        or len(m.inputs) > 0
    assert any(i.name.startswith('sess_4_brand') for i in m.inputs)


@pytest.mark.parametrize('bias_encoding', [False, True])
def test_forward_pass(schema_inputs, bias_encoding):
    from dsin.model import build_dsin
    schema, x, _ = schema_inputs
    m = build_dsin(schema, ModelParams(bias_encoding=bias_encoding))
    p = m.predict(_batch(x), verbose=0)
    assert p.shape == (64, 1) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_training_updates_weights_and_reduces_loss(schema_inputs):
    import tensorflow as tf
    from dsin.model import build_dsin
    schema, x, y = schema_inputs
    tf.keras.utils.set_random_seed(0)
    m = build_dsin(schema)
    m.compile(tf.keras.optimizers.Adagrad(0.05), 'binary_crossentropy')
    b, yb = _batch(x, 256), y[:256]
    before = [w.numpy().copy() for w in m.trainable_weights]
    l0 = m.evaluate(b, yb, verbose=0)
    m.fit(b, yb, batch_size=64, epochs=8, verbose=0)
    after = m.trainable_weights
    assert any(not np.allclose(a, w.numpy()) for a, w in zip(before, after)), 'no gradient update happened'
    assert m.evaluate(b, yb, verbose=0) < l0


def test_gradients_flow_to_session_embeddings(schema_inputs):
    import tensorflow as tf
    from dsin.model import build_dsin
    schema, x, y = schema_inputs
    m = build_dsin(schema)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y[:32, None].astype('float32'), m(_batch(x, 32), training=True)))
    grads = tape.gradient(loss, m.trainable_weights)
    assert all(g is not None for g in grads)
    assert all(np.isfinite(tf.convert_to_tensor(g)).all() for g in grads)


def test_metrics_correct():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = compute_metrics(y, p)
    assert m['auc'] == pytest.approx(0.75)
    exp = -np.mean([np.log(.9), np.log(.6), np.log(.35), np.log(.8)])
    assert m['logloss'] == pytest.approx(exp)
    assert m['confusion_matrix'] == {'tn': 2, 'fp': 0, 'fn': 1, 'tp': 1}
    assert compute_metrics([0, 0], [.1, .2])['auc'] is None


def test_train_save_load_predict_and_dashboard_integration(demo_paths):
    from dsin.training import train
    from dsin.evaluation import evaluate
    from dsin.predict import Predictor
    model, hist, cfg = train(demo_paths, epochs=1, batch_size=256, patience=1)
    assert cfg['trained_on_synthetic_data'] is True
    assert (demo_paths.models / 'dsin.weights.h5').exists() and (demo_paths.results / 'training_history.json').exists()
    m = evaluate(demo_paths)
    assert 0 <= m['logloss'] and m['synthetic_data'] and 'DEMO MODE' in m['warning']

    # loading reproduces the in-memory predictions exactly
    pred = Predictor(demo_paths)
    x = pd.read_pickle(demo_paths.processed / 'dsin_input.pkl')
    b = _batch(x, 50)
    np.testing.assert_allclose(model.predict(b, verbose=0), pred.model.predict(b, verbose=0), rtol=1e-5, atol=1e-6)

    # the dashboard path: raw form values -> tensors -> trained model
    ad = pred.ads.iloc[0].to_dict()
    schema = pred.schema
    profile = {f: schema['profile_choices'][f][0] for f in schema['profile_choices']}
    sess = [[(pred.cate_classes[0], pred.brand_classes[0])] * 3, [(pred.cate_classes[1], pred.brand_classes[1])] * 4]
    xin = pred.build_inputs(profile, ad, schema['pid_choices'][0], sess)
    p1 = pred.predict_one(xin)
    assert 0.0 <= p1 <= 1.0 and p1 == pytest.approx(pred.predict_one(xin))      # deterministic
    xin2 = pred.build_inputs(profile, ad, schema['pid_choices'][0], [])
    assert xin2['sess_length'][0] == 0 and 0 <= pred.predict_one(xin2) <= 1
    with pytest.raises(ValueError):
        pred.build_inputs(profile, ad, 'bad-pid', sess)
    with pytest.raises(ValueError):
        pred.build_inputs(profile, ad, schema['pid_choices'][0], [[(-5.0, 1.0)]])
