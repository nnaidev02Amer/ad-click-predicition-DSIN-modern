"""DSIN model builder (TensorFlow 2.x / Keras).

The network itself is DeepCTR 0.9.4's `DSIN` - the reference implementation written by the DSIN paper's
co-author, already on tf.keras. It contains every component the paper describes:

    * session division (sess_0 .. sess_{K-1}, each up to T behaviours; embeddings of cate_id + brand)
    * session interest extractor: multi-head self-attention (Transformer layer) inside each session,
      with bias encoding (BE: session / position / unit biases) or positional encoding, then average pooling
    * session interest interacting layer: Bi-LSTM across the K session interests
    * session interest activating layers: target-attention (local activation units) over the
      extractor output AND over the Bi-LSTM output, keyed by the ad's cate_id+brand embedding
    * concatenation with user-profile + ad embeddings + dense (price) -> DNN (200,80) -> sigmoid CTR

Here we only build the feature columns from the saved schema and call it.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import ModelParams


def build_feature_columns(schema: dict, params: ModelParams = ModelParams()):
    from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
    vocab = schema['vocab_sizes']
    dim = params.embedding_dim
    sparse = [SparseFeat(f, vocabulary_size=vocab[f], embedding_dim=dim)
              for f in schema['sparse_features'] + schema['sess_features']]
    dense = [DenseFeat(f, 1) for f in schema['dense_features']]
    varlen = []
    for i in range(schema['sess_count']):
        for f in schema['sess_features']:
            varlen.append(VarLenSparseFeat(
                SparseFeat(f'sess_{i}_{f}', vocabulary_size=vocab[f], embedding_dim=dim, embedding_name=f),
                maxlen=schema['sess_max_len']))
    return sparse + varlen + dense


def build_dsin(schema: dict, params: ModelParams = ModelParams(), seed: int = 1024):
    """Return an (uncompiled) Keras DSIN model for the given data schema."""
    from deepctr.models import DSIN
    cols = build_feature_columns(schema, params)
    return DSIN(cols, list(schema['sess_features']), sess_max_count=schema['sess_count'],
                bias_encoding=params.bias_encoding, att_embedding_size=params.att_embedding_size,
                att_head_num=params.att_head_num, dnn_hidden_units=tuple(params.dnn_hidden_units),
                dnn_activation=params.dnn_activation, seed=seed)


def params_to_dict(p: ModelParams) -> dict:
    d = dict(p.__dict__) if hasattr(p, '__dict__') else {}
    d['dnn_hidden_units'] = list(p.dnn_hidden_units)
    return d


def params_from_dict(d: dict) -> ModelParams:
    d = dict(d)
    d['dnn_hidden_units'] = tuple(d['dnn_hidden_units'])
    return ModelParams(**{k: v for k, v in d.items() if k in ModelParams.__dataclass_fields__})


def load_trained(model_dir: Path, schema: dict):
    """Rebuild the architecture from model_config.json and load the saved weights."""
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / 'model_config.json').read_text())
    model = build_dsin(schema, params_from_dict(cfg['model_params']))
    model.load_weights(str(model_dir / 'dsin.weights.h5'))
    return model, cfg


def model_input_names(model):
    return [i.name.split(':')[0] for i in model.inputs]
