"""Portable JSON model and inference artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_model(path, model):
    save(path, dict(schema='camus-nn-frequency/v1', architecture=[3, model.hidden_width, 3],
        feature_mode='sorted', seed=model.seed, weight_decay=model.weight_decay,
        mean=model.mean_.tolist(), scale=model.scale_.tolist(),
        parameters=model.parameters_.tolist(), optimization=model.optimization_))


def load_model(path):
    from .model import ShallowNetwork
    value = json.loads(Path(path).read_text())
    architecture = value.get('architecture', [])
    if (value.get('schema') != 'camus-nn-frequency/v1' or value.get('feature_mode') != 'sorted'
            or len(architecture) != 3 or architecture[0] != 3 or architecture[2] != 3):
        raise ValueError('Expected a CAMUS-NN sorted-frequency model')
    model = ShallowNetwork(seed=value['seed'], hidden_width=architecture[1],
                           weight_decay=value['weight_decay'])
    model.mean_ = np.asarray(value['mean'], dtype=float)
    model.scale_ = np.asarray(value['scale'], dtype=float)
    model.parameters_ = np.asarray(value['parameters'], dtype=float)
    model.unpack(model.parameters_, 3, model.hidden_width)
    if (model.mean_.shape != (3,) or model.scale_.shape != (3,) or
            not all(np.isfinite(x).all() for x in (model.mean_, model.scale_, model.parameters_))
            or np.any(model.scale_ <= 0)):
        raise ValueError('Invalid model parameters or scaling')
    return model
