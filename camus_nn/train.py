"""Train the shared sorted-frequency model from labeled quartet CSVs."""
import argparse
from collections import Counter
import csv
from itertools import permutations
from pathlib import Path

import numpy as np

from .io import save_model


def training_arrays(paths):
    rows = []
    for path in paths:
        with Path(path).open(newline='') as stream:
            reader = csv.DictReader(stream)
            required = {'case', 'split', 'c0', 'c1', 'c2', 'y0', 'y1', 'y2'}
            if not required <= set(reader.fieldnames or ()):
                raise ValueError('Training CSV needs case,split,c0,c1,c2,y0,y1,y2')
            rows.extend(reader)
    if not rows or any(not r['case'] or r['split'] != 'train' for r in rows):
        raise ValueError('Provide only labeled training networks (split=train)')
    counts = np.array([[float(r[f'c{i}']) for i in range(3)] for r in rows])
    labels = np.array([[float(r[f'y{i}']) for i in range(3)] for r in rows])
    if (not np.isfinite(counts).all() or np.any(counts < 0) or
            np.any(counts != np.floor(counts)) or np.any(counts.sum(axis=1) <= 0)):
        raise ValueError('Counts must be nonnegative integers with positive resolved totals')
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError('Binary membership labels must include both classes')
    # Equal total influence per network, then uniform weight over tied ranks.
    sizes = Counter(r['case'] for r in rows)
    weights = np.array([1 / sizes[r['case']] for r in rows])
    weights *= len(weights) / weights.sum()
    order = np.argsort(-counts, axis=1, kind='stable')
    x = np.take_along_axis(counts, order, axis=1) / counts.sum(axis=1, keepdims=True)
    y = np.take_along_axis(labels, order, axis=1)
    tied = x[:, :, None] == x[:, None, :]
    choices = [list(p) for p in permutations(range(3))]
    masks = [np.all(tied[:, np.arange(3), p], axis=1) for p in choices]
    multiplicity = np.sum(masks, axis=0)
    return (np.concatenate([x[m] for m in masks]),
            np.concatenate([y[m][:, p] for p, m in zip(choices, masks)]),
            np.concatenate([weights[m] / multiplicity[m] for m in masks]), sorted(sizes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    parser.add_argument('--model', type=Path, required=True, help='New JSON model file')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden-width', type=int, default=8)
    args = parser.parse_args()
    if args.model.exists():
        parser.error('Model exists; choose a new filename')
    if args.hidden_width < 1 or len({p.resolve() for p in args.data}) != len(args.data):
        parser.error('Use a positive hidden width and distinct input files')
    # Load the optimizer before the tensor backend for consistent shared runtimes.
    import scipy.optimize
    import torch
    from .model import ShallowNetwork
    torch.set_num_threads(1)
    x, y, weights, cases = training_arrays(args.data)
    model = ShallowNetwork(seed=args.seed, hidden_width=args.hidden_width)
    model.fit(x, y, weights)
    if not model.optimization_['converged']:
        raise RuntimeError('Training did not converge')
    args.model.parent.mkdir(parents=True, exist_ok=True)
    save_model(args.model, model)
    print(f'Saved {args.model}: {len(cases)} training networks')


if __name__ == '__main__':
    main()
