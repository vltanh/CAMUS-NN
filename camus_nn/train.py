"""Fit the neural network from labeled quartets or network/gene-tree cases."""
import argparse
from collections import Counter
import csv
from itertools import permutations
from pathlib import Path
import tempfile

import numpy as np

from .io import save_model
from .infer import DEFAULT_BIN_DIR


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
    labels = labels.astype(np.int64)
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
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--quartet-table', type=Path, nargs='+', metavar='TRAINING.csv', help='Labeled quartet training data (one or more CSV files)')
    inputs.add_argument('--training-cases', type=Path, metavar='CASES.json', help='Training case list with true networks and gene trees (JSON file)')
    parser.add_argument('--output-model', type=Path, required=True, metavar='MODEL.json', help='New trained neural-network model (JSON file)')
    parser.add_argument('--bin-dir', type=Path, default=DEFAULT_BIN_DIR, help='Directory with camus-count for --training-cases (default: bundled)')
    parser.add_argument('--threads', type=int, default=1, help='Counting threads for --training-cases (default: 1; positive)')
    parser.add_argument('--max-switchings', type=int, default=4096, help='Exact labeling budget per network for --training-cases (default: 4096)')
    parser.add_argument('--seed', type=int, default=42, help='Nonnegative NN initialization seed (default: 42)')
    parser.add_argument('--hidden-width', type=int, default=8, help='Positive number of ReLU hidden units (default: 8)')
    args = parser.parse_args()
    if args.output_model.exists():
        parser.error('Model exists; choose a new filename')
    if min(args.hidden_width, args.threads, args.max_switchings) < 1:
        parser.error('Use positive hidden width, threads and max-switchings')
    if args.quartet_table and len({p.resolve() for p in args.quartet_table}) != len(args.quartet_table):
        parser.error('Use distinct training CSV files')
    # Load the optimizer before the tensor backend for consistent shared runtimes.
    import scipy.optimize
    import torch
    from .model import ShallowNetwork
    torch.set_num_threads(1)
    if args.quartet_table:
        x, y, weights, cases = training_arrays(args.quartet_table)
    else:
        from .prepare import prepare_training_csv
        with tempfile.TemporaryDirectory(prefix='camus-nn-train-') as directory:
            table = Path(directory) / 'training.csv'
            prepare_training_csv(args.training_cases, table, counter=args.bin_dir / 'camus-count',
                                 threads=args.threads, max_switchings=args.max_switchings)
            x, y, weights, cases = training_arrays([table])
    model = ShallowNetwork(seed=args.seed, hidden_width=args.hidden_width)
    model.fit(x, y, weights)
    if not model.optimization_['converged']:
        raise RuntimeError('Training did not converge')
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    save_model(args.output_model, model)
    print(f'Saved {args.output_model}: {len(cases)} training networks')


if __name__ == '__main__':
    main()
