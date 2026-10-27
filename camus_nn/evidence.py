"""Stream count evidence and model predictions in bounded batches."""
import csv
from itertools import islice

import numpy as np

ORDERS = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))


def membership(counts, model):
    counts = np.asarray(counts, dtype=float)
    if (counts.ndim != 2 or counts.shape[1] != 3 or not np.isfinite(counts).all()
            or np.any(counts < 0) or np.any(counts.sum(axis=1) <= 0)):
        raise ValueError('Each quartet needs three counts with a positive resolved total')
    order = np.argsort(-counts, axis=1, kind='stable')
    frequencies = np.take_along_axis(counts, order, axis=1) / counts.sum(axis=1)[:, None]
    raw = model.predict_proba(frequencies)
    tied = frequencies[:, :, None] == frequencies[:, None, :]
    values = (tied * raw[:, None, :]).sum(axis=2) / tied.sum(axis=2)
    result = np.empty_like(values)
    np.put_along_axis(result, order, values, axis=1)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError('Invalid model probabilities')
    return result


def write_evidence(source, destination, model, method, taxa, *, multiplicities=None, batch_rows=65536):
    if method not in ('orig', 'lqf', 'lqw') or batch_rows < 1:
        raise ValueError('Require orig, lqf or lqw and a positive batch size')
    total = 0
    previous = -1
    ids = {name: i for i, name in enumerate(sorted(taxa))}
    with source.open(newline='') as stream, destination.open('w', newline='') as output:
        reader = csv.DictReader(stream)
        fields = [f't{i}' for i in range(4)] + [f'c{i}' for i in range(3)] + ['n_loci', 'constraint']
        if reader.fieldnames != fields:
            raise ValueError('Invalid native count table')
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(['a', 'b', 'c', 'd', 'weight'] if method == 'lqw' else
                        fields + ['keep0', 'keep1', 'keep2'])
        while rows := list(islice(reader, batch_rows)):
            groups = [tuple(r[f't{i}'] for i in range(4)) for r in rows]
            keys = []
            for group in groups:
                indices = [ids[t] for t in group]
                if indices != sorted(set(indices)):
                    raise ValueError('Noncanonical quartet group')
                keys.append(sum(value << (15 * i) for i, value in enumerate(indices)))
            if keys[0] <= previous or any(b <= a for a, b in zip(keys, keys[1:])):
                raise ValueError('Duplicate or unordered native quartet group')
            previous = keys[-1]
            counts = np.array([[int(r[f'c{i}']) for i in range(3)] for r in rows], dtype=np.int64)
            loci = np.array([int(r['n_loci']) for r in rows], dtype=np.int64)
            if np.any(counts.sum(axis=1) > loci) or np.any(loci < 1):
                raise ValueError('Counts exceed locus coverage')
            p = membership(counts, model) if method != 'orig' else None
            draws = np.array([multiplicities[g] for g in groups]) if multiplicities is not None else 1
            if method == 'lqw':
                weights = (2 * p - 1) * ((loci * draws) / 1000)[:, None]
                for group, values in zip(groups, weights):
                    for order, value in zip(ORDERS, values):
                        if value != 0:
                            writer.writerow([group[i] for i in order] + [float(value)])
            else:
                # Original CAMUS applies its count filter inside the C++ core.
                keep = np.ones_like(counts) if method == 'orig' else ((p > .5) & (counts > 0)).astype(int)
                counts = counts * (draws[:, None] if multiplicities is not None else 1)
                loci = loci * draws
                for r, group, c, n, mask in zip(rows, groups, counts, loci, keep):
                    writer.writerow([*group, *c, n, r['constraint'], *mask])
            total += len(rows)
    if not total or multiplicities is not None and total != len(multiplicities):
        raise ValueError('Incomplete quartet evidence')
    return total
