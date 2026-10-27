"""Prepare a labeled training CSV from a JSON list of networks and gene trees."""
import argparse
import csv
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import tempfile

import networkx as nx
import numpy as np

from .infer import DEFAULT_BIN_DIR
from .topology import Tree, parse_network


def displayed_memberships(graph, quartets, max_switchings=4096):
    """Exact unrooted quartet membership over incoming-edge choices."""
    hybrids = [u for u in graph if graph.in_degree(u) == 2]
    parents = [tuple(graph.predecessors(u)) for u in hybrids]
    if 2 ** len(hybrids) > max_switchings:
        raise ValueError('Exact truth labeling exceeds --max-switchings')
    result = np.zeros((len(quartets), 3), dtype=bool)
    for choices in product(*parents):
        tree = graph.copy()
        for u, incoming, chosen in zip(hybrids, parents, choices):
            tree.remove_edges_from((p, u) for p in incoming if p != chosen)
        topologies = Tree.from_graph(tree, suppress_unary=True).quartet_topologies(quartets)
        resolved = np.flatnonzero(topologies >= 0)
        result[resolved, topologies[resolved]] = True
    if np.any(~result.any(axis=1)):
        raise ValueError('Every training quartet needs a resolved topology in the true network')
    return result


def read_cases(manifest):
    manifest = Path(manifest).resolve()
    rows = json.loads(manifest.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('JSON input must be a nonempty list of training cases')
    cases = []
    for row in rows:
        if (not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                for key in ('case', 'network', 'genes')) or row.get('split', 'train') != 'train'):
            raise ValueError('Each training case needs case, network and genes; split must be train')
        cases.append(dict(case=row['case'], network=(manifest.parent / row['network']).resolve(strict=True),
                          genes=(manifest.parent / row['genes']).resolve(strict=True)))
    if any(len({row[key] for row in cases}) != len(cases) for key in ('case', 'network', 'genes')):
        raise ValueError('Use distinct case IDs, true networks and gene-tree files')
    return cases


def prepare_training_csv(manifest, output, *, counter=DEFAULT_BIN_DIR / 'camus-count',
                         threads=1, max_switchings=4096):
    if threads < 1 or max_switchings < 1:
        raise ValueError('Threads and max-switchings must be positive')
    cases = read_cases(manifest)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Training CSV exists; choose a new filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + '.partial')
    taxa_columns = [f't{i}' for i in range(4)]
    count_columns = [f'c{i}' for i in range(3)]
    total = 0
    with partial.open('x', newline='') as stream:
        try:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['case', 'split', *taxa_columns, *count_columns, 'n_loci', 'y0', 'y1', 'y2'])
            with tempfile.TemporaryDirectory(prefix='camus-nn-prepare-') as directory:
                counts = Path(directory) / 'counts.csv'
                for index, case in enumerate(cases):
                    graph = nx.DiGraph(parse_network(case['network'].read_text(), allow_parallel=True))
                    taxa = {graph.nodes[u]['label'] for u in graph if graph.out_degree(u) == 0}
                    if len(taxa) < 4:
                        raise ValueError('Training requires at least four taxa')
                    hybrids = sum(graph.in_degree(u) == 2 for u in graph)
                    if 2 ** hybrids > max_switchings:
                        raise ValueError('Exact truth labeling exceeds --max-switchings')
                    subprocess.run([str(counter), '-gene-trees', str(case['genes']),
                                    '-n', str(threads), '-o', str(counts)], check=True)
                    with counts.open(newline='') as source:
                        rows = sorted(csv.DictReader(source), key=lambda r: tuple(r[key] for key in taxa_columns))
                    quartets = [tuple(row[key] for key in taxa_columns) for row in rows]
                    if len(rows) != math.comb(len(taxa), 4) or {t for q in quartets for t in q} != taxa:
                        raise ValueError('Gene trees and true network must have the same taxa and resolved quartet coverage')
                    labels = displayed_memberships(graph, quartets, max_switchings)
                    for row, label in zip(rows, labels):
                        writer.writerow([case['case'], 'train', *[row[key] for key in taxa_columns + count_columns],
                                         row['n_loci'], *label.astype(int)])
                    total += len(rows)
                    print(f'Training network {index + 1}: {len(taxa)} taxa, {len(rows)} quartets', flush=True)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    partial.replace(output)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-cases', type=Path, required=True, metavar='CASES.json', help='Training case list with case/network/genes records (JSON file)')
    parser.add_argument('--output-table', type=Path, required=True, metavar='TRAINING.csv', help='New labeled quartet training data (CSV file)')
    parser.add_argument('--bin-dir', type=Path, default=DEFAULT_BIN_DIR, help='Directory with camus-count (default: bundled)')
    parser.add_argument('--threads', type=int, default=1, help='C++ counting threads (default: 1; positive)')
    parser.add_argument('--max-switchings', type=int, default=4096, help='Exact truth-labeling budget per network (default: 4096; positive)')
    args = parser.parse_args()
    rows = prepare_training_csv(args.training_cases, args.output_table, counter=args.bin_dir / 'camus-count',
                                threads=args.threads, max_switchings=args.max_switchings)
    print(f'Saved {args.output_table}: {rows} labeled quartets')


if __name__ == '__main__':
    main()
