"""Small scientific and end-to-end checks; all temporary files stay in TMPDIR."""
from collections import Counter
import csv
from itertools import combinations, permutations, product
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from camus_nn.evidence import membership, write_evidence
from camus_nn.io import load_model
from camus_nn.infer import DEFAULT_BIN_DIR
from camus_nn.iterate import refine, switching_guides
from camus_nn.sampling import sample_quartets
from camus_nn.topology import Tree, parse_network, validate_network
from camus_nn.train import training_arrays

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get('CAMUS_NN_BIN', DEFAULT_BIN_DIR))
ONE = '(OUT,((A,(B)#H1),(#H1,C)));'
TWO = '(OUT,(((A,(B)#H1),(#H1,C)),((D,(E)#H2),(#H2,F))));'


def displayed_choices(network, groups):
    """Independent small-network oracle: enumerate all parent switchings."""
    graph = parse_network(network)
    hybrids = [u for u in graph if graph.in_degree(u) == 2]
    choices = [[(p, u) for p in graph.predecessors(u)] for u in hybrids]
    displayed = np.zeros((len(groups), 3), dtype=bool)
    for removed in product(*choices):
        tree = graph.copy()
        tree.remove_edges_from(removed)
        values = Tree.from_graph(tree, suppress_unary=True).quartet_topologies(groups)
        if np.any(values < 0):
            raise AssertionError('Invalid displayed tree')
        displayed[np.arange(len(groups)), values] = True
    return displayed


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = load_model(ROOT / 'models/all-frequency.json')

    def test_ties_and_topology_permutations_are_symmetric(self):
        counts = np.array([[4, 4, 2], [2, 4, 4], [3, 3, 3], [8, 1, 1]])
        probabilities = membership(counts, self.model)
        for permutation in permutations(range(3)):
            np.testing.assert_array_equal(membership(counts[:, permutation], self.model),
                                          probabilities[:, permutation])
        self.assertEqual(probabilities[0, 0], probabilities[0, 1])
        self.assertTrue(np.all(probabilities[2] == probabilities[2, 0]))

    def test_locus_count_rescaling_preserves_membership_probabilities(self):
        counts = np.array([[6, 3, 1], [4, 4, 2], [9, 0, 1]])
        expected = membership(counts, self.model)
        for multiplier in (2, 10, 1000):
            np.testing.assert_array_equal(membership(counts * multiplier, self.model), expected)

    def test_training_network_weights_and_split_rejection(self):
        x, y, weights, cases = training_arrays([ROOT / 'examples/train.csv'])
        self.assertEqual(cases, ['reticulate', 'single'])
        np.testing.assert_allclose(x.sum(axis=1), 1)
        self.assertAlmostEqual(weights.sum(), 12)
        # Both source networks have six rows. Tie expansion retains total weight.
        self.assertAlmostEqual(weights[(y.sum(axis=1) == 1)].sum(), 6)
        self.assertAlmostEqual(weights[(y.sum(axis=1) == 2)].sum(), 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'leak.csv'
            path.write_text((ROOT / 'examples/train.csv').read_text().replace(',train,', ',test,', 1))
            with self.assertRaisesRegex(ValueError, 'only labeled training'):
                training_arrays([path])

    def test_repeated_draws_change_evidence_without_changing_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'counts.csv'
            source.write_text('t0,t1,t2,t3,c0,c1,c2,n_loci,constraint\nA,B,C,OUT,6,4,0,10,-1\n')
            write_evidence(source, root / 'one.csv', self.model, 'lqw', ['A', 'B', 'C', 'OUT'])
            write_evidence(source, root / 'three.csv', self.model, 'lqw', ['A', 'B', 'C', 'OUT'],
                           multiplicities=Counter({('A', 'B', 'C', 'OUT'): 3}), batch_rows=1)
            def values(path):
                with path.open() as stream:
                    return [float(r['weight']) for r in csv.DictReader(stream)]
            np.testing.assert_allclose(values(root / 'three.csv'), np.array(values(root / 'one.csv')) * 3,
                                       rtol=1e-15, atol=0)


class SwitchingTests(unittest.TestCase):
    def test_single_cycle_neighbors_and_sampler_budgets(self):
        guide = '(OUT,(((A,B),C),((D,E),F)));'
        neighbors = list(switching_guides(TWO, guide))
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(len(set(neighbors)), 2)
        groups = list(combinations(sorted(Tree.from_newick(guide).tips), 4))
        allowed = displayed_choices(TWO, groups)
        for neighbor in neighbors:
            topology = Tree.from_newick(neighbor).quartet_topologies(groups)
            self.assertTrue(allowed[np.arange(len(groups)), topology].all())
        tree = Tree.from_newick(guide)
        for sampler in ('uniform', 'edge-uniform', 'edge-walk'):
            draws = sample_quartets(tree, sampler, seed=42, multiplicity=3)
            self.assertEqual(len(draws), 3 * (2 * len(tree.tips) - 3))
        rooted = sample_quartets(tree, 'rooted')
        self.assertEqual(len(rooted), 20)
        self.assertTrue(all('OUT' in q for q in rooted))

    def test_iteration_compares_total_scores_and_keeps_a_better_incumbent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / 'weights.csv'
            evidence.write_text('a,b,c,d,weight\n')
            def solve(guide, directory, iterative):
                initial = directory.name == '0000'
                result = dict(k=1 if initial else 0, objective=20. if initial else 0.,
                              network=ONE if initial else guide.read_text().strip())
                score = 100. if initial else 20.
                (directory / 'network.metrics.json').write_text(json.dumps(dict(objective=dict(
                    guide_score=score, added_gain=result['objective'], total_score=score + result['objective']))))
                for name in ('network.csv', 'network.branches.json'):
                    (directory / name).write_text('test fixture')
                return result
            result, report = refine('(OUT,((A,B),C));', evidence, root, solve)
            self.assertEqual(result['objective'], 120.)
            self.assertEqual(report['winner'], 0)
            self.assertEqual(len(report['trials']), 2)


@unittest.skipUnless((BIN / 'camus-native').is_file() and (BIN / 'camus-count').is_file(), 'Install the package first')
class NativeTests(unittest.TestCase):
    def test_serializer_preserves_reference_hybrid_ids_above_twelve_cycles(self):
        fixture = json.loads((ROOT / 'tests/fixtures/serializer-14-reticulations.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'serialize.cpp'
            source.write_text('#include "serialize.hpp"\n#include <iostream>\n'
                'int main(int argc,char** argv){camus::Tree tree(argv[1]);std::vector<camus::Branch> branches;'
                'for(int i=2;i<argc;i+=2)branches.push_back({std::stoi(argv[i]),std::stoi(argv[i+1])});'
                'std::cout<<camus::network_newick(tree,branches);}\n')
            binary = root / 'serialize'
            subprocess.run([os.environ.get('CXX', 'g++'), '-O2', '-std=c++17', '-fopenmp',
                '-I', str(ROOT / 'native'), str(source), '-o', str(binary)], check=True, capture_output=True)
            guide = root / 'guide.nwk'
            guide.write_text(fixture['guide'])
            actual = subprocess.check_output([str(binary), str(guide),
                *[str(node) for branch in fixture['branches'] for node in branch]], text=True)
            self.assertEqual(actual, fixture['expected'])

    def test_signed_candidates_match_independent_displayed_quartets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guide = root / 'guide.nwk'
            guide.write_text('(OUT,(((A,B),C),((D,E),F)));')
            groups = list(combinations(sorted(Tree.from_newick(guide.read_text()).tips), 4))
            target = displayed_choices(TWO, groups)
            weights = np.where(target, 1., -1.)
            with (root / 'weights.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['a', 'b', 'c', 'd', 'weight'])
                for group, values in zip(groups, weights):
                    for order, value in zip(((0,1,2,3), (0,2,1,3), (0,3,1,2)), values):
                        writer.writerow([*[group[i] for i in order], value])
            with (root / 'native.log').open('w') as log:
                subprocess.run([str(BIN / 'camus-native'), '-q', '0', '-quartet-weights', '-signed-weights',
                    '--anchor-outgroup', '-o', str(root / 'network'), str(guide), str(root / 'weights.csv')],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            start = displayed_choices(guide.read_text(), groups)
            with (root / 'network.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertGreater(len(rows), 1)
            for row in rows:
                graph = parse_network(row['Extended Newick'])
                validate_network(graph, level1=True)
                self.assertEqual(sum(graph.in_degree(u) == 2 for u in graph), int(row['Number of Branches']))
                displayed = displayed_choices(row['Extended Newick'], groups)
                self.assertFalse(np.any(start & ~displayed))
                self.assertEqual(float(row['Signed Quartet Gain']), weights[displayed & ~start].sum())

    def test_cli_methods_iteration_caps_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guide = root / 'guide.nwk'
            guide.write_text('(OUT,(((A,B),C),((D,E),F)));\n')
            genes = root / 'genes.nwk'
            genes.write_text(guide.read_text() * 6 + '(OUT,((A,(B,C)),(D,(E,F))));\n' * 4)
            common = [sys.executable, '-B', '-m', 'camus_nn.infer', '--genes', str(genes),
                      '--guide', str(guide), '--bin-dir', str(BIN)]
            options = [('lqf', []), ('lqw', []), ('iterate', ['--iterate']),
                       ('cap', ['--iterate', '--search-cap', '1']),
                       ('overall', ['--iterate', '--inner-solver', 'overall']),
                       ('zero', ['--iterate', '--search-cap', '0'])]
            for name, extra in options:
                output = root / name
                result = subprocess.run([*common, '--method', 'lqf' if name == 'lqf' else 'lqw',
                    '--out', str(output), *extra], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr + (output / 'run.log').read_text())
                prediction = json.loads((output / 'prediction.json').read_text())
                validate_network(parse_network(prediction['network']), level1=True)
                self.assertFalse((output / 'tmp').exists())
                if name in ('iterate', 'cap', 'overall', 'zero'):
                    report = json.loads((output / 'iteration.json').read_text())
                    self.assertLessEqual(len(report['trials']), 5)
                    self.assertTrue(report['weights_frozen'])
                    self.assertEqual(report['inner_optimization_exact'], name == 'overall')
                    self.assertEqual(prediction['objective'], max(t['total_score'] for t in report['trials']))
                if name in ('cap', 'zero'):
                    self.assertLessEqual(prediction['k'], 1 if name == 'cap' else 0)
            output = root / 'timeout'
            timed = subprocess.run([*common, '--out', str(output), '--timeout', '.001'], capture_output=True)
            self.assertNotEqual(timed.returncode, 0)
            self.assertEqual(json.loads((output / 'failure.json').read_text())['status'], 'timeout')
            self.assertFalse((output / 'tmp').exists())


if __name__ == '__main__':
    unittest.main()
