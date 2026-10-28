"""Cluster metrics checked against exhaustive switchings and saved-file CLI behavior."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx

from camus_nn.evaluate import (
    cluster_summary, evaluate_networks, hardwired_cluster_distance,
    hardwired_clusters, main, softwired_cluster_distance,
)
from camus_nn.topology import parse_network

TREE = '(OUT,((A,B),(C,D)));'
ONE = '(OUT,((A,(B)#H1),(#H1,(C,D))));'
FIXTURES = (
    TREE, ONE,
    '(OUT,(((A,(B)#H1),(#H1,C)),((D,(E)#H2),(#H2,F))));',
    '(OUT,((A,(((B,(C)#H2),(#H2,D)))#H1),(#H1,E)));',
    '(OUT,((A,((B)#H1,(C)#H2)),(#H1,(#H2,D))));',
    '(A,(B)#H1,((#H1,D),(C,OUT)));',
    '((((((((x2)#H4),(x3),#H4,x4)#H3),OUT)#H2,#H3),#H2,(x1)#H1),#H1);',
    '(OUT,(((A,B))#H1,#H1));',
    "[comment](OUT,('A space':1,('B''quote':2,C:3)95:4));",
)


def reachable_clusters(graph, taxa, outgroup):
    """Independent leaf-reachability oracle, including dead-end switchings."""
    clusters = set()
    for node in graph:
        below = nx.descendants(graph, node) | {node}
        leaves = frozenset(graph.nodes[u]['label'] for u in below
            if graph.nodes[u].get('label') in taxa and graph.out_degree(u) == 0)
        if outgroup is not None and outgroup in leaves:
            leaves = taxa-leaves
        if 1 < len(leaves) < len(taxa):
            clusters.add(leaves)
    return frozenset(clusters)


def exhaustive_clusters(text, outgroup):
    graph = nx.DiGraph(parse_network(text, allow_parallel=True))
    taxa = frozenset(graph.nodes[u]['label'] for u in graph if graph.out_degree(u) == 0)
    hybrids = [u for u in graph if graph.in_degree(u) == 2]
    parents = [list(graph.predecessors(u)) for u in hybrids]
    profiles = []
    for choices in product((0, 1), repeat=len(hybrids)):
        tree = graph.copy()
        for node, incoming, choice in zip(hybrids, parents, choices):
            tree.remove_edge(incoming[1-choice], node)
        profiles.append(reachable_clusters(tree, taxa, outgroup))
    return taxa, frozenset().union(*profiles), frozenset.intersection(*profiles)


class EvaluationTests(unittest.TestCase):
    def test_factored_sw_matches_independent_full_enumeration(self):
        for network in FIXTURES:
            for outgroup in (None, 'OUT'):
                with self.subTest(network=network, outgroup=outgroup):
                    taxa, union, invariant = exhaustive_clusters(network, outgroup)
                    actual = cluster_summary(network, outgroup=outgroup)
                    self.assertEqual(actual.taxa, taxa)
                    self.assertEqual(actual.union, union)
                    self.assertEqual(actual.invariant, invariant)

    def test_hardwired_matches_reachability_without_displayed_trees(self):
        for network in FIXTURES:
            for outgroup in (None, 'OUT'):
                with self.subTest(network=network, outgroup=outgroup):
                    graph = parse_network(network, allow_parallel=True)
                    taxa, actual = hardwired_clusters(network, outgroup=outgroup)
                    self.assertEqual(actual, reachable_clusters(graph, taxa, outgroup))

    def test_rates_and_raw_hwcd_have_separate_scales(self):
        estimate = '(OUT,((A,B),C,D));'
        scores = evaluate_networks(TREE, estimate)
        self.assertEqual(scores['true_clusters'], 3)
        self.assertEqual(scores['inferred_clusters'], 2)
        self.assertEqual(scores['softwired_fn'], 1/3)
        self.assertEqual(scores['softwired_fp'], 0)
        self.assertEqual(scores['softwired_error'], 1/6)
        self.assertEqual(scores['hardwired_distance'], 1)
        self.assertEqual(scores['hardwired_fn_count'], 1)
        self.assertEqual(scores['hardwired_fp_count'], 0)
        reverse = evaluate_networks(estimate, TREE)
        self.assertEqual(reverse['softwired_fn'], 0)
        self.assertEqual(reverse['softwired_fp'], 1/3)
        self.assertEqual(reverse['hardwired_distance'], 1)

    def test_complete_network_and_displayed_tree_union_are_different(self):
        network = '(OUT,((A,(E,(B,C)#H1)),(D,#H1)));'
        _, hardwired = hardwired_clusters(network, outgroup='OUT')
        softwired = cluster_summary(network, outgroup='OUT').union
        self.assertIn(frozenset('EBC'), hardwired)
        self.assertNotIn(frozenset('AE'), hardwired)
        self.assertIn(frozenset('AE'), softwired)

    def test_parallel_edges_and_unary_duplicates_count_once(self):
        scores = evaluate_networks('(OUT,(((A,B))#H1,#H1));', '(OUT,(A,B));')
        self.assertEqual(scores['hardwired_distance'], 0)
        self.assertEqual(scores['softwired_error'], 0)
        self.assertEqual(scores['true_hardwired_clusters'], 1)

    def test_default_outgroup_orientation_and_no_root_filter(self):
        first, second = '((A,B),(C,OUT));', '(A,(B,(C,OUT)));'
        self.assertEqual(evaluate_networks(first, second)['softwired_error'], 0)
        self.assertEqual(evaluate_networks(first, second)['hardwired_distance'], 0)
        self.assertGreater(evaluate_networks(first, second, outgroup=None)['hardwired_distance'], 0)
        arbitrary = FIXTURES[5]
        scores = evaluate_networks(ONE, arbitrary)
        self.assertIsInstance(scores['hardwired_distance'], int)
        self.assertTrue(all(value is not None for value in scores.values()))

    def test_invalid_taxa_missing_outgroup_and_empty_profiles(self):
        for metric in ('both', 'softwired', 'hardwired'):
            with self.assertRaisesRegex(ValueError, 'taxa differ'):
                evaluate_networks(TREE, '(OUT,((A,B),(C,E)));', metric=metric)
            with self.assertRaisesRegex(ValueError, 'Outgroup missing'):
                evaluate_networks(TREE, TREE, outgroup='missing', metric=metric)
        empty = evaluate_networks('(A,B,C);', '(A,B,C);', outgroup=None)
        self.assertEqual(empty['softwired_error'], 0)
        self.assertEqual(empty['hardwired_distance'], 0)
        self.assertEqual(empty['cluster_f1'], 1)
        with self.assertRaises(ValueError):
            evaluate_networks(TREE, TREE, metric='invalid')

    def test_many_blobs_use_local_budget_and_never_approximate(self):
        blobs = [f'((a{i},(b{i})#H{i}),(#H{i},c{i}))' for i in range(20)]
        network = '(OUT,('+','.join(blobs)+'));'
        summary = cluster_summary(network, cap=21, outgroup='OUT')
        self.assertEqual(summary.global_switchings, 2**20)
        self.assertEqual(summary.evaluated_switchings, 21)
        with self.assertRaisesRegex(ValueError, 'exceeds cap'):
            softwired_cluster_distance(network, network, cap=20, outgroup='OUT')
        for cap in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                cluster_summary(TREE, cap=cap)

    def test_hardwired_only_does_not_run_softwired_extraction(self):
        with patch('camus_nn.evaluate.cluster_summary', side_effect=AssertionError('SW was called')):
            result = evaluate_networks(ONE, TREE, metric='hardwired', cap=1)
        self.assertNotIn('softwired_fn', result)
        self.assertEqual(result, hardwired_cluster_distance(ONE, TREE, outgroup='OUT'))

    def test_cli_json_hashes_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth, estimate, output = root/'truth.nwk', root/'estimate.nwk', root/'scores.json'
            truth.write_text(TREE)
            estimate.write_text('(OUT,((A,B),C,D));')
            argv = ['--truth', str(truth), '--estimate', str(estimate), '--output', str(output)]
            text = StringIO()
            with redirect_stdout(text):
                main(argv)
            result = json.loads(text.getvalue())
            self.assertEqual(result, json.loads(output.read_text()))
            self.assertEqual(result['scores']['hardwired_distance'], 1)
            self.assertFalse(result['root_filter_applied'])
            self.assertEqual(result['inputs']['truth']['sha256'], hashlib.sha256(truth.read_bytes()).hexdigest())
            before = output.read_bytes()
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(argv)
            self.assertEqual(output.read_bytes(), before)
            run = subprocess.run([sys.executable, '-B', '-m', 'camus_nn.evaluate',
                '--truth', str(truth), '--estimate', str(estimate), '--metric', 'hardwired'],
                capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(run.stdout)['scores']['hardwired_distance'], 1)

    def test_cli_input_error_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth, estimate, output = root/'truth.nwk', root/'estimate.nwk', root/'scores.json'
            truth.write_text(TREE)
            estimate.write_text('(OUT,((A,B),(C,E)));')
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(['--truth', str(truth), '--estimate', str(estimate), '--output', str(output)])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
