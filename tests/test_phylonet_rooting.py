"""Rooting must alter rooted triplets while preserving unrooted resolutions."""
import re
import unittest
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace

from camus_nn.topology import Tree

from camus_nn.rooting import root_on_outgroup, tree_graph, unrooted_splits


def independent_splits(text):
    tree = Tree.from_newick(text)
    taxa = frozenset(tree.tips)
    splits = set()
    for node in tree.preorder:
        a = frozenset(tree.labels[u] for u in tree.desc[node] if not tree.children[u])
        b = taxa - a
        if min(len(a), len(b)) > 1:
            splits.add(min(tuple(sorted(a)), tuple(sorted(b))))
    return taxa, splits


class PhyloNetRootingTests(unittest.TestCase):
    def assert_out_root(self, text, outgroup='OUT'):
        graph = tree_graph(text)
        root = graph.graph['root']
        self.assertEqual(graph.out_degree(root), 2)
        self.assertTrue(any(graph.nodes[u]['label'] == outgroup
                            and graph.out_degree(u) == 0 for u in graph.successors(root)))

    def test_arbitrary_root_and_annotated_genes_are_corrected(self):
        for text in ['(a:1,OUT:2,(b:3,c:4)95:5);',
                     '(a,(OUT,(b,c)));', '(a,(b,(OUT,c)));',
                     '(OUT,a,b,c);', "('O''Brien',('a b',OUT),c);"]:
            rooted = root_on_outgroup(text)
            self.assert_out_root(rooted)
            self.assertEqual(independent_splits(text), independent_splits(rooted))
            self.assertEqual(unrooted_splits(text), unrooted_splits(rooted))
            self.assertNotIn(':', rooted)

    def test_rooted_triplet_changes_to_the_outgroup_orientation(self):
        self.assertEqual(root_on_outgroup('(a,OUT,b);'), '(OUT,(a,b));')

    def test_rooting_an_already_rooted_tree_preserves_rooted_clades(self):
        text = '(OUT,((a,b),(c,d)));'
        rooted = root_on_outgroup(text)
        a = tree_graph(text); b = tree_graph(rooted)
        import networkx as nx
        clades = lambda g: {frozenset(g.nodes[v]['label'] for v in nx.descendants(g, u)
                                     if g.out_degree(v) == 0) for u in g if g.out_degree(u)}
        self.assertEqual(clades(a), clades(b))

    def test_custom_outgroup(self):
        self.assert_out_root(root_on_outgroup('(a,O,b);', 'O'), 'O')

    def test_missing_outgroup_and_taxon_mismatch_fail(self):
        for text in ['(a,b,c);', '(OUT,a,a);', '(OUT,(a)#H1,(#H1,b));']:
            with self.assertRaises(ValueError):
                root_on_outgroup(text)
        with self.assertRaises(ValueError):
            root_on_outgroup('(OUT,a,b);', expected_taxa={'OUT','a','c'})

    def test_standalone_wrapper_saves_and_passes_rooted_copies(self):
        source = Path(__file__).resolve().parents[1] / 'misc/phylonet_mpl.py'
        spec = importlib.util.spec_from_file_location('standalone_mpl', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            genes = directory / 'genes.nwk'
            guide = directory / 'guide.nwk'
            genes.write_text('(a,OUT,(b,c));\n(b,(OUT,a),c);\n')
            guide.write_text('(c,OUT,(a,b));\n')
            original = genes.read_bytes(), guide.read_bytes()
            args = SimpleNamespace(gene_trees=genes, starting_tree=guide, outgroup='OUT',
                max_reticulations=2, threads=8, seed=42, runs=10)
            destination = directory / 'input.nex'
            module.write_nexus_input(args, destination)
            trees = re.findall(r'^(?:Tree gt\d+|Network st) = (.+)$', destination.read_text(), re.M)
            self.assertEqual(len(trees), 3)
            for tree in trees:
                self.assert_out_root(tree)
            self.assertEqual((genes.read_bytes(),guide.read_bytes()), original)
            self.assertEqual(len((directory/'genes-rooted.nwk').read_text().splitlines()),2)
            self.assertIn('-pl 8 -rs 42 -x 10;', destination.read_text())


if __name__ == '__main__':
    unittest.main()
