"""Prepare rooted topology-only inputs for PhyloNet's rooted-triplet MPL."""

import networkx as nx

from .topology import network_newick, parse_network


def tree_graph(text):
    graph = parse_network(text)
    if any(graph.in_degree(u) > 1 or graph.nodes[u].get("hybrid") for u in graph):
        raise ValueError("MPL input must be a gene tree, not a network")
    if not nx.is_tree(graph.to_undirected()):
        raise ValueError("MPL input must be a tree")
    return graph


def unrooted_splits(text):
    """Return leaf identities and nontrivial splits, independently of the root."""
    graph = tree_graph(text)
    taxa = frozenset(graph.nodes[u]["label"] for u in graph if graph.out_degree(u) == 0)
    descendants, splits = {}, set()
    for u in reversed(list(nx.topological_sort(graph))):
        descendants[u] = (frozenset([graph.nodes[u]["label"]]) if graph.out_degree(u) == 0
                          else frozenset().union(*(descendants[v] for v in graph.successors(u))))
        a, b = descendants[u], taxa - descendants[u]
        if min(len(a), len(b)) > 1:
            splits.add(min(tuple(sorted(a)), tuple(sorted(b))))
    return taxa, frozenset(splits)


def root_on_outgroup(text, outgroup="OUT", expected_taxa=None):
    """Root on the outgroup's incident edge; retain every unrooted resolution.

    MPL uses topologies only. Lengths, supports and internal labels are omitted;
    the outgroup remains a taxon. Missing outgroups/taxon mismatches are errors.
    """
    if not isinstance(outgroup, str) or not outgroup:
        raise ValueError("MPL outgroup must be a nonempty taxon label")
    graph = tree_graph(text)
    tips = {graph.nodes[u]["label"]: u for u in graph if graph.out_degree(u) == 0}
    if outgroup not in tips:
        raise ValueError(f"MPL outgroup {outgroup!r} is missing")
    if len(tips) < 3:
        raise ValueError("MPL requires at least three taxa")
    if expected_taxa is not None and set(tips) != set(expected_taxa):
        raise ValueError("MPL gene-tree taxa differ from the starting-tree taxa")
    tree = graph.to_undirected()
    for u in list(tree):
        if tree.nodes[u].get("label") is None and tree.degree(u) == 2:
            a, b = list(tree.neighbors(u))
            tree.remove_node(u)
            tree.add_edge(a, b)
    if any(tree.degree(u) == 1 and tree.nodes[u].get("label") is None for u in tree):
        raise ValueError("MPL input contains a dangling unlabeled root")
    tip = tips[outgroup]
    neighbor = next(tree.neighbors(tip))
    root = max(tree) + 1
    tree.remove_edge(tip, neighbor)
    tree.add_node(root, label=None, hybrid=None)
    tree.add_edges_from([(root, tip), (root, neighbor)])
    rooted = nx.DiGraph()
    rooted.add_nodes_from(tree.nodes(data=True))
    rooted.add_edges_from(nx.dfs_edges(tree, source=root))
    rooted.graph["root"] = root
    result = network_newick(rooted)
    if unrooted_splits(text) != unrooted_splits(result):
        raise ValueError("MPL rooting changed the unrooted topology")
    return result
