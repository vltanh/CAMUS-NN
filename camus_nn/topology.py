"""Newick topology I/O for guides and switching proposals.

The Newick parser accepts ordinary and extended Newick (#H... / #R... labels),
quoted labels, branch lengths, inheritance fields, and bracket comments.
Branch lengths do not determine displayed quartet *topologies*.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import math
import re
import networkx as nx
import numpy as np

Quartet = tuple[str, str, str, str]


def parse_network(text: str, *, allow_parallel: bool = False) -> nx.DiGraph:
    text = text.strip()
    graph = nx.MultiDiGraph() if allow_parallel else nx.DiGraph()
    hybrids: dict[str, int] = {}
    pos = 0

    def skip() -> None:
        nonlocal pos
        while pos < len(text):
            if text[pos].isspace():
                pos += 1
            elif text[pos] == "[":
                depth = 1
                pos += 1
                while pos < len(text) and depth:
                    depth += (text[pos] == "[") - (text[pos] == "]")
                    pos += 1
                if depth:
                    raise ValueError("Unterminated Newick comment")
            else:
                break

    def label() -> str:
        nonlocal pos
        skip()
        if pos < len(text) and text[pos] == "'":
            pos += 1
            chars = []
            while pos < len(text):
                if text[pos] == "'":
                    pos += 1
                    if pos < len(text) and text[pos] == "'":
                        chars.append("'")
                        pos += 1
                    else:
                        return "".join(chars)
                else:
                    chars.append(text[pos])
                    pos += 1
            raise ValueError("Unterminated quoted label")
        start = pos
        while (
            pos < len(text) and text[pos] not in ",():;[]" and not text[pos].isspace()
        ):
            pos += 1
        return text[start:pos]

    def node() -> tuple[int, dict]:
        nonlocal pos
        skip()
        kids = []
        if pos < len(text) and text[pos] == "(":
            pos += 1
            kids.append(node())
            skip()
            while pos < len(text) and text[pos] == ",":
                pos += 1
                kids.append(node())
                skip()
            if pos >= len(text) or text[pos] != ")":
                raise ValueError(f"Expected ')' at character {pos}")
            pos += 1
        name = label()
        match = re.search(r"#[A-Za-z]+[A-Za-z0-9_]*", name)
        tag = match.group(0) if match else None
        if tag and tag in hybrids:
            u = hybrids[tag]
            if kids and graph.out_degree(u):
                raise ValueError(f"Hybrid {tag} is defined more than once")
        else:
            u = len(graph)
            graph.add_node(u, label=None if tag or kids else name, hybrid=tag)
            if tag:
                hybrids[tag] = u
            if not kids and not tag and not name:
                raise ValueError("Empty leaf label")
        for v, attrs in kids:
            if graph.has_edge(u, v) and not allow_parallel:
                raise ValueError("Parallel edges are not supported")
            graph.add_edge(u, v, **attrs)
        skip()
        attrs = {}
        if pos < len(text) and text[pos] == ":":
            vals = []
            while pos < len(text) and text[pos] == ":":
                pos += 1
                start = pos
                while (
                    pos < len(text)
                    and text[pos] not in ":,);["
                    and not text[pos].isspace()
                ):
                    pos += 1
                word = text[start:pos]
                vals.append(float(word) if word else None)
                skip()
            if vals and vals[0] is not None:
                attrs["length"] = vals[0]
            if len(vals) >= 3 and vals[2] is not None:
                attrs["gamma"] = vals[2]
        return u, attrs

    root, _ = node()
    skip()
    if pos >= len(text) or text[pos] != ";":
        raise ValueError("Missing terminal ';'")
    pos += 1
    skip()
    if pos != len(text):
        raise ValueError("Expected exactly one Newick network")
    graph.graph["root"] = root
    validate_network(graph)
    return graph


def validate_network(g: nx.DiGraph, level1: bool = False) -> None:
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Network contains a directed cycle")
    roots = [u for u in g if g.in_degree(u) == 0]
    if len(roots) != 1 or len(nx.descendants(g, roots[0])) != len(g) - 1:
        raise ValueError("Expected one root reaching every node")
    tips = [g.nodes[u].get("label") for u in g if g.out_degree(u) == 0]
    if any(not x for x in tips) or len(tips) != len(set(tips)):
        raise ValueError("Leaf labels must be nonempty and unique")
    if any(g.in_degree(u) > 2 for u in g):
        raise ValueError("Only indegree <= 2 is supported")
    if level1:
        cycles = nx.cycle_basis(g.to_undirected())
        if any(set(a) & set(b) for a, b in combinations(cycles, 2)):
            raise ValueError("Cycles are not vertex-disjoint (not level-1)")


def quote_label(s: str) -> str:
    return (
        s if re.fullmatch(r"[A-Za-z0-9_.-]+", s) else "'" + s.replace("'", "''") + "'"
    )


def network_newick(g: nx.DiGraph, lengths: bool = False) -> str:
    root = next(u for u in g if g.in_degree(u) == 0)
    hybrids = {
        u: f"#H{i + 1}" for i, u in enumerate(u for u in g if g.in_degree(u) == 2)
    }
    visited = set()

    def emit(u: int, parent: int | None = None, edge_key=None) -> str:
        if u in hybrids and u in visited:
            value = hybrids[u]
        else:
            visited.add(u)
            kids = (
                [(v, key) for _, v, key in g.out_edges(u, keys=True)]
                if g.is_multigraph()
                else [(v, None) for v in g.successors(u)]
            )
            value = (
                "(" + ",".join(emit(v, u, key) for v, key in kids) + ")"
                if kids
                else quote_label(g.nodes[u]["label"])
            )
            if u in hybrids:
                value += hybrids[u]
        if lengths and parent is not None:
            attrs = (
                g.edges[parent, u, edge_key]
                if g.is_multigraph()
                else g.edges[parent, u]
            )
            value += ":" + format(attrs.get("length", 0.0), ".10g")
            if "gamma" in attrs:
                value += "::" + format(attrs["gamma"], ".10g")
        return value

    return emit(root) + ";"


@dataclass
class Tree:
    parent: list[int]
    labels: list[str | None]
    lengths: list[float]

    def __post_init__(self) -> None:
        self.children = [[] for _ in self.parent]
        for u, p in enumerate(self.parent):
            if p >= 0:
                self.children[p].append(u)
        self.root = self.parent.index(-1)
        self.depth = [0] * len(self.parent)
        self.preorder = []

        def walk(u: int) -> None:
            self.preorder.append(u)
            for v in self.children[u]:
                self.depth[v] = self.depth[u] + 1
                walk(v)

        walk(self.root)
        self.tips = {self.labels[u]: u for u in self.preorder if not self.children[u]}
        if None in self.tips or len(self.tips) != sum(not x for x in self.children):
            raise ValueError("Missing or duplicate tree leaf labels")
        self.desc = [set() for _ in self.parent]
        for u in reversed(self.preorder):
            self.desc[u] = {u}.union(*(self.desc[c] for c in self.children[u]))
        self._dist = None

    @classmethod
    def from_graph(cls, g: nx.DiGraph, suppress_unary: bool = False) -> "Tree":
        if any(g.in_degree(u) > 1 for u in g):
            raise ValueError("Expected a tree, not an unswitched network")
        h = g.copy()
        # Prune unlabeled dead ends created when a hybrid incoming edge is removed.
        while True:
            dead = [
                u for u in h if h.out_degree(u) == 0 and not h.nodes[u].get("label")
            ]
            if not dead:
                break
            h.remove_nodes_from(dead)
        if suppress_unary:
            changed = True
            while changed:
                changed = False
                for u in list(h):
                    if h.out_degree(u) == 1:
                        v = next(h.successors(u))
                        parents = list(h.predecessors(u))
                        if parents:
                            p = parents[0]
                            length = h.edges[p, u].get("length", 0) + h.edges[u, v].get(
                                "length", 0
                            )
                            h.add_edge(p, v, length=length)
                        h.remove_node(u)
                        changed = True
                        break
        root = next(u for u in h if h.in_degree(u) == 0)
        order = list(nx.dfs_preorder_nodes(h, root))
        index = {u: i for i, u in enumerate(order)}
        parent, labels, lengths = [], [], []
        for u in order:
            p = next(h.predecessors(u), None)
            parent.append(-1 if p is None else index[p])
            labels.append(h.nodes[u].get("label"))
            lengths.append(0.0 if p is None else h.edges[p, u].get("length", 0.0))
        return cls(parent, labels, lengths)

    @classmethod
    def from_newick(cls, text: str, binary: bool = False) -> "Tree":
        t = cls.from_graph(parse_network(text), suppress_unary=True)
        if binary and any(len(c) not in (0, 2) for c in t.children):
            raise ValueError("Constraint tree must be rooted and binary")
        return t

    def graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for u in self.preorder:
            g.add_node(u, label=self.labels[u] if not self.children[u] else None)
            if self.parent[u] >= 0:
                g.add_edge(self.parent[u], u, length=self.lengths[u])
        g.graph["root"] = self.root
        return g

    def newick(self, lengths: bool = False) -> str:
        return network_newick(self.graph(), lengths)

    def lca(self, u: int, v: int) -> int:
        while self.depth[u] > self.depth[v]:
            u = self.parent[u]
        while self.depth[v] > self.depth[u]:
            v = self.parent[v]
        while u != v:
            u, v = self.parent[u], self.parent[v]
        return u

    def sibling(self, u: int) -> int:
        a, b = self.children[self.parent[u]]
        return b if a == u else a

    def quartet_topologies(self, quartets: list[Quartet]) -> np.ndarray:
        """0/1/2 topology; -1 unresolved; -2 missing taxon. Unweighted distances."""
        if self._dist is None:
            adj = [[] for _ in self.parent]
            for u, p in enumerate(self.parent):
                if p >= 0:
                    adj[u].append(p)
                    adj[p].append(u)
            self._dist = np.zeros((len(self.parent), len(self.parent)), dtype=np.int32)
            for a in self.tips.values():
                d = {a: 0}
                todo = [a]
                for u in todo:
                    for v in adj[u]:
                        if v not in d:
                            d[v] = d[u] + 1
                            todo.append(v)
                for b, value in d.items():
                    self._dist[a, b] = value
        result = np.full(len(quartets), -2, dtype=np.int8)
        present = [
            (i, [self.tips[x] for x in q])
            for i, q in enumerate(quartets)
            if all(x in self.tips for x in q)
        ]
        if present:
            ids = np.array([i for i, _ in present])
            a, b, c, d = np.array([v for _, v in present]).T
            ds = self._dist
            sums = np.column_stack(
                (ds[a, b] + ds[c, d], ds[a, c] + ds[b, d], ds[a, d] + ds[b, c])
            )
            best = sums.argmin(axis=1).astype(np.int8)
            best[(sums == sums.min(axis=1, keepdims=True)).sum(axis=1) != 1] = -1
            result[ids] = best
        return result
