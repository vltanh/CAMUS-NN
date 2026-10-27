"""All, Rooted, Uniform, EU and EW quartet sampling."""
from abc import ABC, abstractmethod
from .topology import Tree
from itertools import combinations
from math import comb
import random


class QuartetSampler(ABC):
    @abstractmethod
    def sample(self, tree: Tree, count: int, seed: int) -> list[tuple[str, ...]]:
        """Sample four-taxon sets without inspecting gene frequencies or truth."""


def edge_count(tree, include_leaf_edges=True):
    """Count unrooted binary edges after suppressing degree-two roots."""
    graph = tree.graph().to_undirected()
    for node in list(graph):
        if graph.degree(node) == 2:
            a, b = list(graph.neighbors(node))
            graph.remove_node(node)
            graph.add_edge(a, b)
    if len(tree.tips) < 4 or any(graph.degree(v) not in (1, 3) for v in graph):
        raise ValueError("Multiplicity sampling requires an unrooted binary tree with >=4 tips")
    return sum(include_leaf_edges or (graph.degree(u)>1 and graph.degree(v)>1)
               for u,v in graph.edges())


class AllQuartetSampler(QuartetSampler):
    def sample(self, tree, count, seed):
        return list(combinations(sorted(tree.tips), 4))


class RootedQuartetSampler(QuartetSampler):
    def sample(self, tree, count, seed):
        names = sorted(tree.tips)
        if "OUT" not in names:
            raise ValueError("Rooted sampling requires a tip named OUT")
        ingroup = [name for name in names if name != "OUT"]
        if len(ingroup) < 3:
            raise ValueError("Rooted sampling requires at least three ingroup tips")
        return [tuple(sorted(("OUT", *triple))) for triple in combinations(ingroup, 3)]


class UniformQuartetSampler(QuartetSampler):
    def sample(self, tree, count, seed, *, replacement=False):
        names = sorted(tree.tips)
        total = comb(len(names), 4)
        if count < 1:
            raise ValueError("Sample size must be positive")
        # Sample combination ranks: no list of all n-choose-4 subsets is built.
        rng = random.Random(seed)
        ranks = [rng.randrange(total) for _ in range(count)] if replacement else rng.sample(range(total), min(count,total))
        result = []
        for rank in ranks:
            indices, start = [], 0
            for remaining in range(3, -1, -1):
                for i in range(start, len(names) - remaining):
                    block = comb(len(names) - i - 1, remaining)
                    if rank < block:
                        indices.append(i)
                        start = i + 1
                        break
                    rank -= block
            result.append(tuple(names[i] for i in indices))
        return result


class EdgeQuartetSampler(QuartetSampler):
    """Visit every edge in shuffled rounds; retain unique four-taxon sets."""

    uniform_components = False

    def __init__(self, include_leaf_edges=False):
        self.include_leaf_edges = include_leaf_edges

    def candidates(self, graph, node, parent, leaves):
        return leaves(node, parent)

    @abstractmethod
    def choose(self, graph, node, parent, forwards, candidates, rng):
        """Choose one leaf from an outward component."""

    def sample(self, tree, count, seed, *, multiplicity=None):
        if multiplicity is not None and (not isinstance(multiplicity,int) or multiplicity < 1):
            raise ValueError("Multiplicity must be a positive integer")
        if count < 1:
            raise ValueError("Sample size must be positive")
        graph = tree.graph().to_undirected()
        # Remove the artificial root (and any other degree-two vertices).
        for node in list(graph):
            if graph.degree(node) == 2:
                a, b = list(graph.neighbors(node))
                graph.remove_node(node)
                graph.add_edge(a, b)
        if any(graph.degree(u) not in (1, 3) for u in graph):
            raise ValueError("Edge sampling requires an unrooted binary tree")

        from functools import lru_cache
        from math import prod
        import warnings

        @lru_cache(None)
        def leaves(node, parent):
            if graph.degree(node) == 1:
                return (graph.nodes[node]["label"],)
            return tuple(sorted(x for v in graph.neighbors(node) if v != parent
                                for x in leaves(v, node)))

        def forwards(node, parent):
            return sorted((v for v in graph.neighbors(node) if v != parent),
                          key=lambda v: leaves(v, node))

        if len(tree.tips) < 4:
            raise ValueError("At least four taxa are required")
        pools = {}
        edges = []
        for u, v in graph.edges():
            key = tuple(sorted((leaves(u, v), leaves(v, u))))
            fixed = None
            if graph.degree(u) == 1 or graph.degree(v) == 1:
                if not self.include_leaf_edges:
                    continue
                leaf, parent = (u, v) if graph.degree(u) == 1 else (v, u)
                fixed = graph.nodes[leaf]["label"]
                branches = [(w, parent) for w in forwards(parent, leaf)]
            else:
                if leaves(u, v) > leaves(v, u):
                    u, v = v, u
                branches = [(w, u) for w in forwards(u, v)] + [(w, v) for w in forwards(v, u)]
            for w, parent in branches:
                pools[w, parent] = self.candidates(graph, w, parent, leaves)
            sizes = [len(pools[b]) for b in branches]
            capacity = (comb(sizes[0], 2)*sizes[1] + sizes[0]*comb(sizes[1], 2)
                        if fixed is not None else prod(sizes))
            edges.append((key, fixed, branches, capacity))
        edges.sort(key=lambda e: e[0])
        target = min(count, comb(len(tree.tips), 4))
        quotas = [multiplicity if multiplicity is not None else e[3] for e in edges]
        rng = random.Random(seed)
        seen, result = set(), []
        # A quartet can now be produced by several edges. Track edge-local
        # exhaustion separately from global duplicate removal.
        drawn = [set() for _ in edges]
        edge_draws = [0 for _ in edges]
        attempts = 0
        while (multiplicity is not None or len(result) < target) and attempts < max(10000, count * 1000):
            order = [i for i, e in enumerate(edges) if (edge_draws[i] if multiplicity is not None else len(drawn[i])) < quotas[i]]
            if not order:
                break
            rng.shuffle(order)
            for i in order:
                _, fixed, branches, _ = edges[i]
                taxa = [] if fixed is None else [fixed]
                for node, parent in branches:
                    taxa.append(self.choose(graph, node, parent, forwards,
                                            pools[node, parent], rng))
                if fixed is not None:
                    # One from each part, then a second from an eligible part.
                    eligible = [b for b in branches if len(pools[b]) > 1]
                    if multiplicity is not None and self.uniform_components:
                        # Weight the doubled side by its number of possible pairs.
                        node, parent = rng.choices(eligible, weights=[len(pools[b])-1 for b in eligible])[0]
                    else:
                        node, parent = rng.choice(eligible)
                    candidates = tuple(x for x in pools[node, parent] if x not in taxa)
                    # Exclude the first leaf by pruning empty forward branches.
                    # At each remaining fork, edge-walk makes a fair choice.
                    allowed = set(candidates)
                    def available(cur, prev):
                        return [w for w in forwards(cur, prev) if allowed.intersection(leaves(w, cur))]
                    taxa.append(self.choose(graph, node, parent, available, candidates, rng))
                quartet = tuple(sorted(taxa))
                attempts += 1
                drawn[i].add(quartet)
                edge_draws[i] += 1
                if multiplicity is not None or quartet not in seen:
                    seen.add(quartet)
                    result.append(quartet)
                if (multiplicity is None and len(result) == target) or attempts >= max(10000, count * 1000):
                    break
        if (multiplicity is not None or len(result) < target) and any((edge_draws[i] if multiplicity is not None else len(drawn[i])) < quotas[i] for i, e in enumerate(edges)):
            raise ValueError(f"Edge sampler draw budget exhausted: {len(result)}/{target} unique quartets")
        self.last_edge_coverage = [dict(split=e[0], capacity=e[3], sampled=edge_draws[i] if multiplicity is not None else len(drawn[i]), unique=len(drawn[i])) for i,e in enumerate(edges)]
        if multiplicity is None and len(result) < count:
            warnings.warn(f"Edge sampler supports only {len(result)} unique quartets; "
                          f"requested {count}, returning {len(result)}", stacklevel=2)
        return result


class EdgeUniformQuartetSampler(EdgeQuartetSampler):
    """Uniform leaf choice within each surrounding component."""

    uniform_components = True

    def choose(self, graph, node, parent, forwards, candidates, rng):
        return rng.choice(candidates)


class EdgeWalkQuartetSampler(EdgeQuartetSampler):
    """Fair outward branch choices favor nearby leaves without measuring distance."""

    def choose(self, graph, node, parent, forwards, candidates, rng):
        while graph.degree(node) != 1:
            parent, node = node, rng.choice(forwards(node, parent))
        return graph.nodes[node]["label"]


SAMPLERS = {
    "all": AllQuartetSampler,
    "rooted": RootedQuartetSampler,
    "uniform": UniformQuartetSampler,
    "edge-uniform": EdgeUniformQuartetSampler,
    "edge-walk": EdgeWalkQuartetSampler,
}


def make_sampler(strategy, include_leaf_edges=False):
    if strategy not in SAMPLERS:
        raise ValueError(f"Unknown quartet sampler: {strategy}")
    cls = SAMPLERS[strategy]
    return cls(include_leaf_edges=include_leaf_edges) if issubclass(cls, EdgeQuartetSampler) else cls()


def sample_quartets(tree, strategy="uniform", seed=42, *, multiplicity=50,
                    count=None, include_leaf_edges=True):
    """Sample by edge multiplicity or by a fixed total count."""
    sampler = make_sampler(strategy, include_leaf_edges=include_leaf_edges)
    if count is not None:
        if multiplicity is not None:
            raise ValueError("Choose multiplicity or count, not both")
        return sampler.sample(tree, count, seed)
    if not isinstance(multiplicity, int) or isinstance(multiplicity, bool) or multiplicity < 1:
        raise ValueError("Multiplicity must be a positive integer")
    if strategy == "all":
        raise ValueError("Use count=0, multiplicity=None for exhaustive sampling")
    if strategy == "rooted":
        return sampler.sample(tree, 0, seed)
    budget = multiplicity * edge_count(tree, include_leaf_edges)
    if isinstance(sampler, EdgeQuartetSampler):
        return sampler.sample(tree, budget, seed, multiplicity=multiplicity)
    return sampler.sample(tree, budget, seed, replacement=True)
