"""Exact softwired and hardwired cluster-set evaluation of saved networks.

Rates are fractions; HWCD is a raw symmetric-difference count, without
percentage scaling or division by two. Branch lengths and inheritance
probabilities do not affect these topology metrics. No root filter is applied.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import networkx as nx

from .topology import parse_network


def precision(tp, fp):
    return tp / (tp + fp) if tp + fp else 0.0


def recall(tp, fn):
    return tp / (tp + fn) if tp + fn else 0.0


def f1(tp, fp, fn, *, empty=0.0):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else empty


@dataclass(frozen=True)
class ClusterSummary:
    """Distinct nontrivial clusters, ignoring lengths and inheritance weights."""

    taxa: frozenset[str]
    union: frozenset[frozenset[str]]
    invariant: frozenset[frozenset[str]]
    reticulations: int
    global_switchings: int
    evaluated_switchings: int
    max_blob_reticulations: int
    outgroup: str | None

    @property
    def variable(self):
        return self.union - self.invariant


def cluster_summary(text, cap=4096, *, outgroup=None):
    """Compute exact union/intersection across displayed trees.

    ``cap`` limits the configurations actually evaluated:
    ``1 + sum(2**h - 1 for each bridge component with h reticulations)``.
    ``global_switchings`` records the potentially much larger full product.
    No switchings are sampled. Parallel edges are collapsed, matching the
    existing softwired-cluster evaluator. Binary reticulations are required
    by the shared parser; tree multifurcations and unary vertices are allowed.

    With ``outgroup``, orient every displayed-tree split away from that taxon
    before forming the union and intersection. This is equivalent to rooting
    each displayed tree on the outgroup edge, and accepts arbitrary input
    Newick roots. Otherwise use the input root. Both modes include exactly
    clusters of size greater than one and less than the number of taxa.
    """
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise ValueError("cap must be a positive integer")
    graph = nx.DiGraph(parse_network(text, allow_parallel=True))
    taxa = sorted(graph.nodes[u]["label"] for u in graph if graph.out_degree(u) == 0)
    if outgroup is not None and outgroup not in taxa:
        raise ValueError("Outgroup missing from network")
    tips = {label: 1 << i for i, label in enumerate(taxa)}
    full_mask = (1 << len(taxa)) - 1
    outgroup_mask = tips[outgroup] if outgroup is not None else 0

    order = list(nx.topological_sort(graph))
    index = {u: i for i, u in enumerate(order)}
    hybrids = [u for u in order if graph.in_degree(u) == 2]
    hybrid_index = {u: i for i, u in enumerate(hybrids)}
    parents = {u: list(graph.predecessors(u)) for u in hybrids}
    groups = []
    assigned = set()
    undirected = graph.to_undirected()
    undirected.remove_edges_from(list(nx.bridges(undirected)))
    for component in nx.connected_components(undirected):
        group = [hybrid_index[u] for u in component
                 if u in hybrid_index and all(p in component for p in parents[u])]
        if group:
            if assigned.intersection(group):
                raise ValueError("Reticulation assigned to multiple bridge components")
            assigned.update(group)
            groups.append(sorted(group))
    if assigned != set(range(len(hybrids))):
        raise ValueError("Reticulation incoming edges do not share a bridge component")
    configurations = 1 + sum((1 << len(group)) - 1 for group in groups)
    if configurations > cap:
        raise ValueError(f"Exact cluster extraction needs {configurations} local "
                         f"switching configurations, exceeds cap {cap}")

    leaf_masks = [tips.get(graph.nodes[u].get("label"), 0)
                  if graph.out_degree(u) == 0 else 0 for u in order]
    children = [
        [(index[v], hybrid_index.get(v, -1), parents[v].index(u) if v in parents else 0)
         for v in graph.successors(u)]
        for u in order
    ]

    def switched_clusters(choices):
        masks = leaf_masks.copy()
        result = set()
        for u in range(len(order) - 1, -1, -1):
            for v, hybrid, parent in children[u]:
                if hybrid < 0 or choices[hybrid] == parent:
                    masks[u] |= masks[v]
            mask = masks[u]
            if mask & outgroup_mask:
                mask ^= full_mask
            if 1 < mask.bit_count() < len(taxa):
                result.add(mask)
        return result

    choices = [0] * len(hybrids)
    union = switched_clusters(choices)
    invariant = union.copy()
    for group in groups:
        for state in range(1, 1 << len(group)):
            for bit, hybrid in enumerate(group):
                choices[hybrid] = (state >> bit) & 1
            current = switched_clusters(choices)
            union.update(current)
            invariant.intersection_update(current)
        for hybrid in group:
            choices[hybrid] = 0

    def decode(masks):
        return frozenset(frozenset(label for i, label in enumerate(taxa) if mask & (1 << i))
                         for mask in masks)

    return ClusterSummary(
        taxa=frozenset(taxa), union=decode(union), invariant=decode(invariant),
        reticulations=len(hybrids), global_switchings=1 << len(hybrids),
        evaluated_switchings=configurations,
        max_blob_reticulations=max(map(len, groups), default=0), outgroup=outgroup,
    )


def compare_cluster_sets(target, predicted):
    """Score already-extracted cluster sets using the shared FP/FN convention.

    Callers must supply the same taxon universe and rooting convention. Cluster
    identifiers may be frozensets of taxa or consistently encoded bit masks.
    """
    tp = len(target & predicted)
    fn = len(target - predicted) / len(target) if target else 0.0
    fp = len(predicted - target) / len(predicted) if predicted else 0.0
    return dict(
        cluster_precision=precision(tp, len(predicted - target)),
        cluster_recall=recall(tp, len(target - predicted)),
        cluster_f1=f1(tp, len(predicted - target), len(target - predicted), empty=1.0),
        softwired_fn=fn,
        softwired_fp=fp,
        softwired_error=(fn + fp) / 2,
        true_clusters=len(target),
        inferred_clusters=len(predicted),
    )


def hardwired_clusters(text, *, outgroup=None):
    """Return taxa and distinct clusters without enumerating displayed trees.

    With an outgroup, orient each descendant bipartition away from that taxon,
    as in the existing softwired comparison. This does not reroot hybrid edges
    or minimize over alternative root placements.
    """
    graph = parse_network(text, allow_parallel=True)
    taxa = frozenset(graph.nodes[u]["label"] for u in graph if graph.out_degree(u) == 0)
    if outgroup is not None and outgroup not in taxa:
        raise ValueError("Outgroup missing from network")
    descendants = {}
    clusters = set()
    for u in reversed(list(nx.topological_sort(graph))):
        if graph.out_degree(u) == 0:
            leaves = frozenset([graph.nodes[u]["label"]])
        else:
            leaves = frozenset().union(*(descendants[v] for v in graph.successors(u)))
        descendants[u] = leaves
        if outgroup is not None and outgroup in leaves:
            leaves = taxa - leaves
        if 1 < len(leaves) < len(taxa):
            clusters.add(leaves)
    return taxa, frozenset(clusters)


def compare_hardwired_clusters(target, predicted):
    """Compare cluster sets on the same taxa and with the same orientation."""
    scores = compare_cluster_sets(target, predicted)
    for suffix in ("fn", "fp", "error"):
        scores["hardwired_" + suffix] = scores.pop("softwired_" + suffix)
    for suffix in ("precision", "recall", "f1"):
        scores["hardwired_" + suffix] = scores.pop("cluster_" + suffix)
    scores["true_hardwired_clusters"] = scores.pop("true_clusters")
    scores["inferred_hardwired_clusters"] = scores.pop("inferred_clusters")
    scores["hardwired_tp"] = len(target & predicted)
    scores["hardwired_fn_count"] = len(target - predicted)
    scores["hardwired_fp_count"] = len(predicted - target)
    scores["hardwired_distance"] = len(target ^ predicted)
    return scores


def hardwired_cluster_distance(truth, estimate, *, outgroup=None):
    taxa, target = hardwired_clusters(truth, outgroup=outgroup)
    other_taxa, predicted = hardwired_clusters(estimate, outgroup=outgroup)
    if taxa != other_taxa:
        raise ValueError("Inferred and true network taxa differ")
    return compare_hardwired_clusters(target, predicted)


def softwired_cluster_distance(truth, estimate, cap=65536, *, outgroup=None):
    """Exact softwired FN/FP/error for two extended-Newick strings.

    ``outgroup=None`` uses the supplied roots. Set ``outgroup="OUT"`` to match
    the outgroup-oriented experiment scores. Exact switching enumeration is
    factored by bridge components; exceeding ``cap`` raises an error.
    """
    target = cluster_summary(truth, cap, outgroup=outgroup)
    predicted = cluster_summary(estimate, cap, outgroup=outgroup)
    if target.taxa != predicted.taxa:
        raise ValueError("Inferred and true network taxa differ")
    return compare_cluster_sets(target.union, predicted.union)


def evaluate_networks(truth, estimate, *, outgroup="OUT", cap=65536, metric="both"):
    """Return softwired rates and/or raw hardwired distance on the same taxa.

    Input values are single extended-Newick strings. ``metric`` is ``both``,
    ``softwired`` or ``hardwired``. Pass ``outgroup=None`` for supplied roots.
    The default OUT-oriented cluster-set convention matches the experiment
    reports; it does not reroot hybrid edges or optimize root placements.
    """
    if metric not in ("both", "softwired", "hardwired"):
        raise ValueError("metric must be both, softwired or hardwired")
    scores = {}
    if metric in ("both", "softwired"):
        scores.update(softwired_cluster_distance(truth, estimate, cap, outgroup=outgroup))
    if metric in ("both", "hardwired"):
        scores.update(hardwired_cluster_distance(truth, estimate, outgroup=outgroup))
    return scores


def _positive(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare saved networks with Python cluster metrics, PhyloNet SW, or PhyloNetworks HWCD.")
    parser.add_argument("--truth", required=True, type=Path, help="True network, one extended-Newick record")
    parser.add_argument("--estimate", required=True, type=Path, help="Estimated network, one extended-Newick record")
    parser.add_argument("--metric", choices=("both", "softwired", "hardwired", "phylonet-sw", "phylonetworks-hwcd"), default="both")
    rooting = parser.add_mutually_exclusive_group()
    rooting.add_argument("--outgroup", help="Python metrics: orient clusters away from this taxon (default: OUT); unavailable for external backends")
    rooting.add_argument("--rooted", action="store_true", help="Use supplied roots; PhyloNetworks otherwise uses semidirected comparison; PhyloNet always uses supplied roots")
    parser.add_argument("--max-switchings", type=_positive,
        help="Python SW: exact local configuration budget per network (default: 65536)")
    parser.add_argument("--output", type=Path, help="Also save JSON to this new file; existing files are never overwritten")
    sw = parser.add_argument_group("optional PhyloNet SW backend")
    sw.add_argument("--sw-jar", type=Path, help="Pinned PhyloNet 3.8.2 JAR (default: vendor/PhyloNet-3.8.2.jar)")
    sw.add_argument("--sw-java", help="Java executable (default: java on PATH)")
    sw.add_argument("--sw-heap", help="Java heap, such as 2g (default: 2g)")
    sw.add_argument("--sw-timeout", type=float, help="Seconds per comparison (default: 3600)")
    hw = parser.add_argument_group("optional PhyloNetworks HWCD backend")
    hw.add_argument("--hwcd-julia", help="Julia executable (default: julia on PATH)")
    hw.add_argument("--hwcd-project", type=Path, help="Existing Julia environment with PhyloNetworks 1.4.0 (default: active Julia environment)")
    hw.add_argument("--hwcd-depot", type=Path, help="Existing Julia package depot (default: Julia's configured depot)")
    hw.add_argument("--hwcd-timeout", type=float, help="Seconds per comparison (default: 3600)")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("output already exists: " + str(args.output))
    external = args.metric in ("phylonet-sw", "phylonetworks-hwcd")
    if external and args.output is None:
        parser.error("External backends require --output to retain scoring logs and provenance beside the JSON")
    if external and args.outgroup is not None:
        parser.error("--outgroup applies only to Python cluster metrics; external backends use their own rooting convention")
    if external and args.max_switchings is not None:
        parser.error("--max-switchings applies only to Python softwired extraction")
    if args.metric != "phylonet-sw" and any(v is not None for v in (args.sw_jar, args.sw_java, args.sw_heap, args.sw_timeout)):
        parser.error("--sw-* options require --metric phylonet-sw")
    if args.metric != "phylonetworks-hwcd" and any(v is not None for v in (args.hwcd_julia, args.hwcd_project, args.hwcd_depot, args.hwcd_timeout)):
        parser.error("--hwcd-* options require --metric phylonetworks-hwcd")
    outgroup = None if args.rooted or external else ("OUT" if args.outgroup is None else args.outgroup)
    cap = None if external else (65536 if args.max_switchings is None else args.max_switchings)
    try:
        inputs = {}
        texts = {}
        for label, path in (("truth", args.truth), ("estimate", args.estimate)):
            path = path.resolve()
            raw = path.read_bytes()
            texts[label] = raw.decode("utf-8")
            inputs[label] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
        if args.metric == "phylonet-sw":
            from .phylonet_sw import PhyloNetSW, METRIC_VERSION
            workdir = args.output.resolve().parent / (args.output.stem + "-phylonet-sw")
            with PhyloNetSW(workdir=workdir, jar=args.sw_jar, java=args.sw_java,
                            heap="2g" if args.sw_heap is None else args.sw_heap,
                            timeout=3600 if args.sw_timeout is None else args.sw_timeout) as scorer:
                scores = scorer.score(texts["truth"], texts["estimate"])
            version = METRIC_VERSION
            hwcd_units = None
        elif args.metric == "phylonetworks-hwcd":
            from .phylonetworks_hwcd import PhyloNetworksHWCD, METRIC_VERSIONS
            workdir = args.output.resolve().parent / (args.output.stem + "-hwcd")
            with PhyloNetworksHWCD(workdir=workdir, julia=args.hwcd_julia,
                                  julia_project=args.hwcd_project, depot=args.hwcd_depot,
                                  timeout=3600 if args.hwcd_timeout is None else args.hwcd_timeout) as scorer:
                scores = scorer.score(texts["truth"], texts["estimate"], rooted=args.rooted)
            version = METRIC_VERSIONS[args.rooted]
            hwcd_units = "raw edge-typed cluster differences with multiplicity"
        else:
            scores = evaluate_networks(texts["truth"], texts["estimate"], outgroup=outgroup,
                                       cap=cap, metric=args.metric)
            version = "python-cluster-set/v1-" + ("rooted" if outgroup is None else "reference-taxon")
            hwcd_units = "raw distinct-cluster differences"
        parameters = (json.loads(scores.pop("metric_parameters_json")) if external else
                      dict(implementation="camus_nn.evaluate", outgroup=outgroup,
                           source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        result = dict(schema="camus-nn-evaluation/v1", inputs=inputs, metric=args.metric,
            metric_version=version, metric_parameters=parameters,
            outgroup=outgroup, max_switchings=cap,
            rate_units=None if args.metric == "phylonetworks-hwcd" else "fraction", hwcd_units=hwcd_units,
            root_filter_applied=False, scores=scores)
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                stream.write(rendered)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(rendered, end="")


if __name__ == "__main__":
    main()
