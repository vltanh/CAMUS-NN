"""Deterministic single-cycle switching with fixed learned quartet evidence."""
import json
import math
from pathlib import Path
import shutil

from .topology import Tree, parse_network, quote_label, validate_network
from .io import digest, save


def gain_tolerance(a, b):
    return max(1e-7, 1e-12 * max(abs(a), abs(b)))


def canonical_tree(tree):
    """Rooted topology identity, ignoring child order and all annotations."""
    if isinstance(tree, str):
        tree = Tree.from_newick(tree, binary=True)
    def emit(u):
        if len(tree.children[u]) == 1:
            return emit(tree.children[u][0])
        return ('(' + ','.join(sorted(emit(v) for v in tree.children[u])) + ')'
                if tree.children[u] else quote_label(tree.labels[u]))
    return emit(tree.root) + ';'


def anchored_tree(text, taxa=None, *, anchor_outgroup=True):
    tree = Tree.from_newick(text, binary=True)
    if taxa is not None and set(tree.tips) != set(taxa):
        raise ValueError('Every iterative guide must have exactly the original taxa')
    if anchor_outgroup and ('OUT' not in tree.tips or tree.parent[tree.tips['OUT']] != tree.root):
        raise ValueError('Iterative inference requires leaf OUT as a root child')
    return tree


def clades(tree):
    labels = {}
    result = set()
    for u in reversed(tree.preorder):
        labels[u] = (frozenset([tree.labels[u]]) if not tree.children[u] else
                     frozenset().union(*(labels[v] for v in tree.children[u])))
        if 1 < len(labels[u]) < len(tree.tips):
            result.add(labels[u])
    return result


def switching_guides(network, guide, *, anchor_outgroup=True):
    """Flip parents relative to the supplied guide, without a 2**r scan.

    In a binary level-1 network each cycle changes its own set of clusters.
    Minimizing cluster disagreement independently recovers a reference
    switching. Verify the reconstructed guide before proposing any moves.
    """
    graph = parse_network(network)
    validate_network(graph, level1=True)
    reference = anchored_tree(guide, anchor_outgroup=anchor_outgroup)
    root = next(u for u in graph if graph.in_degree(u) == 0)
    if anchor_outgroup and not any(graph.nodes[v].get('label') == 'OUT' for v in graph.successors(root)):
        raise ValueError('The network must keep OUT outside all cycles')
    hybrids = [u for u in graph if graph.in_degree(u) == 2]
    if any((graph.in_degree(u), graph.out_degree(u)) not in
           ((0,2),(1,2),(1,0),(2,1)) for u in graph):
        raise ValueError('Switching refinement requires a rooted binary network')
    parents = [list(graph.predecessors(u)) for u in hybrids]
    def switched(bits):
        current = graph.copy()
        for i, (u, pair) in enumerate(zip(hybrids, parents)):
            current.remove_edge(pair[1-((bits >> i) & 1)], u)
        return Tree.from_graph(current, suppress_unary=True)
    target = clades(reference)
    bits = 0
    distance = len(clades(switched(bits)) ^ target)
    for i in range(len(hybrids)):
        alternative = bits ^ (1 << i)
        delta = len(clades(switched(alternative)) ^ target)
        if delta < distance:
            bits, distance = alternative, delta
    if canonical_tree(switched(bits)) != canonical_tree(reference):
        raise ValueError('Could not recover the current guide as a network switching')
    masks = [1 << i for i in range(len(hybrids))]
    seen = {canonical_tree(reference)}
    for mask in masks:
        text = canonical_tree(switched(bits ^ mask))
        anchored_tree(text, reference.tips, anchor_outgroup=anchor_outgroup)
        if text not in seen:
            seen.add(text)
            yield text


def refine(guide, evidence, output, solve, *, max_guide_runs=5, max_rounds=10,
           search_cap=None, inner_solver='per-k-stopping', anchor_outgroup=True):
    """Best improvement per pass; all trials compete on total quartet score.

    The first solve counts toward the total budget. Sampling and learned
    evidence stay fixed across guide solves.
    """
    original = anchored_tree(guide, anchor_outgroup=anchor_outgroup)
    evidence_hash = digest(evidence)
    seen, trials, accepted = set(), [], []
    incumbent = None
    stop_reason = 'local_optimum'

    def checkpoint(status='running'):
        report = dict(schema='camus-nn-iteration/v1', status=status,
            weights_frozen=True,
            guide_rooting='outgroup' if anchor_outgroup else 'supplied',
            outgroup_anchored=anchor_outgroup,
            objective='total_signed_displayed_quartet_score',
            inner_solver=inner_solver, inner_optimization_exact=inner_solver == 'overall',
            search_cap=search_cap, max_guide_runs=max_guide_runs, max_rounds=max_rounds,
            hybrid_order='newick', gene_starts=0, plateau_budget=0,
            trials=trials, accepted=accepted, stop_reason=stop_reason,
            winner=incumbent['index'] if incumbent else None)
        save(output / 'iteration.json', report)
        return report

    def evaluate(text, origin):
        text = canonical_tree(anchored_tree(text, original.tips, anchor_outgroup=anchor_outgroup))
        if text in seen or len(trials) >= max_guide_runs:
            return None
        seen.add(text)
        directory = output / 'trials' / f'{len(trials):04d}'
        directory.mkdir(parents=True)
        (directory / 'guide.nwk').write_text(text + '\n')
        record = dict(index=len(trials), guide=text, origin=origin,
                      directory=str(directory), status='running')
        trials.append(record)
        checkpoint()
        try:
            result = solve(directory / 'guide.nwk', directory, iterative=True)
        except BaseException as error:
            record.update(status='failed', error=repr(error))
            checkpoint()
            raise
        if search_cap is not None and result['k'] > search_cap:
            raise ValueError('Native solver exceeded the reticulation cap')
        native = json.loads((directory / 'network.metrics.json').read_text())['objective']
        guide_score = float(native['guide_score'])
        gain = result['objective']
        total = guide_score + gain
        if (not all(math.isfinite(v) for v in (guide_score, gain, total)) or
                abs(native['added_gain'] - gain) > gain_tolerance(native['added_gain'], gain) or
                abs(native['total_score'] - total) > gain_tolerance(native['total_score'], total)):
            raise ValueError('Native candidate and total-score receipts disagree')
        record.update(result, guide_score=guide_score, added_gain=gain,
                      total_score=total, status='complete')
        save(directory / 'prediction.json', record)
        checkpoint()
        return record

    def better(a, b):
        return a['total_score'] > b['total_score'] + gain_tolerance(a['total_score'], b['total_score'])

    current = incumbent = evaluate(guide, dict(kind='initial'))
    accepted.append(incumbent['index'])
    for round_index in range(max_rounds):
        if len(trials) >= max_guide_runs:
            stop_reason = 'max_guide_runs'
            break
        improved = None
        for text in switching_guides(current['network'], current['guide'], anchor_outgroup=anchor_outgroup):
            if len(trials) >= max_guide_runs:
                break
            candidate = evaluate(text, dict(kind='switching', parent=current['index'], round=round_index))
            if candidate is not None and better(candidate, incumbent) and (improved is None or better(candidate, improved)):
                improved = candidate
        if improved is None:
            break
        current = incumbent = improved
        accepted.append(incumbent['index'])
    else:
        stop_reason = 'max_rounds'
    if len(trials) >= max_guide_runs:
        stop_reason = 'max_guide_runs'
    if digest(evidence) != evidence_hash:
        raise ValueError('Frozen evidence changed during iteration')
    report = checkpoint('complete')
    selected = Path(incumbent['directory'])
    for name in ('network.csv', 'network.metrics.json', 'network.branches.json'):
        shutil.copyfile(selected / name, output / name)
    return dict(network=incumbent['network'], k=incumbent['k'], objective=incumbent['total_score'],
                guide_score=incumbent['guide_score'], added_gain=incumbent['added_gain']), report
