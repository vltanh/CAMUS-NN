# Evaluation

[README](../README.md) · [Inference](inference.md) · [Evaluation](evaluation.md) · [Training](training.md)

Compare a true and inferred network on exactly the same taxa. The evaluator
returns softwired (SW) cluster rates and hardwired cluster distance (HWCD).
Both support the same two choices of cluster orientation, described below.

## Run

Run from the repository root. With a known outgroup named `OUT`:

```bash
camus-nn-evaluate --truth true-network.nwk \
  --estimate ../camus-results/lqw/network.nwk --outgroup OUT \
  --output ../camus-results/scores-outgroup.json
```

To use the roots represented in the two network Newick records instead:

```bash
camus-nn-evaluate --truth true-network.nwk \
  --estimate ../camus-results/lqw/network.nwk --rooted \
  --output ../camus-results/scores-rooted.json
```

The command prints JSON and optionally saves it to a new file. It never
modifies the input networks or overwrites an existing output file. Use
`--metric softwired` or `--metric hardwired` to compute only one family;
the default `both` computes both. From a source checkout,
`python -m camus_nn.evaluate` is the equivalent entry point.

## Cluster definitions

**Softwired clusters** are the union of clusters across all displayed trees.
At each reticulation, a displayed tree keeps one incoming parent edge.

**Hardwired clusters** are descendant leaf sets in the complete directed
network, keeping all incoming reticulation edges.

Both comparisons use distinct nontrivial cluster sets: singletons and the
full taxon set are omitted, and duplicate clusters count once. Branch lengths
and inheritance probabilities are ignored. Apply the chosen orientation
before comparing the sets.

## Softwired outputs

Let `T` be the true SW cluster set and `P` the inferred SW cluster set.

| JSON field | Definition | Units |
| --- | --- | --- |
| `softwired_fn` | `\|T − P\| / \|T\|` | Fraction |
| `softwired_fp` | `\|P − T\| / \|P\|` | Fraction |
| `softwired_error` | `(FN + FP) / 2` | Fraction |
| `true_clusters`, `inferred_clusters` | Sizes of `T` and `P` | Counts |

An empty denominator gives rate zero. Multiply rates by 100 for percentages.
The output also includes `cluster_precision`, `cluster_recall`, and `cluster_f1`.

## HWCD and normalized hardwired error

Let `H_T` and `H_P` be the true and inferred hardwired cluster sets.

```text
HWCD = |H_T − H_P| + |H_P − H_T|
```

| JSON field | Meaning | Units |
| --- | --- | --- |
| `hardwired_distance` | HWCD, the full symmetric-difference size | Raw count |
| `hardwired_fn_count` | Missing true hardwired clusters | Raw count |
| `hardwired_fp_count` | Extra inferred hardwired clusters | Raw count |
| `hardwired_fn` | Missing count divided by `\|H_T\|` | Fraction |
| `hardwired_fp` | Extra count divided by `\|H_P\|` | Fraction |
| `hardwired_error` | Mean of the two hardwired rates | Fraction |

**`hardwired_distance` and `hardwired_error` are different metrics.** The first
is HWCD; the second is a normalized error rate. HWCD is not divided by two
or converted to a percentage. Mean HWCD over a dataset is the mean of the
per-network raw counts. Empty denominators for the rates give zero.

Additional fields report true/inferred hardwired cluster counts, matching
clusters, precision, recall, and F1.

## Two orientation modes for both SW and HWCD

| Mode | CLI | Python API | What happens |
| --- | --- | --- | --- |
| Reference-taxon orientation | `--outgroup OUT` (default) | `outgroup="OUT"` | Replace each cluster containing the reference taxon by its complement. |
| Supplied roots | `--rooted` | `outgroup=None` | Keep descendant clusters as represented by each input root. |

The first mode can use another shared label with `--outgroup LABEL`. A chosen
reference label does not establish a biological outgroup. The reference must
exist in both networks. The two CLI flags are mutually exclusive.

For SW, reference-taxon orientation is applied to each displayed-tree split.
For HWCD, it is applied to each full-network descendant bipartition. It does
not reroot hybrid edges, search over alternative network roots, or guarantee
that differently directed versions of a network have equal scores. Supplied-root
comparison uses the directed networks exactly as represented.

For example, these trees have the same unrooted topology and different roots:

```text
True:      ((A,B),(C,OUT));
Estimated: (A,(B,(C,OUT)));
```

| Orientation | SW error (%) | HWCD |
| --- | ---: | ---: |
| Reference taxon `OUT` | 0 | 0 |
| Supplied roots | 50 | 2 |

Use the same orientation for every method in a comparison. Root placement
does not filter out cases. These are evaluation settings; inference's
`--guide-rooting` flag separately controls treatment of its starting guide.

## Python API and its defaults

The combined entry point accepts extended-Newick strings:

```python
from camus_nn.evaluate import evaluate_networks

scores = evaluate_networks(true_newick, estimated_newick, outgroup="OUT")
print(scores["softwired_error"])      # Fraction.
print(scores["hardwired_distance"])   # Raw HWCD count.

rooted_scores = evaluate_networks(true_newick, estimated_newick, outgroup=None)
```

The separate SW and HW functions call the same implementations used by the
combined entry point, but their default orientation differs:

| Function | Default `outgroup` | Returns |
| --- | --- | --- |
| `evaluate_networks` | `"OUT"` | Both metric families, unless `metric` selects one |
| `softwired_cluster_distance` | `None` | SW FN/FP/error and cluster statistics |
| `hardwired_cluster_distance` | `None` | Raw HWCD, hardwired rates, and cluster statistics |
| `cluster_summary`, `hardwired_clusters` | `None` | Extracted clusters rather than comparison scores |

Pass the same `outgroup` explicitly to get matching results:

```python
from camus_nn.evaluate import softwired_cluster_distance, hardwired_cluster_distance

sw = softwired_cluster_distance(true_newick, estimated_newick, outgroup="OUT")
hw = hardwired_cluster_distance(true_newick, estimated_newick, outgroup="OUT")
```

Omitting that argument from these helpers selects supplied-root comparison,
which can explain differences from the default CLI output.

## Other HWCD conventions

This package implements the distinct-cluster-set definition above. It does
not call Julia's PhyloNetworks `hardwiredclusterdistance`. That library has
its own treatment of edge types, multiplicities, and semidirected roots;
its results must be identified separately. A half-scaled cluster distance is
also a different convention. Neither is selected by `--rooted`.

## Options and exact extraction

| Option | Meaning and default |
| --- | --- |
| `--truth FILE`, `--estimate FILE` | Required extended-Newick networks on the same taxa. |
| `--metric {both,softwired,hardwired}` | Metric family; default `both`. |
| `--outgroup LABEL` | Reference-taxon orientation; default `OUT`. |
| `--rooted` | Use supplied roots instead of reference-taxon orientation. |
| `--max-switchings N` | Positive exact SW configuration budget per network; default `65536`. |
| `--output FILE` | Optional new JSON file; results are also printed. |

SW extraction factors switchings by components separated by bridges. For
components with `h` reticulations, the evaluated configuration count is
`1 + sum(2**h − 1)`. Exceeding the budget raises an error; no sampling is
substituted. Hardwired-only evaluation skips displayed-tree extraction.
The evaluator supports binary reticulations, tree polytomies, and parallel
edges; repeated clusters are counted once.

See the [implementation](../camus_nn/evaluate.py) and
[evaluation tests](../tests/test_evaluate.py) for the exact behavior.
