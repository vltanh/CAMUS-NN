# Evaluation

[README](../README.md) · [Inference](inference.md) · [Evaluation](evaluation.md) · [Training](training.md)

Compare a true and inferred network on exactly the same taxa. The evaluator
returns softwired (SW) cluster rates and hardwired cluster distance (HWCD).
Choose the implementation explicitly; the Python cluster-set backend is the default.

| `--metric` | Implementation | Root handling |
| --- | --- | --- |
| `both`, `softwired`, `hardwired` | Python distinct-cluster sets | Reference-taxon orientation by default; `--rooted` uses supplied roots |
| `phylonet-sw` | PhyloNet 3.8.2 `CmpNets -m cluster` | Supplied roots |
| `phylonetworks-hwcd` | PhyloNetworks 1.4.0 | Semidirected by default; `--rooted` uses supplied roots |

Java and Julia are optional and only used by their selected backend. External
backends require `--output`; their logs and receipts are retained beside the
output JSON. The scorer never installs packages or downloads a runtime.

## Python cluster-set backend

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

## Python cluster definitions

**Softwired clusters** are the union of clusters across all displayed trees.
At each reticulation, a displayed tree keeps one incoming parent edge.

**Hardwired clusters** are descendant leaf sets in the complete directed
network, keeping all incoming reticulation edges.

Both comparisons use distinct nontrivial cluster sets: singletons and the
full taxon set are omitted, and duplicate clusters count once. Branch lengths
and inheritance probabilities are ignored. Apply the chosen orientation
before comparing the sets.

## Python softwired outputs

Let `T` be the true SW cluster set and `P` the inferred SW cluster set.

| JSON field | Definition | Units |
| --- | --- | --- |
| `softwired_fn` | `\|T − P\| / \|T\|` | Fraction |
| `softwired_fp` | `\|P − T\| / \|P\|` | Fraction |
| `softwired_error` | `(FN + FP) / 2` | Fraction |
| `true_clusters`, `inferred_clusters` | Sizes of `T` and `P` | Counts |

An empty denominator gives rate zero. Multiply rates by 100 for percentages.
The output also includes `cluster_precision`, `cluster_recall`, and `cluster_f1`.

## Python HWCD and normalized hardwired error

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

## Python orientation modes for SW and HWCD

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

## Python cluster-set API and its defaults

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

## PhyloNet SW backend

Install Java (tested with OpenJDK 17) and obtain the **PhyloNet 3.8.2** JAR
from the [official project](https://github.com/NakhlehLab/PhyloNet).
Keep the JAR outside the checkout, for example at
`../camus-tools/PhyloNet-3.8.2.jar`. This backend requires these exact JAR bytes:

```text
eefabc198f113d3c835d67a44206203382e99a582d199478cc53f0bb746658fb
```

Run the actual upstream rooted cluster comparison:

```bash
camus-nn-evaluate --metric phylonet-sw \
  --truth true-network.nwk --estimate inferred-network.nwk \
  --sw-jar ../camus-tools/PhyloNet-3.8.2.jar \
  --output ../camus-results/phylonet-sw.json
```

The truth is passed first to `CmpNets truth estimate -m cluster`. Its FN, FP,
and mean error are returned as `softwired_fn`, `softwired_fp`, and
`softwired_error`, in fractions. The values come from PhyloNet itself, rather
than the Python cluster extractor. PhyloNet's cluster conventions can give
different values even when Python also uses supplied roots.

Both network roots and hybrid directions are preserved. `--rooted` is
redundant for this backend; `--outgroup` and `--max-switchings` are rejected.
The wrapper makes topology-only scoring copies with safe shared taxon tokens,
removing branch lengths, inheritance values, and internal annotations for
compatibility with PhyloNet's parser. The original inputs are retained.

Use `--sw-java COMMAND` to select Java, `--sw-heap SIZE` for its heap (default
`2g`), and `--sw-timeout SECONDS` for a comparison limit (default `3600`).
Java uses one core. Timeouts stop the whole process group. Input copies,
NEXUS commands, taxon maps, stdout/stderr, and success/failure receipts remain
under `<output-stem>-phylonet-sw/`; owned temporary/cache directories are
cleaned only after the process has stopped and the receipt is saved.

The result has `metric_version="phylonet-sw/v1-rooted"`. Its
`metric_parameters` record the JAR version/checksum, Java version, wrapper
hash, normalization, and receipt path.

## PhyloNetworks HWCD backend

Install Julia 1.11+ and **PhyloNetworks 1.4.0**. An isolated Julia environment
can be prepared outside the checkout:

```bash
julia --project=../camus-tools/phylonetworks -e \
  'using Pkg; Pkg.add(PackageSpec(name="PhyloNetworks", version="1.4.0")); Pkg.precompile()'
```

Keep the resulting `Project.toml` and `Manifest.toml` for repeatable setup.
An existing environment containing that version can also be selected.

```bash
camus-nn-evaluate --metric phylonetworks-hwcd \
  --truth true-network.nwk --estimate inferred-network.nwk \
  --hwcd-project ../camus-tools/phylonetworks \
  --output ../camus-results/phylonetworks-hwcd.json
```

This calls `PhyloNetworks.hardwiredclusterdistance(truth, estimate, false)`
and returns its raw `hardwired_distance`. The default treats networks as
semidirected and minimizes over compatible internal roots, preserving hybrid
edge directions. Add `--rooted` to pass `true` and compare supplied roots.
Neither mode uses an outgroup label; `--outgroup` is rejected.

This is PhyloNetworks' edge-typed, multiplicity-aware convention, rather than
Python's distinct-cluster-set count. See the
[upstream definition](https://juliaphylo.github.io/PhyloNetworks.jl/stable/lib/public/#PhyloNetworks.hardwiredclusterdistance).
The result is not halved or normalized, and Python hardwired FN/FP/error
fields are not attached to it. `--max-switchings` does not apply.

Use `--hwcd-julia COMMAND` to select Julia, `--hwcd-depot DIR` to select an
existing package depot, and `--hwcd-timeout SECONDS` for the comparison limit
(default `3600`). By default, Julia is found on `PATH` and uses its configured
environment and depot. One single-thread Julia worker serves repeated API
calls. Root-pair enumeration can be expensive; a timeout stops the worker,
and a later call starts a fresh one. Bad input pairs leave a healthy worker
available for later comparisons.

Logs, input pairs, runtime information, and success/failure receipts remain
under `<output-stem>-hwcd/`. Owned temporary files are removed after the worker
stops. Output versions are `phylonetworks-hwcd/v1-semidirected` or
`phylonetworks-hwcd/v1-rooted`. `metric_parameters` record package/Julia versions,
the scorer source and driver hashes, rooting mode, and receipt path. For an
explicit Julia project, its project/manifest hashes are recorded too.

## External Python APIs

Supply a work directory for retained logs and provenance:

```python
from camus_nn.phylonet_sw import PhyloNetSW
from camus_nn.phylonetworks_hwcd import PhyloNetworksHWCD

with PhyloNetSW(jar="../camus-tools/PhyloNet-3.8.2.jar",
               workdir="../camus-results/sw") as scorer:
    sw = scorer.score(true_newick, estimated_newick)

with PhyloNetworksHWCD(julia_project="../camus-tools/phylonetworks",
                      workdir="../camus-results/hwcd") as scorer:
    hw = scorer.score(true_newick, estimated_newick)  # rooted=False
    rooted_hw = scorer.score(true_newick, estimated_newick, rooted=True)
```

These dictionaries include numeric scores, true/inferred reticulation counts,
and `metric_parameters_json` containing implementation provenance. The
`evaluate_networks` function remains the combined Python cluster-set API.

## Comparing and recording conventions

Every CLI result identifies its implementation with `metric_version` and
`metric_parameters`. Keep results from different implementations or rooting
conventions in separately labeled columns. Root handling and cluster
multiplicity can affect scores; native and upstream values are not
interchangeable. Root placement never filters cases.

A half-scaled cluster distance is another convention and is not selected by
any of these options. No backend modifies the inference predictions.

## Options and exact extraction

| Option | Meaning and default |
| --- | --- |
| `--truth FILE`, `--estimate FILE` | Required extended-Newick networks on the same taxa. |
| `--metric` | Python: `both`, `softwired`, `hardwired`; upstream: `phylonet-sw`, `phylonetworks-hwcd`. Default `both`. |
| `--outgroup LABEL` | Python metrics only: reference-taxon orientation; default `OUT`. |
| `--rooted` | Supplied roots; redundant for PhyloNet, replaces semidirected comparison for PhyloNetworks. |
| `--max-switchings N` | Python SW only: positive exact configuration budget per network; default `65536`. |
| `--output FILE` | New JSON file; required for external backends. Results are also printed. |

Python SW extraction factors switchings by components separated by bridges. For
components with `h` reticulations, the evaluated configuration count is
`1 + sum(2**h − 1)`. Exceeding the budget raises an error; no sampling is
substituted. Hardwired-only evaluation skips displayed-tree extraction.
The evaluator supports binary reticulations, tree polytomies, and parallel
edges; repeated clusters are counted once.

See the [implementation](../camus_nn/evaluate.py) and
[evaluation tests](../tests/test_evaluate.py) for the exact behavior.
