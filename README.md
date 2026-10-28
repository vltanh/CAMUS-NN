# CAMUS-NN

CAMUS-NN provides C++ inference for **original CAMUS**, **CAMUS-LQF**,
**CAMUS-LQW**, and iterative CAMUS-LQW, plus training for the learned methods.
LQF and LQW use the included frequency-only neural network: three inputs,
eight ReLU hidden units, and three membership outputs.

## Install

Requires Linux, Python 3.11+, and a C++17 compiler with OpenMP support
(GCC 11+). Installation builds the native programs and bundles the trained NN.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/vltanh/CAMUS-NN.git
cd CAMUS-NN
uv venv --python 3.11
source .venv/bin/activate
uv pip install --torch-backend=cpu '.[train]'
```

The `train` extra includes fitting support and pins the library versions used
for the included NN. Use `uv pip install --torch-backend=cpu .` for inference
and preprocessing. Pip also supports `python -m pip install '.[train]'`.

## Inference

Run the included validation and held-out test examples from the repository
root. Each has 16 taxa including `OUT`, 1,000 IQ-TREE gene trees, and an ASTRAL guide.

```bash
camus-nn --method orig --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out out-val-camus

camus-nn --method lqf --model models/all-frequency.json \
  --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out out-val-lqf

camus-nn --method lqw --model models/all-frequency.json \
  --genes examples/test/genes.nwk \
  --guide examples/test/guide.nwk --out out-test-lqw

camus-nn --method lqw --model models/all-frequency.json \
  --iterate --max-guide-runs 5 \
  --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out out-val-iter
```

The output directory contains the selected network in `network.nwk`, the
returned candidate curve in `network.csv`, `prediction.json`, and logs.
Iteration adds `iteration.json` and each trial's guide, curve, and score
under `trials/`. Temporary count/evidence tables are removed after processes stop.

LQF and LQW need a trained model. The examples explicitly select the included
`models/all-frequency.json`, which is also the default when `--model` is omitted.

Original CAMUS uses its count-based quartet filter (mode 2, threshold 0.5)
and the C++ search. It does not load or apply the neural network.

**Inputs and method**

| Argument | Meaning and default |
| --- | --- |
| `--genes FILE` | Required. Gene trees, one Newick tree per line. |
| `--guide FILE` | Required. Binary guide Newick with leaf `OUT` directly under the root. |
| `--out DIR` | Required. A fresh output directory that must not already exist. |
| `--method {orig,lqf,lqw}` | Original CAMUS or a learned method. Default: `lqw`. |
| `--model FILE` | Trained NN JSON for LQF/LQW. Default: the bundled `models/all-frequency.json`. Unused by original CAMUS. |

<details>
<summary>Click to show quartet sampling options</summary>

| Argument | Meaning and default |
| --- | --- |
| `--sampler {all,rooted,uniform,eu,ew}` | Default: `all` (every quartet). `rooted` uses every quartet containing `OUT` once; `uniform` samples quartet sets globally; `eu` (edge-uniform) and `ew` (edge-walk) sample around guide edges. |
| `--multiplicity N` | Positive sampling budget multiplier for `uniform`, `eu`, and `ew`: total draws equal `N` times the number of unrooted guide edges, including leaf edges. Default: `50`. Repeated draws contribute repeatedly. |
| `--seed INTEGER` | Deterministic sampling seed. Default: `42`. |
| `--case ID` | Case identity combined with the seed to determine sampling. Default: `single-case`. |

</details>

<details>
<summary>Click to show LQW iteration and search options</summary>

| Argument | Meaning and default |
| --- | --- |
| `--iterate` | Enable single-cycle guide refinement for LQW. Sampling and learned evidence remain fixed across solves. |
| `--max-guide-runs N` | Positive solve budget for iteration, including the initial solve. Default: `5`. |
| `--max-rounds N` | Nonnegative outer switching-pass limit for iteration. Default: `10`; `0` keeps the initial solve. |
| `--inner-solver {per-k-stopping,overall}` | Default: `per-k-stopping` (heuristic early stopping). `overall` selects exact inner optimization and requires `--iterate`. |
| `--search-cap K` | Nonnegative maximum reticulations in every LQW solve. Default: uncapped. |

</details>

<details>
<summary>Click to show runtime and executable options</summary>

| Argument | Meaning and default |
| --- | --- |
| `--threads N` | Positive C++ counting/search thread count. Default: `1`. |
| `--batch-rows N` | Positive rows per NN prediction/evidence batch; smaller batches reduce preprocessing memory. Default: `65536`. |
| `--timeout SECONDS` | Total time limit for counting, preprocessing, and all solves. Must be positive and at most `86400`; default: `86400`. |
| `--bin-dir DIR` | Directory containing `camus-count` and `camus-native`. Default: bundled executables. |
| `-h, --help` | Show usage and argument descriptions. |

</details>

Gene trees and guide must have matching taxa. Branch lengths are ignored.
Any positive number of loci is supported: the NN uses normalized quartet
frequencies. Each selected quartet needs a resolved gene-tree observation.

## Network evaluation

Compare a true network with any inferred network saved in extended Newick:

```bash
camus-nn-evaluate --truth true-network.nwk \
  --estimate out-test-lqw/network.nwk --output ../results/scores.json

# Equivalent command from the source checkout.
python -m camus_nn.evaluate --truth true-network.nwk \
  --estimate out-test-lqw/network.nwk
```

The command prints JSON and optionally saves it to a new file. Inputs are
never modified. The Python API accepts Newick strings:

```python
from camus_nn.evaluate import evaluate_networks

scores = evaluate_networks(true_newick, estimated_newick)
print(scores["softwired_fn"], scores["softwired_fp"], scores["softwired_error"])
print(scores["hardwired_distance"])  # Raw HWCD count.
```

Softwired clusters are the union of clusters across all displayed trees.
Hardwired clusters are descendant leaf sets in the complete directed network.
Both use **distinct nontrivial cluster sets**, omitting singletons and the
full taxon set. Duplicate clusters, including duplicates from parallel hybrid
edges, count once. For true and estimated cluster sets `T` and `P`:

```text
FN = |T - P| / |T|       FP = |P - T| / |P|
error = (FN + FP) / 2
HWCD = |T symmetric-difference P|   (hardwired clusters)
```

Rates are fractions in `[0, 1]`; multiply by 100 for percentages. An empty
denominator gives rate zero. `hardwired_distance` is a raw count, without
percentage normalization or division by two. Results also include normalized
hardwired FN/FP/error, cluster counts, precision, recall and F1. Mean HWCD
across datasets is the average of the per-network raw counts. This is the
cluster-set convention, distinct from PhyloNetworks' edge-typed multiset variant.

| Argument | Meaning and default |
| --- | --- |
| `--truth FILE` | Required. One true network in extended Newick. |
| `--estimate FILE` | Required. One estimated network on exactly the same taxa. |
| `--metric {both,softwired,hardwired}` | Default: `both`. Hardwired-only evaluation avoids displayed-tree extraction. |
| `--outgroup LABEL` | Default: `OUT`. Orient clusters away from this taxon; it must exist in both inputs. |
| `--rooted` | Use the roots supplied in Newick, instead of outgroup orientation. Mutually exclusive with `--outgroup`. |
| `--max-switchings N` | Positive exact SW switching budget per network; default `65536`. |
| `--output FILE` | Optional new JSON output file. Also prints results to stdout. |

SW extraction factors switchings by components separated by bridges; the budget
counts `1 + sum(2**h - 1)` configurations over components with `h` reticulations.
It raises an error when the budget is exceeded and never substitutes sampling.
The evaluator supports binary reticulations, tree polytomies and parallel edges.
Branch lengths and inheritance probabilities do not affect these metrics.
Outgroup orientation does not reroot hybrid edges or minimize over alternative
network roots. Every valid input is scored; root placement does not filter cases.

The extraction functions `cluster_summary`, `hardwired_clusters`,
`softwired_cluster_distance` and `hardwired_cluster_distance` are available
from `camus_nn.evaluate`. These lower-level functions use supplied roots by
default; pass `outgroup="OUT"` to match `evaluate_networks` and the CLI.

## Preprocessing

Convert a list of true networks and gene-tree files into a reusable labeled
training table:

```bash
camus-nn-prepare --training-cases examples/training/cases.json \
  --output-table out-training/quartets.csv
```

Preprocessing counts every quartet set with C++ and labels the topologies
displayed by the supplied true network. The example contains one separate
training case with 26 taxa and 1,000 IQ-TREE gene trees.
See [input formats](#input-formats) for the case-list and table structures.

**Inputs and output**

| Argument | Meaning and default |
| --- | --- |
| `--training-cases FILE` | Required. JSON case list with `case`, `network`, and `genes` fields. |
| `--output-table FILE` | Required. New labeled quartet CSV; the destination must not exist. |

<details>
<summary>Click to show counting and labeling options</summary>

| Argument | Meaning and default |
| --- | --- |
| `--threads N` | Positive C++ counting thread count. Default: `1`. |
| `--max-switchings N` | Positive limit on exact parent-choice switchings per true network. Default: `4096`. Increase it to label networks requiring more switchings. |
| `--bin-dir DIR` | Directory containing `camus-count`. Default: bundled executable. |
| `-h, --help` | Show usage and argument descriptions. |

</details>

## Training

Fit from the table prepared above, or fit directly from the case list.
Both input modes use the same CSV trainer; direct case input prepares a
temporary table internally.

```bash
# Fit from labeled quartet rows.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  camus-nn-train --quartet-table out-training/quartets.csv \
  --output-model out-training/model.json --seed 42

# Fit directly from networks and gene trees.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  camus-nn-train --training-cases examples/training/cases.json \
  --output-model out-training/direct-model.json --seed 42

# Use a newly fitted model for inference.
camus-nn --method lqw --model out-training/model.json \
  --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out out-new-model
```

Choose exactly one of the two input arguments.

**Training inputs and output**

| Argument | Meaning and default |
| --- | --- |
| `--quartet-table FILE [FILE ...]` | One or more labeled training CSVs. Input files must be distinct. |
| `--training-cases FILE` | JSON list of true networks and gene trees; count and label their quartets before fitting. |
| `--output-model FILE` | Required. New trained NN JSON; the destination must not exist. |

<details>
<summary>Click to show model fitting options</summary>

| Argument | Meaning and default |
| --- | --- |
| `--seed INTEGER` | Nonnegative seed for NN initialization. Default: `42`. |
| `--hidden-width N` | Positive number of ReLU hidden units. Default: `8`. |

</details>

<details>
<summary>Click to show preprocessing and runtime options</summary>

| Argument | Meaning and default |
| --- | --- |
| `--threads N` | Positive counting thread count with `--training-cases`. Default: `1`. |
| `--max-switchings N` | Positive exact-labeling budget per network with `--training-cases`. Default: `4096`. |
| `--bin-dir DIR` | Directory containing `camus-count` for `--training-cases`. Default: bundled executable. |
| `-h, --help` | Show usage and argument descriptions. |

</details>

Fitting uses float64 multilabel cross-entropy, L-BFGS-B, and weight penalty
`1e-4`. Each network receives equal total weight, frequency-rank ties are
treated symmetrically, and scaling is computed from training rows.

CPU math libraries can affect refits. For matching refits on AVX2-capable
CPUs, also set `ATEN_CPU_CAPABILITY=avx2`, `OPENBLAS_CORETYPE=Zen`, and
`MKL_ENABLE_INSTRUCTIONS=AVX2` with the single-thread settings shown above.

## Input formats

### Training case list (JSON)

The file is a nonempty JSON array with one object per training network.
The included [examples/training/cases.json](examples/training/cases.json) is:

```json
[
  {"case": "training-example", "network": "network.nwk", "genes": "genes.nwk"}
]
```

| Field | Required value |
| --- | --- |
| `case` | Nonempty string identifying the training network. |
| `network` | String path to the true network in extended Newick. |
| `genes` | String path to its gene-tree file, one Newick tree per line. |
| `split` | Optional; when supplied, it must be `"train"`. |

- Paths may be absolute or relative to the JSON file's directory.
- Case IDs, network files, and gene-tree files must each be distinct across records.
- Each case needs at least four taxa, with matching labels in the true network and every gene tree.
- Each quartet needs a resolved gene-tree observation and a resolved topology displayed by the true network. Multiple displayed topologies receive label 1.
- Keep training cases separate from validation and test cases.

### Labeled quartet table (CSV)

Include a header and one data row per training quartet. These are the first
two quartets from the included training case, showing the required columns:

```csv
case,split,c0,c1,c2,y0,y1,y2
training-example,train,580,196,224,1,0,0
training-example,train,398,287,315,1,0,0
```

| Columns | Required values |
| --- | --- |
| `case` | Nonempty network ID; every row from the same training network shares this ID. |
| `split` | The literal value `train` on every row. |
| `c0,c1,c2` | Finite, nonnegative integer counts of resolved gene-tree topologies. Their sum must be positive on each row. |
| `y0,y1,y2` | Binary displayed-topology membership labels, each `0` or `1`. The complete training data must contain both 0 and 1 labels. |

Counts and labels use the same channel order:

- Channel 0: `t0 t1 | t2 t3`.
- Channel 1: `t0 t2 | t1 t3`.
- Channel 2: `t0 t3 | t1 t2`.

The preprocessor also writes `t0,t1,t2,t3` (taxon labels in sorted order)
and `n_loci` (number of input gene trees). Extra columns are ignored by the
CSV trainer.

## Other inference methods

The [misc scripts](misc/README.md) run original CAMUS, BROOQS with TOB-QMC,
SNaQ, and PhyloNet-MPL. Each script includes installation commands for the versions used;
the linked guide gives runnable examples and explains every argument.

## Tests

With the `train` extra installed, run:

```bash
python -m unittest discover -s tests -v
```

## AI assistance

OpenAI ChatGPT/Codex assisted with implementation, refactoring, documentation,
and test development for this repository. See [LICENSE](LICENSE) for licensing.
