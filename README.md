# CAMUS-NN

CAMUS-NN reconstructs level-1 phylogenetic networks from discordant gene trees.
Starting from a rooted guide tree, it adds reticulation edges to optimize a
weighted quartet score.

The inputs are **gene trees and a rooted binary guide tree on the same taxa**.
The output is an **inferred level-1 network in extended Newick**, together with
candidate networks, scores, and logs.

- **Original CAMUS** uses frequency thresholds to filter quartets
  (`--method orig`).
- **CAMUS-LQF** (Learned Quartet Filtering) uses a trained neural network to
  filter quartets (`--method lqf`).
- **CAMUS-LQW** (Learned Quartet Weighting) uses the network's predictions as
  quartet weights (`--method lqw`, the default).
- **CAMUS-LQW + iteration** refines the guide through single-cycle SPR moves
  and reruns LQW with the same quartet evidence, for up to five solves by
  default (`--iterate`).

LQF and LQW share the included trained model.

## Install

Requires Linux, Python 3.11+, and a C++17 compiler with OpenMP support
(GCC 11+). With [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/vltanh/CAMUS-NN.git
cd CAMUS-NN
uv venv --python 3.11
source .venv/bin/activate
uv pip install --torch-backend=cpu .
```

Installation builds the native programs and includes the trained model.
Pip is also supported: `python -m pip install .`.

## Inference

Inputs are gene trees (one Newick tree per line) and a binary guide tree on
exactly the same taxa. Branch lengths are ignored. The bundled validation and
test examples each contain 16 taxa, 1,000 IQ-TREE gene trees, and an ASTRAL guide.

Run these examples from the repository root. Each command uses a fresh output
directory outside the checkout; temporary files are managed and cleaned
automatically.

```bash
# Original CAMUS.
camus-nn --method orig --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out ../camus-results/camus

# Learned quartet filtering.
camus-nn --method lqf --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out ../camus-results/lqf

# Learned quartet weighting.
camus-nn --method lqw --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out ../camus-results/lqw

# LQW with guide refinement.
camus-nn --method lqw --iterate --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out ../camus-results/lqw-iter
```

LQF and LQW use the bundled model automatically. To use your own model, add
`--model model.json`. For the held-out example, replace `examples/validation`
with `examples/test` and choose a new output directory.

### Guide rooting

The example guide has a known outgroup named `OUT` directly below the root.
This is the default `--guide-rooting outgroup` mode. Iteration keeps `OUT`
outside reticulation cycles. Gene-tree roots are not used for quartet counting.

For data without a known outgroup, use the root represented in the guide:

```bash
camus-nn --method lqw --genes gene-trees.nwk --guide astral.nwk \
  --guide-rooting supplied --out ../camus-results/supplied-guide
```

This keeps the guide's taxon names and applies no outgroup constraint. The
represented root is the outermost node in Newick: in `((A,B),(C,D));`, it
joins the A/B and C/D subtrees. A label such as `Root` does not determine root
placement. ASTRAL's represented root may be arbitrary; using it does not
establish a biological root. The same rooting flag works with `--iterate`.

### Common options

| Option | Purpose and default |
| --- | --- |
| `--genes FILE`, `--guide FILE` | Required gene trees and binary guide. |
| `--out DIR` | Required fresh output directory. |
| `--method {orig,lqf,lqw}` | Inference method; default `lqw`. |
| `--model FILE` | Custom model for LQF/LQW; default is the bundled model. |
| `--guide-rooting {outgroup,supplied}` | Use a known `OUT` or the guide's represented root; default `outgroup`. |
| `--sampler {all,rooted,uniform,eu,ew}` | Quartet sampler; default `all`. |
| `--multiplicity N` | Draw budget multiplier for Uniform/EU/EW; default `50` per unrooted guide edge. |
| `--seed N` | Sampling seed; default `42`. |
| `--iterate` | Enable LQW guide refinement with fixed quartet evidence. |
| `--max-guide-runs N` | Maximum iterative solves, including the initial solve; default `5`. |
| `--search-cap K` | Optional reticulation limit per LQW solve; default uncapped. |
| `--threads N` | Counting and search threads; default `1`. |
| `--timeout SECONDS` | Total run limit; default and maximum `86400` seconds. |

All uses every quartet. Rooted uses every quartet containing `OUT` once.
Uniform samples globally; EU and EW sample around guide edges. Repeated draws
retain their multiplicity. See the [full inference reference](docs/reference.md#inference)
for sampling, memory, and search details, or run `camus-nn --help`.

### Outputs

| File | Contents |
| --- | --- |
| `network.nwk` | Selected network in extended Newick. |
| `network.csv` | Candidate networks and scores across reticulation counts. |
| `prediction.json` | Selected score, reticulation count, method, and phase timings. |
| `run.json` | Arguments and input/model/executable hashes. |
| `iteration.json`, `trials/` | Iterative search history and trial guides, networks, and scores. |

Logs are retained alongside these files. Large temporary count/evidence tables
are removed after the run's processes stop, including on failure.

## Network evaluation

Compare a true and inferred network on the same taxa:

```bash
camus-nn-evaluate --truth true-network.nwk \
  --estimate ../camus-results/lqw/network.nwk \
  --output ../camus-results/scores.json
```

The command prints JSON and optionally saves it to a new file. It reports
softwired FN, FP, and error, plus hardwired cluster distance (HWCD):

- **FN:** fraction of true clusters missing from the estimate.
- **FP:** fraction of estimated clusters absent from the truth.
- **Softwired error:** the mean of softwired FN and FP.
- **HWCD:** the size of the symmetric difference of the hardwired cluster sets.

FN, FP, and error are fractions; multiply by 100 for percentages. HWCD is a
raw count. Softwired clusters come from displayed trees; hardwired clusters
are descendant leaf sets in the complete network. Both omit trivial clusters
and count duplicate clusters once.

By default, evaluation orients clusters away from `OUT`. Use `--outgroup LABEL`
for another reference taxon, or `--rooted` to compare the supplied roots.
See [evaluation details and the Python API](docs/reference.md#network-evaluation)
for exact definitions and extraction limits.

## Training and preprocessing

Install the training extra:

```bash
uv pip install --torch-backend=cpu '.[train]'
```

Train directly from true networks and their gene trees:

```bash
camus-nn-train --training-cases examples/training/cases.json \
  --output-model ../camus-results/training/model.json --seed 42
```

Alternatively, prepare a reusable labeled quartet table, then fit from it:

```bash
camus-nn-prepare --training-cases examples/training/cases.json \
  --output-table ../camus-results/training/quartets.csv

camus-nn-train --quartet-table ../camus-results/training/quartets.csv \
  --output-model ../camus-results/training/model-from-table.json --seed 42
```

The case list is a JSON array. Paths are relative to the JSON file:

```json
[
  {"case": "training-example", "network": "network.nwk", "genes": "genes.nwk"}
]
```

Each case needs a true network and gene trees with matching taxon labels.
Keep training cases separate from validation and test cases. The included
example has one 26-taxon training network; it demonstrates the commands and
does not recreate the bundled model's full training cohort.

Use `--model ../camus-results/training/model.json` for inference with the new
model. Training uses seed `42` and eight hidden units by default. Each network
receives equal total weight. See [training options](docs/reference.md#training)
and [input formats](docs/reference.md#input-formats) for the CSV schema,
labeling rules, optimizer, and reproducibility settings.

## Reproducibility and tests

After installing the package, check that fresh runs produce identical candidate
curves and selected networks:

```bash
python scripts/check_reproducibility.py --out ../camus-results/reproducibility
```

This covers Original CAMUS, LQF, LQW, iteration, and supplied-root inference.
With the training extra installed, add `--training` to check repeated fitting
and agreement between CSV and direct training. Refits can depend on CPU math
libraries; see [reproducibility details](docs/reference.md#reproducibility).

Run the tests with the training extra installed:

```bash
python -m unittest discover -s tests -v
```

## Other methods and reference

The [misc guide](misc/README.md) covers runners for original CAMUS, BROOQS,
SNaQ, and PhyloNet-MPL. The [command reference](docs/reference.md) contains the
complete CLI options, metric definitions, and training formats.

## License and acknowledgments

See [LICENSE](LICENSE). OpenAI ChatGPT/Codex assisted with implementation,
refactoring, documentation, and test development.
