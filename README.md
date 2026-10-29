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

## Infer a network

Run the included example from the repository root:

```bash
camus-nn --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out ../camus-results/lqw
```

This runs CAMUS-LQW with every quartet and the bundled model. Add `--iterate`
for guide refinement, or `--sampler uniform` for uniform quartet sampling.
Choose a new output directory for each run; temporary files are managed and
cleaned automatically.

Gene trees contain one Newick tree per line; the guide is a binary Newick
tree. Taxon labels must match, and branch lengths are ignored. The example
uses a known outgroup named `OUT` at the guide root. For data without a known
outgroup, add `--guide-rooting supplied`: CAMUS starts from the root encoded
in the guide tree, keeps the taxon names, and applies no outgroup constraint.
This does not establish a biological root. Gene-tree roots are not used for
quartet counting.

Find the selected network in `network.nwk`, the candidate curve in `network.csv`,
and run details in `prediction.json` and `run.json`. Iterative runs also save
`iteration.json` and their trial guides and networks.

## Evaluate or train

Compare an inferred network with a known network:

```bash
camus-nn-evaluate --truth true-network.nwk \
  --estimate ../camus-results/lqw/network.nwk
```

The evaluator reports softwired FN, FP, and error, plus hardwired cluster
distance (HWCD). See the [metric definitions](docs/reference.md#network-evaluation)
for units and rooting options.

To train a model from known networks and gene trees, install the training
extra and run the example:

```bash
uv pip install --torch-backend=cpu '.[train]'
camus-nn-train --training-cases examples/training/cases.json \
  --output-model ../camus-results/training/model.json
```

This demonstrates training on one example; it does not recreate the bundled
model. Use `--model FILE` to infer with your own model.

The [command reference](docs/reference.md) covers all options, input formats,
metric definitions, training, and reproducibility checks. Use `camus-nn --help`
for CLI help. Runners for BROOQS, SNaQ, and PhyloNet-MPL are documented in
[misc](misc/README.md).

## License and acknowledgments

See [LICENSE](LICENSE). OpenAI ChatGPT/Codex assisted with implementation,
refactoring, documentation, and test development.
