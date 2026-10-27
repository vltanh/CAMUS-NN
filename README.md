# CAMUS-NN

Training and C++ inference for **CAMUS-LQF**, **CAMUS-LQW**, and iterative
CAMUS-LQW. Both methods use the same included, trained frequency-only neural
network: three inputs, eight ReLU hidden units, and three membership outputs.

## Install

Requires Linux and Python 3.11+. Installing from source requires a C++17 compiler
with OpenMP support (GCC 11+). Installation builds and bundles the native programs
and trained NN automatically.

```bash
git clone https://github.com/vltanh/CAMUS-NN.git
cd CAMUS-NN
python -m pip install .
```

For training support, install `python -m pip install '.[train]'`.

## Inference

The examples contain one validation and one held-out test case, each with
16 taxa (including `OUT`), 1,000 IQ-TREE gene trees and its ASTRAL guide.
Run these commands from the repository root:

```bash
# LQF on validation and LQW on held-out test.
camus-nn --method lqf --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out out-val-lqf
camus-nn --method lqw --genes examples/test/genes.nwk \
  --guide examples/test/guide.nwk --out out-test-lqw

# Iterative LQW on validation, at most five guide solves.
camus-nn --method lqw --iterate --max-guide-runs 5 \
  --genes examples/validation/genes.nwk --guide examples/validation/guide.nwk --out out-val-iter
```

The trained NN in `models/all-frequency.json` is used by default for both
methods. Set `--model your-model.json` to use another fit. Outputs include
`network.nwk` (selected network), `network.csv` (returned candidate curve),
`prediction.json`, and logs. Iteration also saves `iteration.json` and every
trial's guide, curve and score in `trials/`.

Each gene tree must occupy one line. Use the same taxa in the gene trees and
guide. The binary guide must have leaf `OUT` directly under its root. Branch
lengths are ignored. Each selected quartet needs a resolved gene observation.

All quartets are used by default. `--sampler rooted` uses every quartet
containing `OUT` once; `--sampler uniform`, `eu` or `ew` draws 50 quartets per
unrooted guide edge, including leaf edges. `--multiplicity` changes that budget;
`--seed` and `--case` determine the draws. Repeated draws contribute repeatedly.

Use a new output directory for each run. `--threads` defaults to one.
`--timeout` limits the whole run, including iterations, to at most 86,400 seconds.
Temporary count/evidence tables are removed after the processes stop.

## Training

The bundled NN is ready for inference. To fit the same architecture yourself:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  camus-nn-train --data examples/train.csv --model new-model.json --seed 42
camus-nn --method lqw --model new-model.json \
  --genes examples/validation/genes.nwk --guide examples/validation/guide.nwk --out out-new-model
```

`examples/train.csv` demonstrates the training format with synthetic rows.
Train on labeled quartets from known **training** networks. Required CSV columns:
`case,split,c0,c1,c2,y0,y1,y2`. Extra columns are ignored. `split` must be `train`.
For taxa `t0,t1,t2,t3`, channels are `t0 t1 | t2 t3`, `t0 t2 | t1 t3`, and
`t0 t3 | t1 t2`. Counts are gene-tree observations; labels indicate whether the
training network displays each topology. Multiple labels may be 1.

Training gives each network equal weight, sorts resolved frequencies, handles
rank ties symmetrically, and fits float64 multilabel cross-entropy with weight
penalty `1e-4` using L-BFGS-B. Scaling uses training rows only. Defaults are
seed 42 and eight hidden units. CPU math libraries can affect refits.

## Methods and iterations

LQF retains observed topology counts when `p > 0.5`. LQW uses the signed
membership score `2*p - 1` for all three channels, multiplied by the quartet's
sampling multiplicity. The NN takes normalized quartet frequencies, so any
positive number of input loci is supported. Membership probabilities are
independent outputs.

Stored LQW weights also include the score scale `n_loci/1000`.
For the complete-taxon inputs supported here, `n_loci` is the total number of
input gene trees and this is a common positive factor across all quartets.
It changes score units, not the mathematical optimum, and does not require
1,000 loci. With identical frequencies, changing the locus total leaves NN
predictions unchanged; numerical comparison tolerances still apply.

Iterative LQW freezes the initial sampling and learned evidence. It proposes
guides by switching one cycle at a time, in deterministic Newick order, and
accepts improvements in **guide score + added gain**. A non-improving pass or
the guide-solve budget stops refinement. Five solves include the initial solve.
The default inner DP stops at the first non-improving reticulation count and
is inexact. `--inner-solver overall` chooses exact unrestricted inner DP;
`--search-cap K` bounds every LQW solve. The outer iteration remains a heuristic.
All returned candidates are saved.

## Tests

```bash
python -m unittest discover -s tests -v
```

## AI assistance

OpenAI ChatGPT/Codex assisted with implementation, refactoring, documentation,
and test development for this repository.

See [LICENSE](LICENSE) for licensing.
