# Inference

[README](../README.md) · [Inference](inference.md) · [Evaluation](evaluation.md) · [Training](training.md)

Run the included validation and held-out test examples from the repository
root. Each has 16 taxa including `OUT`, 1,000 IQ-TREE gene trees, and an ASTRAL guide.
Temporary files are managed automatically inside each output directory.

```bash
camus-nn --method orig --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out ../camus-results/val-camus

camus-nn --method lqf \
  --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out ../camus-results/val-lqf

camus-nn --method lqw \
  --genes examples/test/genes.nwk \
  --guide examples/test/guide.nwk --out ../camus-results/test-lqw

camus-nn --method lqw \
  --iterate \
  --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out ../camus-results/val-iter
```

For data without a known outgroup, use `--guide-rooting supplied`. This starts
from the root encoded in the guide Newick (for example, ASTRAL output), keeps
the taxon names, and applies no outgroup constraint. It does not infer a
biological root. Use the same flag for fixed-guide and iterative LQW:

```bash
camus-nn --method lqw \
  --genes gene-trees.nwk --guide astral.nwk --guide-rooting supplied \
  --out ../camus-results/lqw

camus-nn --method lqw \
  --genes gene-trees.nwk --guide astral.nwk --guide-rooting supplied \
  --iterate --out ../camus-results/lqw-iter
```

Newick represents the root with its outermost node, not a special label.
For example, `((A,B),(C,D));` has a represented root joining the A/B and C/D
subtrees. An optional name such as `Root` does not determine the placement.
An unrooted tree can be written with an arbitrary represented root, as in
ASTRAL output. Gene-tree roots are not used for quartet counting.

The default `--guide-rooting outgroup` uses a guide with the known outgroup
named `OUT` directly below the root. Iteration keeps that outgroup outside
reticulation cycles. Supplied-root iteration allows cycles involving the root.

The output directory contains the selected network in `network.nwk`, the
returned candidate curve in `network.csv`, `prediction.json`, and logs.
`run.json` records arguments and SHA-256 identities of the inputs, model and
executables. Each native solve saves its command in `command.json`.
Iteration adds `iteration.json` and each trial's guide, curve, and score
under `trials/`. Temporary count/evidence tables are removed after processes stop.

LQF and LQW need a trained model. The included
`models/all-frequency.json` is selected automatically; use `--model FILE` for a custom model.

Original CAMUS uses its count-based quartet filter (mode 2, threshold 0.5)
and the C++ search. It does not load or apply the neural network.

**Inputs and method**

| Argument | Meaning and default |
| --- | --- |
| `--genes FILE` | Required. Gene trees, one Newick tree per line. |
| `--guide FILE` | Required. Binary guide Newick on the gene-tree taxa. |
| `--guide-rooting {outgroup,supplied}` | Default: `outgroup`, requiring `OUT` as a root child. `supplied` starts from the root encoded in the guide Newick, keeps the taxon names, and applies no outgroup constraint. |
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
| `--inner-solver {per-k-stopping,overall}` | Default: `per-k-stopping` (heuristic early stopping). `overall` selects exact inner optimization and requires `--iterate` with outgroup rooting. |
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

## Reproducibility

After installing the package, repeat the documented methods in fresh directories:

```bash
python scripts/check_reproducibility.py --out ../camus-results/reproducibility
```

The check compares complete candidate curves and selected networks for Original
CAMUS, LQF, LQW, and iteration. It also exercises supplied-root LQW without
`OUT`. Each run counts its own quartets and cleans its temporary tables.
`report.json` records package versions and input/model hashes.

This checks repeatability in the recorded environment. With the training extra
installed, add `--training` to check repeated fitting and agreement between CSV
and direct training; see [training reproducibility](training.md#reproducible-fits).
The [README](../README.md#reproducibility-and-tests) gives the test command.
