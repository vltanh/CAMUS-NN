# Training

[README](../README.md) · [Inference](inference.md) · [Evaluation](evaluation.md) · [Training](training.md)

Run these commands from the repository root. Install fitting support first:

```bash
uv pip install --torch-backend=cpu '.[train]'
```

## Preprocessing

Convert a list of true networks and gene-tree files into a reusable labeled
training table:

```bash
camus-nn-prepare --training-cases examples/training/cases.json \
  --output-table ../camus-results/training/quartets.csv
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

## Fitting

Fit from the table prepared above, or fit directly from the case list.
Both input modes use the same CSV trainer; direct case input prepares a
temporary table internally.

```bash
# Fit from labeled quartet rows.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  camus-nn-train --quartet-table ../camus-results/training/quartets.csv \
  --output-model ../camus-results/training/model.json --seed 42

# Fit directly from networks and gene trees.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  camus-nn-train --training-cases examples/training/cases.json \
  --output-model ../camus-results/training/direct-model.json --seed 42

# Use a newly fitted model for inference.
camus-nn --method lqw --model ../camus-results/training/model.json \
  --genes examples/validation/genes.nwk \
  --guide examples/validation/guide.nwk --out ../camus-results/new-model
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

## Input formats

### Training case list (JSON)

The file is a nonempty JSON array with one object per training network.
The included [examples/training/cases.json](../examples/training/cases.json) is:

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

## Reproducible fits

CPU math libraries can affect refits. For matching refits on AVX2-capable
CPUs, also set `ATEN_CPU_CAPABILITY=avx2`, `OPENBLAS_CORETYPE=Zen`, and
`MKL_ENABLE_INSTRUCTIONS=AVX2` with the single-thread settings shown above.
The single training example exercises the commands; recreating the bundled
model requires its full original training cohort, not this one example.

With the training extra installed, check the example fits with:

```bash
python scripts/check_reproducibility.py --training \
  --out ../camus-results/reproducibility-training
```

This also repeats inference. Repeated seeded fits and CSV/direct training must
match in the recorded environment. See [inference reproducibility](inference.md#reproducibility)
for the checks performed. Tests can be run with:

```bash
python -m unittest discover -s tests -v
```
