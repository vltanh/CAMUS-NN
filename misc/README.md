# Other inference methods

Run these scripts from the repository root with Python 3.11. Each script's
opening comments give installation commands for the versions used. Install
the dependencies for the method you want, then run an example below.

Inputs are gene trees (one complete Newick per line) and a supplied starting
tree in Newick, with matching taxon labels. The examples use the included
IQ-TREE gene trees and ASTRAL guide. Choose a new output directory for every run.

## Original CAMUS

```bash
python misc/camus_original.py \
  --gene-trees examples/validation/genes.nwk \
  --starting-tree examples/validation/guide.nwk \
  --output-directory ../camus-results/camus-original
```

This runs original CAMUS with the included C++ implementation, using every
quartet, filter mode 2, and threshold 0.5. The NN is unused. The runner calls
`camus-nn --method orig` and saves `network.nwk`, the full candidate curve in
`network.csv`, `prediction.json`, native metrics, and logs.

**Inputs and output**

| Argument | Meaning and default |
| --- | --- |
| `--gene-trees FILE` | Required. Gene trees, one Newick per line. |
| `--starting-tree FILE` | Required. Binary guide Newick on the gene-tree taxa. |
| `--guide-rooting {outgroup,supplied}` | Default: `outgroup`, with `OUT` at the guide root. `supplied` starts from the guide's represented root, keeps the taxon names, and applies no outgroup constraint. |
| `--output-directory DIR` | Required. Fresh output directory. |

<details>
<summary>Click to show runtime options</summary>

| Argument | Meaning and default |
| --- | --- |
| `--threads N` | Positive C++ counting/search thread count. Default: `1`. |
| `--timeout SECONDS` | Whole-run limit in `(0, 86400]`. Default: `86400`. |
| `-h, --help` | Show usage and argument descriptions. |

</details>

For All, Rooted, Uniform, EU, or EW quartet sampling, use the package CLI's
`--sampler` argument with `--method orig`; see the [inference options](../docs/inference.md).

## BROOQS and TOB-QMC

Estimate a tree of blobs (ToB) from the supplied guide, then resolve it with
BROOQS:

```bash
python misc/brooqs.py \
  --gene-trees examples/validation/genes.nwk \
  --starting-tree examples/validation/guide.nwk \
  --outgroup OUT --rooted --tob-search 3f1a \
  --output-directory ../camus-results/brooqs

# Exhaustive ToB search instead of 3-fix-1-alter search.
python misc/brooqs.py \
  --gene-trees examples/validation/genes.nwk \
  --starting-tree examples/validation/guide.nwk \
  --outgroup OUT --rooted --tob-search exhaustive \
  --output-directory ../camus-results/brooqs-exhaustive

# Resolve an existing ToB without repeating its estimation.
python misc/brooqs.py \
  --gene-trees examples/validation/genes.nwk \
  --tree-of-blobs ../camus-results/brooqs/tree-of-blobs.nwk --rooted \
  --output-directory ../camus-results/brooqs-resolve
```

The runner uses TREE-QMC 4.1.5 revision `53b15e5c` and BROOQS revision
`17b25b2e`. It runs TOB-QMC's blob search on the supplied guide, then runs
BROOQS with minimum blob size 4. The default omits BROOQS `--rooted` and
TOB-QMC `--root`, matching the published BROOQS-data comparison: the supplied
ASTRAL tree guides the blob search, without assuming a biological outgroup.
For a known outgroup, add `--outgroup OUT`; add `--rooted` only when the ToB
root should constrain BROOQS resolution. The examples above use the known
`OUT` in the included data. For published BROOQS inputs without a known outgroup:

```bash
python misc/brooqs.py --gene-trees gene-trees.nwk --starting-tree astral.nwk \
  --tob-search exhaustive --output-directory ../camus-results/brooqs-astral
```

**Inputs, output, and search mode**

| Argument | Meaning and default |
| --- | --- |
| `--gene-trees FILE` | Required. Gene trees, one Newick per line. |
| `--starting-tree FILE` | Supplied guide for ToB estimation. Use this or `--tree-of-blobs`. |
| `--tree-of-blobs FILE` | Existing ToB. Use this or `--starting-tree`. |
| `--tob-search {exhaustive,3f1a}` | Required with `--starting-tree`; omitted with `--tree-of-blobs`. Exhaustive search passes `--iter_limit_blob 0`; 3f1a passes `--3f1a`. |
| `--output-directory DIR` | Required. Fresh directory for the inferred network and logs. |

<details>
<summary>Click to show TOB-QMC settings</summary>

| Argument | Meaning and default |
| --- | --- |
| `--outgroup LABEL` | Optional known outgroup passed to ToB estimation. Default: no explicit rooting taxon. |
| `--rooted` | Constrain BROOQS to the supplied ToB root. Default: off. |
| `--alpha NUMBER` | ToB quartet tree-test significance level. Default: `1e-7`; strictly between 0 and 1. |
| `--beta NUMBER` | ToB quartet star-test significance level. Default: `0.95`; strictly between 0 and 1. |
| `--polytomy-seed INTEGER` | Nonnegative TOB-QMC polytomy-resolution seed. Default: `12345`. |
| `--maxcut-seed INTEGER` | Nonnegative TOB-QMC max-cut seed. Default: `1`. |
| `--tob-qmc FILE` | Executable. Default: `vendor/TREE-QMC/build/tree-qmc`. |

</details>

<details>
<summary>Click to show BROOQS settings and help</summary>

| Argument | Meaning and default |
| --- | --- |
| `--seed INTEGER` | BROOQS seed in `[0, 2^32)`. Default: `1142`. |
| `--brooqs-script FILE` | Pinned resolver source. Default: `vendor/BROOQS/count_quartets.py`; its checksum is checked. |
| `-h, --help` | Show usage and argument descriptions. |

</details>

Output includes `network.nwk` and `brooqs.log`. ToB estimation also saves
`tree-of-blobs.nwk` and `tob-qmc.log`.

## SNaQ

```bash
python misc/snaq.py \
  --gene-trees examples/test/genes.nwk \
  --starting-tree examples/test/guide.nwk \
  --max-reticulations 1 --runs 32 --seed 42 \
  --search-workers 8 --threads 4 --max-failures 75 \
  --output-directory ../camus-results/snaq
```

Use Julia 1.11.6 with the included environment, which locks SNaQ 1.2.1,
PhyloNetworks 1.4.0, DataFrames 1.8.2, and their dependencies. The script
counts quartet concordance factors from the gene trees and passes them to
native `snaq!`. Defaults use 32 searches across eight search processes,
four Julia threads each, and single-thread BLAS. Set the search count and
process/thread layout to match the run being reproduced.

**Inputs, output, and reticulation limit**

| Argument | Meaning and default |
| --- | --- |
| `--gene-trees FILE` | Required. Gene trees, one Newick per line. |
| `--starting-tree FILE` | Required. Starting tree in Newick. |
| `--max-reticulations N` | Required. Nonnegative native `hmax` search limit; the returned network may contain fewer reticulations. |
| `--output-directory DIR` | Required. Fresh directory for the network and native search files. |

<details>
<summary>Click to show search options</summary>

| Argument | Meaning and default |
| --- | --- |
| `--runs N` | Positive independent search count. Default: `32`. |
| `--seed INTEGER` | Nonnegative native search seed. Default: `42`. |
| `--outgroup LABEL` | Final rooting taxon. Default: `OUT`; pass `--outgroup ''` to omit the constraint. |
| `--max-failures N` | Positive native `Nfail` stopping parameter. Default: `75`. |

</details>

<details>
<summary>Click to show worker, thread, and Julia options</summary>

| Argument | Meaning and default |
| --- | --- |
| `--search-workers N` | Parallel search processes. Default: `8`; `0` runs searches in the main process. |
| `--threads N` | Positive Julia threads per search process. Default: `4`. With workers enabled, the main process uses one Julia thread. |
| `--julia COMMAND` | Julia executable. Default: `julia` on `PATH`. |
| `--julia-project DIR` | Julia environment. Default: the included `misc/snaq-env`. |
| `-h, --help` | Show usage and argument descriptions. |

</details>

Output includes `network.nwk`, `snaq.stdout.log`, and native files beginning
with `snaq`, including search logs and candidate networks.

## PhyloNet-MPL

```bash
python misc/phylonet_mpl.py \
  --gene-trees examples/test/genes.nwk \
  --starting-tree examples/test/guide.nwk \
  --max-reticulations 1 --runs 10 --seed 42 \
  --threads 8 --heap 6g --output-directory ../camus-results/phylonet-mpl
```

This runs PhyloNet 3.8.2 with OpenJDK 17 using native `InferNetwork_MPL`:
ten searches (`-x 10`), seed 42 (`-rs 42`), and eight likelihood threads
(`-pl 8`). The supplied tree starts the search; its topology is allowed to
change. Before inference, the script roots every gene tree and the starting
tree on the `OUT` incident edge, retaining OUT and the original unrooted
topologies. Set `--outgroup` for another taxon. Missing outgroups and mismatched
taxon sets are errors. MPL uses rooted triplets, so rooting only the starting
tree is insufficient. Lengths/supports are omitted because MPL uses topologies.
The script saves `genes-rooted.nwk` and `starting-rooted.nwk` alongside
`input.nex`; the supplied input files are preserved.
The script selects the fitted network with the highest reported log
pseudolikelihood. PhyloNet results can vary between runs even with the same
seed, so matching these settings does not guarantee identical output.

**Inputs, output, and reticulation limit**

| Argument | Meaning and default |
| --- | --- |
| `--gene-trees FILE` | Required. Gene trees, one Newick per line. |
| `--starting-tree FILE` | Required. Starting tree in Newick. |
| `--outgroup TAXON` | Root every input tree on this taxon; default: `OUT`. |
| `--max-reticulations N` | Required. Nonnegative native reticulation limit; the returned network may contain fewer reticulations. |
| `--output-directory DIR` | Required. Fresh directory for `network.nwk`, `input.nex`, and `phylonet.log`. |

<details>
<summary>Click to show search options</summary>

| Argument | Meaning and default |
| --- | --- |
| `--runs N` | Positive native search restart count (`-x`). Default: `10`. |
| `--seed INTEGER` | Nonnegative native search seed (`-rs`). Default: `42`. |

</details>

<details>
<summary>Click to show thread, memory, and Java options</summary>

| Argument | Meaning and default |
| --- | --- |
| `--threads N` | Positive likelihood thread count (`-pl` and Java's active processor count). Default: `8`. |
| `--heap SIZE` | Java maximum heap (`-Xmx`), such as `6g`. Default: `6g`. |
| `--java COMMAND` | Java executable. Default: `java` on `PATH`. |
| `--jar FILE` | PhyloNet JAR. Default: `vendor/PhyloNet-3.8.2.jar`; its checksum is checked. |
| `-h, --help` | Show usage and argument descriptions. |

</details>
