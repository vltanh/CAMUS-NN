# Other inference methods

Run these scripts from the repository root with Python 3.11. Each script's
opening comments give installation commands for the versions used. Install
the dependencies for the method you want, then run an example below.

Inputs are gene trees (one complete Newick per line) and a supplied starting
tree in Newick, with matching taxon labels. The examples use the included
IQ-TREE gene trees and ASTRAL guide. Choose a new output directory for every run.

## BROOQS and TOB-QMC

Estimate a tree of blobs (ToB) from the supplied guide, then resolve it with
BROOQS:

```bash
python misc/brooqs.py \
  --gene-trees examples/validation/genes.nwk \
  --starting-tree examples/validation/guide.nwk \
  --tob-search 3f1a --output-directory out-brooqs

# Exhaustive ToB search instead of 3-fix-1-alter search.
python misc/brooqs.py \
  --gene-trees examples/validation/genes.nwk \
  --starting-tree examples/validation/guide.nwk \
  --tob-search exhaustive --output-directory out-brooqs-exhaustive

# Resolve an existing ToB without repeating its estimation.
python misc/brooqs.py \
  --gene-trees examples/validation/genes.nwk \
  --tree-of-blobs out-brooqs/tree-of-blobs.nwk \
  --output-directory out-brooqs-resolve
```

The runner uses TREE-QMC 4.1.5 revision `53b15e5c` and BROOQS revision
`17b25b2e`. It runs TOB-QMC's blob search on the supplied guide, then runs
BROOQS with rooted input and minimum blob size 4.

| Argument | Meaning and default |
| --- | --- |
| `--gene-trees FILE` | Required. Gene trees, one Newick per line. |
| `--starting-tree FILE` | Supplied guide for ToB estimation. Use this or `--tree-of-blobs`. |
| `--tree-of-blobs FILE` | Existing rooted ToB. Use this or `--starting-tree`. |
| `--tob-search {exhaustive,3f1a}` | Required with `--starting-tree`; omitted with `--tree-of-blobs`. Exhaustive search passes `--iter_limit_blob 0`; 3f1a passes `--3f1a`. |
| `--output-directory DIR` | Required. Fresh directory for the inferred network and logs. |
| `--tob-qmc FILE` | Executable. Default: `vendor/TREE-QMC/build/tree-qmc`. |
| `--outgroup LABEL` | ToB rooting taxon. Default: `OUT`. Used during ToB estimation. |
| `--alpha NUMBER` | ToB quartet tree-test significance level. Default: `1e-7`; strictly between 0 and 1. |
| `--beta NUMBER` | ToB quartet star-test significance level. Default: `0.95`; strictly between 0 and 1. |
| `--polytomy-seed INTEGER` | Nonnegative TOB-QMC polytomy-resolution seed. Default: `12345`. |
| `--maxcut-seed INTEGER` | Nonnegative TOB-QMC max-cut seed. Default: `1`. |
| `--brooqs-script FILE` | Pinned resolver source. Default: `vendor/BROOQS/count_quartets.py`; its checksum is checked. |
| `--seed INTEGER` | BROOQS seed in `[0, 2^32)`. Default: `1142`. |
| `-h, --help` | Show usage and argument descriptions. |

Output includes `network.nwk` and `brooqs.log`. ToB estimation also saves
`tree-of-blobs.nwk` and `tob-qmc.log`.

## SNaQ

```bash
python misc/snaq.py \
  --gene-trees examples/test/genes.nwk \
  --starting-tree examples/test/guide.nwk \
  --max-reticulations 1 --runs 32 --seed 42 \
  --search-workers 8 --threads 4 --max-failures 75 \
  --output-directory out-snaq
```

Use Julia 1.11.6 with the included environment, which locks SNaQ 1.2.1,
PhyloNetworks 1.4.0, DataFrames 1.8.2, and their dependencies. The script
counts quartet concordance factors from the gene trees and passes them to
native `snaq!`. Defaults use 32 searches across eight search processes,
four Julia threads each, and single-thread BLAS. Set the search count and
process/thread layout to match the run being reproduced.

| Argument | Meaning and default |
| --- | --- |
| `--gene-trees FILE` | Required. Gene trees, one Newick per line. |
| `--starting-tree FILE` | Required. Starting tree in Newick. |
| `--max-reticulations N` | Required. Nonnegative native `hmax` search limit; the returned network may contain fewer reticulations. |
| `--output-directory DIR` | Required. Fresh directory for the network and native search files. |
| `--runs N` | Positive independent search count. Default: `32`. |
| `--seed INTEGER` | Nonnegative native search seed. Default: `42`. |
| `--outgroup LABEL` | Final rooting taxon. Default: `OUT`; pass `--outgroup ''` to omit the constraint. |
| `--max-failures N` | Positive native `Nfail` stopping parameter. Default: `75`. |
| `--search-workers N` | Parallel search processes. Default: `8`; `0` runs searches in the main process. |
| `--threads N` | Positive Julia threads per search process. Default: `4`. With workers enabled, the main process uses one Julia thread. |
| `--julia COMMAND` | Julia executable. Default: `julia` on `PATH`. |
| `--julia-project DIR` | Julia environment. Default: the included `misc/snaq-env`. |
| `-h, --help` | Show usage and argument descriptions. |

Output includes `network.nwk`, `snaq.stdout.log`, and native files beginning
with `snaq`, including search logs and candidate networks.

## PhyloNet-MPL

```bash
python misc/phylonet_mpl.py \
  --gene-trees examples/test/genes.nwk \
  --starting-tree examples/test/guide.nwk \
  --max-reticulations 1 --runs 10 --seed 42 \
  --threads 8 --heap 6g --output-directory out-phylonet-mpl
```

This runs PhyloNet 3.8.2 with OpenJDK 17 using native `InferNetwork_MPL`:
ten searches (`-x 10`), seed 42 (`-rs 42`), and eight likelihood threads
(`-pl 8`). The supplied tree starts the search; its topology is allowed to
change. The script selects the fitted network with the highest reported log
pseudolikelihood. PhyloNet results can vary between runs even with the same
seed, so matching these settings does not guarantee identical output.

| Argument | Meaning and default |
| --- | --- |
| `--gene-trees FILE` | Required. Gene trees, one Newick per line. |
| `--starting-tree FILE` | Required. Starting tree in Newick. |
| `--max-reticulations N` | Required. Nonnegative native reticulation limit; the returned network may contain fewer reticulations. |
| `--output-directory DIR` | Required. Fresh directory for `network.nwk`, `input.nex`, and `phylonet.log`. |
| `--runs N` | Positive native search restart count (`-x`). Default: `10`. |
| `--seed INTEGER` | Nonnegative native search seed (`-rs`). Default: `42`. |
| `--threads N` | Positive likelihood thread count (`-pl` and Java's active processor count). Default: `8`. |
| `--heap SIZE` | Java maximum heap (`-Xmx`), such as `6g`. Default: `6g`. |
| `--java COMMAND` | Java executable. Default: `java` on `PATH`. |
| `--jar FILE` | PhyloNet JAR. Default: `vendor/PhyloNet-3.8.2.jar`; its checksum is checked. |
| `-h, --help` | Show usage and argument descriptions. |
