#!/usr/bin/env python3
# Install from the repository root, in an activated Python 3.11 environment:
#   uv pip install numpy==1.26.4 treeswift==1.1.48
#   mkdir -p vendor
#   git clone https://github.com/shayesteh99/BROOQS.git vendor/BROOQS
#   git -C vendor/BROOQS checkout --detach 17b25b2e06e5c8ef5d3e884360c0a6b69cd0a400
# This is the BROOQS source revision used in the comparisons.
#
# Install TOB-QMC (TREE-QMC 4.1.5), with R 4.6.1, CMake >= 3.18,
# a C++20 compiler, and GNU make. Install the recorded R package versions:
#   Rscript -e '
#     install.packages("remotes", repos="https://cloud.r-project.org")
#     versions <- c(Rcpp="1.1.2", RInside="0.2.19", ape="5.8.1",
#                   igraph="2.3.4", phangorn="2.12.1", MSCquartets="3.3")
#     for (package in names(versions)) {
#       remotes::install_version(package, version=versions[[package]],
#           repos="https://cloud.r-project.org", upgrade="never")
#     }
#   '
#   git clone https://github.com/molloy-lab/TREE-QMC.git vendor/TREE-QMC
#   git -C vendor/TREE-QMC checkout --detach 53b15e5c365ebee1114575c7999eb7660163a385
#   cmake -S vendor/TREE-QMC -B vendor/TREE-QMC/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build vendor/TREE-QMC/build --parallel 4
# Run `python misc/brooqs.py --help` for inputs and options.
"""Infer a tree of blobs with TOB-QMC, then resolve it with pinned BROOQS."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = '844ee3ac5ecebacb9838c5bd358036771f0d37642bb62a00f6d4a9a22bb8c662'


def parse_arguments():
    """Read the inputs, ToB search settings, and resolver seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gene-trees', type=Path, required=True, help='Gene trees, one Newick per line')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--starting-tree', type=Path, help='Supplied base tree for TOB-QMC, in Newick')
    inputs.add_argument('--tree-of-blobs', type=Path, help='Use an already estimated rooted tree of blobs')
    parser.add_argument('--output-directory', type=Path, required=True,
                        help='Fresh directory for network.nwk and brooqs.log')
    parser.add_argument('--tob-search', choices=('exhaustive', '3f1a'),
                        help='Required with --starting-tree: exact or 3-fix-1-alter ToB search')
    parser.add_argument('--tob-qmc', type=Path, default=ROOT / 'vendor/TREE-QMC/build/tree-qmc',
                        help='TOB-QMC executable (default: vendor/TREE-QMC/build/tree-qmc)')
    parser.add_argument('--outgroup', help='Known ToB rooting taxon (default: no explicit root)')
    parser.add_argument('--rooted',action='store_true',
                        help='Treat the ToB root as known when resolving it (default: unrooted)')
    parser.add_argument('--alpha', type=float, default=1e-7,
                        help='ToB tree-test significance level (default: 1e-7)')
    parser.add_argument('--beta', type=float, default=0.95,
                        help='ToB star-test significance level (default: 0.95)')
    parser.add_argument('--polytomy-seed', type=int, default=12345,
                        help='TOB-QMC polytomy-resolution seed (default: 12345)')
    parser.add_argument('--maxcut-seed', type=int, default=1, help='TOB-QMC max-cut seed (default: 1)')
    parser.add_argument('--brooqs-script', type=Path, default=ROOT / 'vendor/BROOQS/count_quartets.py',
                        help='Pinned source script (default: vendor/BROOQS/count_quartets.py)')
    parser.add_argument('--seed', type=int, default=1142, help='Nonnegative random seed (default: 1142)')
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32 or min(args.polytomy_seed, args.maxcut_seed) < 0:
        parser.error('Require a 32-bit nonnegative BROOQS seed and nonnegative ToB seeds')
    if not 0 < args.alpha < 1 or not 0 < args.beta < 1:
        parser.error('--alpha and --beta must be strictly between 0 and 1')
    if bool(args.starting_tree) != bool(args.tob_search):
        parser.error('--starting-tree and --tob-search must be provided together')
    return args


def check_inputs(args):
    """Resolve all input paths before creating the output directory."""
    args.gene_trees = args.gene_trees.resolve(strict=True)
    args.brooqs_script = args.brooqs_script.resolve(strict=True)
    source = args.brooqs_script
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('BROOQS source differs from the pinned revision; follow the installation comments')
    if args.starting_tree:
        args.starting_tree = args.starting_tree.resolve(strict=True)
        args.tob_qmc = args.tob_qmc.resolve(strict=True)
    else:
        args.tree_of_blobs = args.tree_of_blobs.resolve(strict=True)


def estimate_tree_of_blobs(args, output):
    """Run TOB-QMC's blob search on the supplied starting tree."""
    tob = output / 'tree-of-blobs.nwk'
    command = [
        str(args.tob_qmc),
        '--blob',
        '--alpha', str(args.alpha),
        '--beta', str(args.beta),
        '--polyseed', str(args.polytomy_seed),
        '--cutseed', str(args.maxcut_seed),
        '--blobsearchonly', str(args.starting_tree),
        '-i', str(args.gene_trees),
        '-o', str(tob),
    ]
    if args.outgroup:
        command += ['--root',args.outgroup]
    if args.tob_search == 'exhaustive':
        command += ['--iter_limit_blob', '0']
    else:
        command += ['--3f1a']

    with (output / 'tob-qmc.log').open('w') as log:
        subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
    return tob


def resolve_tree_of_blobs(args, tob, output):
    """Resolve the ToB; constrain its root only when explicitly requested."""
    network = output / 'network.nwk'
    command = [
        sys.executable, str(args.brooqs_script),
        '-i', str(args.gene_trees),
        '-t', str(tob),
        '-o', str(network),
        '--seed', str(args.seed),
        '--min-blob-size', '4',
    ]
    if args.rooted:
        command.append('--rooted')
    with (output / 'brooqs.log').open('w') as log:
        subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not network.read_text().strip():
        raise RuntimeError('BROOQS produced an empty network; inspect brooqs.log')
    return network


def main():
    args = parse_arguments()
    check_inputs(args)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)

    tob = args.tree_of_blobs
    if args.starting_tree:
        tob = estimate_tree_of_blobs(args, output)
    network = resolve_tree_of_blobs(args, tob, output)
    print(network)


if __name__ == '__main__':
    main()
