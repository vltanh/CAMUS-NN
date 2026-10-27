#!/usr/bin/env python3
# Install Julia 1.11.6: https://julialang.org/downloads/oldreleases/
# With `julia` on PATH, run from the repository root:
#   julia --project=misc/snaq-env -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
# The included environment locks SNaQ 1.2.1, PhyloNetworks 1.4.0,
# DataFrames 1.8.2, and their dependencies to the versions used.
# Run `python misc/snaq.py --help` for inputs and options.
"""Infer a network with SNaQ from gene trees and a supplied starting tree."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SNAQ_PROGRAM = r'''
using Distributed

# Unpack the values passed by run_snaq below.
genes, starting, output = ARGS[1:3]
hmax, runs, seed = parse.(Int, ARGS[4:6])
outgroup = ARGS[7]
fail_limit, worker_count, worker_threads = parse.(Int, ARGS[8:10])

# Each distributed search process gets its own Julia threads and single-thread BLAS.
if worker_count > 0
    worker_options = Cmd([
        "--startup-file=no",
        "--threads=$worker_threads",
        "--project=$(dirname(Base.active_project()))",
    ])
    addprocs(worker_count; exeflags=worker_options)
end
@everywhere using PhyloNetworks, SNaQ, DataFrames, LinearAlgebra
@everywhere BLAS.set_num_threads(1)

pkgversion(SNaQ) == v"1.2.1" || error("Install SNaQ 1.2.1 as shown in the script comments")
pkgversion(PhyloNetworks) == v"1.4.0" || error("Install PhyloNetworks 1.4.0")
pkgversion(DataFrames) == v"1.8.2" || error("Install DataFrames 1.8.2")

# Count quartet concordance factors, then run the native SNaQ search.
gene_trees = readmultinewick(genes)
quartets, taxa = countquartetsintrees(gene_trees)
quartet_table = DataFrame(tablequartetCF(quartets, taxa), copycols=false)
concordance_factors = readtableCF(quartet_table)
rooting_options = isempty(outgroup) ? (;) : (; outgroup=outgroup)
network = snaq!(
    readnewick(starting), concordance_factors;
    hmax=hmax,
    runs=runs,
    seed=seed,
    Nfail=fail_limit,
    filename=joinpath(output, "snaq"),
    rooting_options...,
)
writenewick(network, joinpath(output, "network.nwk"))
worker_count > 0 && rmprocs(workers())
'''


def parse_arguments():
    """Read the input paths and native search settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gene-trees', type=Path, required=True, help='Gene trees, one Newick per line')
    parser.add_argument('--starting-tree', type=Path, required=True, help='Supplied starting tree in Newick')
    parser.add_argument('--max-reticulations', type=int, required=True,
                        help='Nonnegative maximum reticulation count')
    parser.add_argument('--output-directory', type=Path, required=True,
                        help='Fresh directory for network.nwk and native search files')
    parser.add_argument('--runs', type=int, default=32, help='Independent native searches (default: 32)')
    parser.add_argument('--seed', type=int, default=42, help='Nonnegative search seed (default: 42)')
    parser.add_argument('--outgroup', default='OUT',
                        help='Outgroup label; empty string omits the constraint (default: OUT)')
    parser.add_argument('--max-failures', type=int, default=75,
                        help='Native Nfail stopping parameter (default: 75)')
    parser.add_argument('--search-workers', type=int, default=8,
                        help='Parallel search processes; 0 uses the main process (default: 8)')
    parser.add_argument('--threads', type=int, default=4, help='Julia threads per search process (default: 4)')
    parser.add_argument('--julia', default='julia', help='Julia executable (default: julia on PATH)')
    parser.add_argument('--julia-project', type=Path, default=ROOT / 'misc/snaq-env',
                        help='Pinned Julia environment directory (default: misc/snaq-env)')
    args = parser.parse_args()
    if args.max_reticulations < 0:
        parser.error('--max-reticulations must be nonnegative')
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    if args.search_workers < 0:
        parser.error('--search-workers must be nonnegative')
    if min(args.runs, args.threads, args.max_failures) < 1:
        parser.error('--runs, --threads, and --max-failures must be positive')
    args.gene_trees = args.gene_trees.resolve(strict=True)
    args.starting_tree = args.starting_tree.resolve(strict=True)
    args.julia_project = args.julia_project.resolve(strict=True)
    if not (args.julia_project / 'Project.toml').is_file():
        parser.error('Julia project is missing Project.toml; follow the installation comments')
    return args


def run_snaq(args, output):
    """Launch Julia with the same worker/thread layout as the native searches."""
    master_threads = 1 if args.search_workers else args.threads
    command = [
        args.julia,
        '--startup-file=no',
        f'--threads={master_threads}',
        f'--project={args.julia_project}',
        '-e', SNAQ_PROGRAM, '--',
        str(args.gene_trees),
        str(args.starting_tree),
        str(output),
        str(args.max_reticulations),
        str(args.runs),
        str(args.seed),
        args.outgroup,
        str(args.max_failures),
        str(args.search_workers),
        str(args.threads),
    ]
    with (output / 'snaq.stdout.log').open('w') as log:
        subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
    network = output / 'network.nwk'
    if not network.read_text().strip():
        raise RuntimeError('SNaQ produced an empty network; inspect snaq.stdout.log')
    return network


def main():
    args = parse_arguments()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)

    network = run_snaq(args, output)
    print(network)


if __name__ == '__main__':
    main()
