#!/usr/bin/env python3
# Install this checkout from the repository root with Python 3.11 and GCC 11+:
#   uv venv --python 3.11
#   source .venv/bin/activate
#   uv pip install --torch-backend=cpu .
# Installation builds the included C++ counter and inference core.
# Original CAMUS uses quartet filter mode 2 and threshold 0.5, without an NN.
# Run `python misc/camus_original.py --help` for inputs and options.
"""Run original CAMUS on all quartets using the installed C++ implementation."""
import argparse
from pathlib import Path
import subprocess


def parse_arguments():
    """Read the input trees and output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gene-trees', type=Path, required=True,
                        help='Gene trees, one Newick per line')
    parser.add_argument('--starting-tree', type=Path, required=True,
                        help='Binary guide Newick on the gene-tree taxa')
    parser.add_argument('--guide-rooting',choices=('outgroup','supplied'),default='outgroup',
                        help='Use a known OUT root or preserve the supplied root and names (default: outgroup)')
    parser.add_argument('--output-directory', type=Path, required=True,
                        help='Fresh directory for network.nwk, the full candidate curve, and logs')
    parser.add_argument('--threads', type=int, default=1,
                        help='Positive C++ counting/search thread count (default: 1)')
    parser.add_argument('--timeout', type=float, default=86400,
                        help='Whole-run seconds, greater than 0 and at most 86400 (default: 86400)')
    return parser.parse_args()


def run_camus(args):
    """Use the package's original CAMUS pipeline and its output selection."""
    command = [
        'camus-nn',
        '--method', 'orig',
        '--genes', str(args.gene_trees),
        '--guide', str(args.starting_tree),
        '--guide-rooting',args.guide_rooting,
        '--out', str(args.output_directory),
        '--threads', str(args.threads),
        '--timeout', str(args.timeout),
    ]
    subprocess.run(command, check=True)


def main():
    args = parse_arguments()
    run_camus(args)


if __name__ == '__main__':
    main()
