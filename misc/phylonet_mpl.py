#!/usr/bin/env python3
# Install an OpenJDK 17 runtime. On Debian/Ubuntu:
#   sudo apt-get install openjdk-17-jre-headless
# Download the PhyloNet 3.8.2 JAR used in the comparisons, from the repository root:
#   mkdir -p vendor
#   curl -fL https://phylogenomics.rice.edu/media/PhyloNet.jar -o vendor/PhyloNet-3.8.2.jar
# The runner checks the exact 3.8.2 JAR checksum before execution.
# Run `python misc/phylonet_mpl.py --help` for inputs and options.
"""Infer a network with PhyloNet 3.8.2 maximum pseudolikelihood (MPL)."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camus_nn.rooting import root_on_outgroup, tree_graph

JAR_SHA256 = 'eefabc198f113d3c835d67a44206203382e99a582d199478cc53f0bb746658fb'


def parse_arguments():
    """Read the inputs and native InferNetwork_MPL options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gene-trees', type=Path, required=True, help='Gene trees, one Newick per line')
    parser.add_argument('--starting-tree', type=Path, required=True, help='Supplied starting tree in Newick')
    parser.add_argument('--outgroup', default='OUT',
                        help='Root each gene tree and starting tree on this taxon (default: OUT)')
    parser.add_argument('--max-reticulations', type=int, required=True,
                        help='Nonnegative maximum reticulation count')
    parser.add_argument('--output-directory', type=Path, required=True,
                        help='Fresh directory for network.nwk, input.nex, and phylonet.log')
    parser.add_argument('--runs', type=int, default=10, help='Native search restarts, passed as -x (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Nonnegative native search seed, passed as -rs (default: 42)')
    parser.add_argument('--threads', type=int, default=8,
                        help='Native likelihood threads, passed as -pl (default: 8)')
    parser.add_argument('--heap', default='6g', help='Java maximum heap, passed as -Xmx (default: 6g)')
    parser.add_argument('--java', default='java', help='Java executable (default: java on PATH)')
    parser.add_argument('--jar', type=Path, default=ROOT / 'vendor/PhyloNet-3.8.2.jar',
                        help='Pinned JAR (default: vendor/PhyloNet-3.8.2.jar)')
    args = parser.parse_args()
    if args.max_reticulations < 0:
        parser.error('--max-reticulations must be nonnegative')
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    if min(args.runs, args.threads) < 1:
        parser.error('--runs and --threads must be positive')
    if not args.outgroup:
        parser.error('--outgroup must be a nonempty taxon label')
    args.gene_trees = args.gene_trees.resolve(strict=True)
    args.starting_tree = args.starting_tree.resolve(strict=True)
    args.jar = args.jar.resolve(strict=True)
    with args.jar.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != JAR_SHA256:
            parser.error('JAR differs from PhyloNet 3.8.2; follow the installation comments')
    return args


def write_nexus_input(args, destination):
    """Root topology-only copies on the outgroup before building the MPL input."""
    genes = [line.strip() for line in args.gene_trees.read_text().splitlines() if line.strip()]
    starting = args.starting_tree.read_text().strip()
    if not genes or any(not tree.endswith(';') for tree in [starting, *genes]):
        raise ValueError('Provide one complete starting Newick and one complete gene Newick per line')
    starting = root_on_outgroup(starting, args.outgroup)
    guide = tree_graph(starting)
    taxa = {guide.nodes[u]['label'] for u in guide if guide.out_degree(u) == 0}
    genes = [root_on_outgroup(tree, args.outgroup, taxa) for tree in genes]
    (destination.parent / 'genes-rooted.nwk').write_text('\n'.join(genes) + '\n')
    (destination.parent / 'starting-rooted.nwk').write_text(starting + '\n')

    search_command = (
        f'InferNetwork_MPL (all) {args.max_reticulations}'
        f' -s st -pl {args.threads} -rs {args.seed} -x {args.runs};'
    )
    lines = [
        '#NEXUS',
        'BEGIN NETWORKS;',
        f'Network st = {starting}',
        'END;',
        'BEGIN TREES;',
    ]
    lines.extend(f'Tree gt{i} = {tree}' for i, tree in enumerate(genes))
    lines.extend(['END;', 'BEGIN PHYLONET;', search_command, 'END;', ''])
    destination.write_text('\n'.join(lines))


def run_phylonet(args, nexus, output):
    """Run the pinned JAR and retain its complete search output."""
    command = [
        args.java,
        f'-XX:ActiveProcessorCount={args.threads}',
        f'-Xmx{args.heap}',
        '-jar', str(args.jar),
        str(nexus),
    ]
    log = output / 'phylonet.log'
    with log.open('w') as stream:
        subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return log


def select_best_network(log):
    """Choose the fitted network with the highest reported log pseudolikelihood."""
    candidates = re.findall(
        r'Inferred Network #\d+:\s*([^\n]+;)\s*Total log probability:\s*([-+\deE.]+)',
        log.read_text(),
    )
    if not candidates:
        raise RuntimeError('PhyloNet returned no fitted network; inspect phylonet.log')
    network, _ = max(candidates, key=lambda row: float(row[1]))
    return network


def main():
    args = parse_arguments()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)

    nexus = output / 'input.nex'
    write_nexus_input(args, nexus)
    log = run_phylonet(args, nexus, output)
    network = output / 'network.nwk'
    network.write_text(select_best_network(log) + '\n')
    print(network)


if __name__ == '__main__':
    main()
