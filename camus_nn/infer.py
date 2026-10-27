"""Run original CAMUS, CAMUS-LQF or CAMUS-LQW with the C++ core."""
import argparse
from collections import Counter
import csv
import json
import math
import os
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path
import signal
import subprocess
import sys
import time

from .io import save

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
DEFAULT_MODEL = PACKAGE / 'models/all-frequency.json'
if not DEFAULT_MODEL.is_file():
    DEFAULT_MODEL = ROOT / 'models/all-frequency.json'
DEFAULT_BIN_DIR = PACKAGE / 'bin'
if not (DEFAULT_BIN_DIR / 'camus-native').is_file():
    DEFAULT_BIN_DIR = ROOT / 'build'
if not (DEFAULT_BIN_DIR / 'camus-native').is_file():
    try:
        DEFAULT_BIN_DIR = Path(distribution('camus-nn').locate_file('camus_nn/bin'))
    except PackageNotFoundError:
        pass


def choose(path, method, *, iterative=False):
    column = 'Signed Quartet Gain' if method == 'lqw' else 'Quartet Satisfied Percent'
    with path.open(newline='') as stream:
        rows = [dict(k=int(r['Number of Branches']), objective=float(r[column]),
                     network=r['Extended Newick']) for r in csv.DictReader(stream)]
    if not rows or any(r['k'] < 0 or not math.isfinite(r['objective']) for r in rows):
        raise ValueError('Invalid native candidates')
    best = max(r['objective'] for r in rows)
    tolerance = 1e-9 if method == 'lqw' and not iterative else 0
    return min((r for r in rows if best - r['objective'] <= tolerance), key=lambda r: (r['k'], r['network']))


def worker(output):
    from .evidence import write_evidence
    from .iterate import anchored_tree, refine
    from .sampling import sample_quartets

    config = json.loads((output / 'tmp/config.json').read_text())
    files = config['files']
    tree = anchored_tree(Path(files['guide']).read_text())
    # Normalize annotation-free topology once, preserving child encounter order.
    guide = output / 'guide.nwk'
    guide.write_text(tree.newick() + '\n')
    temporary = output / 'tmp'
    counts = temporary / 'counts.csv'
    multiplicities = None
    phases = {}
    began = time.monotonic()
    sampler = {'eu': 'edge-uniform', 'ew': 'edge-walk'}.get(config['sampler'], config['sampler'])
    argv = [files['counter'], '-n', str(config['threads']),
            '-gene-trees', files['genes'], '-o', str(counts)]
    if sampler != 'all':
        import hashlib
        seed = int.from_bytes(hashlib.sha256(f"{config['seed']}:{config['case']}".encode()).digest()[:8], 'little')
        draws = sample_quartets(tree, sampler, seed, multiplicity=config['multiplicity'])
        multiplicities = Counter(draws)
        selection = temporary / 'selection.csv'
        with selection.open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['t0', 't1', 't2', 't3'])
            writer.writerows(multiplicities)
        argv.extend(['-quartet-sets', str(selection)])
    with (output / 'count.log').open('w') as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    phases['sampling_and_counting'] = time.monotonic() - began
    start = time.monotonic()
    model = None
    if config['method'] != 'orig':
        import torch
        from .io import load_model
        torch.set_num_threads(1)
        model = load_model(files['model'])
    evidence = temporary / 'evidence.csv'
    rows = write_evidence(counts, evidence, model, config['method'], tree.tips,
                          multiplicities=multiplicities, batch_rows=config['batch_rows'])
    if sampler == 'all' and rows != math.comb(len(tree.tips), 4):
        raise ValueError('All requires observed resolved counts for every quartet of guide taxa')
    phase = 'count_evidence' if config['method'] == 'orig' else 'model_and_evidence'
    phases[phase] = time.monotonic() - start
    cap = config['search_cap']

    def solve(current_guide, directory, iterative=False):
        if config['method'] == 'orig':
            flags = ['-q', '2', '-t', '.5', '-grouped-counts']
        elif config['method'] == 'lqf':
            flags = ['-q', '0', '-grouped-counts']
        else:
            flags = ['-q', '0', '-quartet-weights', '-signed-weights']
            if iterative and config['inner_solver'] == 'overall' and cap is None:
                flags += ['--signed-unrestricted']
            else:
                flags += (['--anchor-outgroup'] if iterative else []) + ['-signed-max-reticulations', str(-1 if cap is None else cap),
                          '-signed-stop-on-plateau=' + ('false' if iterative and config['inner_solver'] == 'overall' else 'true')]
        command = [files['binary'], '-n', str(config['threads']), *flags,
                   '-o', str(directory / 'network'), str(current_guide), str(evidence)]
        with (directory / 'native.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = choose(directory / 'network.csv', config['method'], iterative=iterative)
        if cap is not None and result['k'] > cap:
            raise ValueError('Native result exceeded the search cap')
        return result

    start = time.monotonic()
    if config['iterate']:
        result, iteration = refine(guide.read_text(), evidence, output, solve,
            max_guide_runs=config['max_guide_runs'], max_rounds=config['max_rounds'],
            search_cap=cap, inner_solver=config['inner_solver'])
        result['guide_runs'] = len(iteration['trials'])
        result['objective_kind'] = 'total_signed_displayed_quartet_score'
    else:
        result = solve(guide, output)
        result['objective_kind'] = 'signed_quartet_gain' if config['method'] == 'lqw' else 'quartet_satisfied_percent'
    phases['native_search_and_iterations'] = time.monotonic() - start
    (output / 'network.nwk').write_text(result['network'] + '\n')
    save(output / 'prediction.json', dict(**result, method=config['method'], sampler=config['sampler'],
        search_cap=cap, quartet_groups=rows, model_used=config['method'] != 'orig',
        quartet_draws=sum(multiplicities.values()) if multiplicities is not None else rows,
        truth_used_for_inference=False, phases=phases))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--_worker':
        worker(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--genes', type=Path, required=True, help='Gene-tree file: one Newick per line')
    parser.add_argument('--guide', type=Path, required=True, help='Binary guide Newick; OUT is a root child')
    parser.add_argument('--out', type=Path, required=True, help='Fresh output directory; must not exist')
    parser.add_argument('--method', choices=('orig', 'lqf', 'lqw'), default='lqw', help='Original CAMUS or a learned method (default: lqw)')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='Trained model for lqf/lqw (default: bundled NN; unused by orig)')
    parser.add_argument('--bin-dir', type=Path, default=DEFAULT_BIN_DIR, help='Directory with camus-count and camus-native (default: bundled)')
    parser.add_argument('--sampler', choices=('all', 'rooted', 'uniform', 'eu', 'ew'), default='all', help='Quartet sampling strategy (default: all)')
    parser.add_argument('--case', default='single-case', help='Case ID combined with seed for sampling (default: single-case)')
    parser.add_argument('--seed', type=int, default=42, help='Deterministic sampling seed (default: 42)')
    parser.add_argument('--multiplicity', type=int, default=50, help='Sampling budget multiplier per guide edge for uniform/eu/ew (default: 50; positive)')
    parser.add_argument('--threads', type=int, default=1, help='C++ counting and search threads (default: 1; positive)')
    parser.add_argument('--batch-rows', type=int, default=65536, help='Quartet rows per NN/evidence batch (default: 65536; positive)')
    parser.add_argument('--iterate', action='store_true', help='Refine the guide by single-cycle switching (LQW only)')
    parser.add_argument('--max-guide-runs', type=int, default=5, help='Iteration solve budget, including the initial solve (default: 5)')
    parser.add_argument('--max-rounds', type=int, default=10, help='Outer switching-pass limit (default: 10; nonnegative)')
    parser.add_argument('--inner-solver', choices=('per-k-stopping', 'overall'), default='per-k-stopping', help='LQW inner solver: heuristic early stopping (per-k-stopping, default) or exact DP (overall, requires --iterate)')
    parser.add_argument('--search-cap', type=int, help='Nonnegative reticulation limit per LQW solve (default: uncapped)')
    parser.add_argument('--timeout', type=float, default=86400, help='Whole-run seconds (default: 86400; 0 < value <= 86400)')
    args = parser.parse_args()
    if args.iterate and args.method != 'lqw' or args.search_cap is not None and (args.search_cap < 0 or args.method != 'lqw'):
        parser.error('Iteration and a nonnegative search cap require LQW')
    if (min(args.threads, args.multiplicity, args.batch_rows, args.max_guide_runs) < 1 or args.max_rounds < 0
            or not math.isfinite(args.timeout) or not 0 < args.timeout <= 86400):
        parser.error('Require positive budgets, nonnegative rounds and a timeout in (0, 86400]')
    if not args.iterate and args.inner_solver != 'per-k-stopping':
        parser.error('--inner-solver overall requires --iterate')
    from .iterate import anchored_tree
    anchored_tree(args.guide.read_text())
    inputs = dict(genes=args.genes, guide=args.guide,
                  binary=args.bin_dir / 'camus-native', counter=args.bin_dir / 'camus-count')
    if args.method != 'orig':
        inputs['model'] = args.model
    files = {name: str(path.resolve(strict=True)) for name, path in inputs.items()}
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'tmp').mkdir()
    config = {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
    config.update(files=files)
    save(output / 'tmp/config.json', config)
    env = dict(os.environ, TMPDIR=str(output / 'tmp'), PYTHONDONTWRITEBYTECODE='1',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    began = time.monotonic()
    process = None
    status = 'failed'
    try:
        with (output / 'run.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-B', '-m', 'camus_nn.infer', '--_worker', str(output)],
                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            code = process.wait(timeout=args.timeout)
        if code:
            raise RuntimeError(f'Inference failed with exit code {code}; see {output / "run.log"}')
        status = 'complete'
    except BaseException as error:
        status = 'timeout' if isinstance(error, subprocess.TimeoutExpired) else 'failed'
        save(output / 'failure.json', dict(status=status, error=repr(error)))
        raise
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        # Regenerable tables are deleted only after stopped processes and receipts.
        import shutil
        save(output / 'completion.json', dict(status=status, wall_seconds=time.monotonic() - began))
        shutil.rmtree(output / 'tmp')
    print(output / 'network.nwk')


if __name__ == '__main__':
    main()
