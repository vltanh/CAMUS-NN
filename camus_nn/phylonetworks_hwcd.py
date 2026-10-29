"""Exact PhyloNetworks 1.4.0 HWCD, with one Julia worker per scoring batch.

The default is hardwiredclusterdistance(truth, estimate, false): minimum
distance over compatible internal roots, retaining hybrid-edge directions.
This module does not replace the historical distinct-cluster-set metric.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import shutil
import subprocess
import time

IMPLEMENTATION = "PhyloNetworks.hardwiredclusterdistance"
VERSION = "1.4.0"
METRIC_VERSIONS = {
    False: "phylonetworks-hwcd/v1-semidirected",
    True: "phylonetworks-hwcd/v1-rooted",
}


class PhyloNetworksHWCD:
    """Context-managed scorer with logs and receipts in the chosen work directory.

    Julia starts lazily on the first score. A malformed pair is reported as a
    ValueError without ending the worker; a timeout kills its whole process
    group. A subsequent pair starts a fresh Julia worker if necessary.
    """

    def __init__(self, *, workdir, julia=None, julia_project=None,
                 depot=None, timeout=3600, startup_timeout=180):
        selected = str(julia or shutil.which("julia") or "julia")
        self.julia = str(Path(selected).resolve()) if os.sep in selected else selected
        self.julia_project = Path(julia_project).resolve() if julia_project is not None else None
        self.depot = Path(depot).resolve() if depot is not None else None
        if not all(math.isfinite(v) and v > 0 for v in (timeout, startup_timeout)):
            raise ValueError("HWCD timeouts must be positive and finite")
        self.timeout, self.startup_timeout = timeout, startup_timeout
        self.directory = Path(workdir).resolve() / f"run-{time.time_ns()}-{os.getpid()}"
        self.driver = Path(__file__).with_suffix(".jl")
        self.process = self.log = self.selector = None
        self.buffer = b""
        self.sequence = 0
        self.runtime = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _start(self):
        if not shutil.which(self.julia):
            raise ValueError("Julia is unavailable; install Julia or supply --hwcd-julia")
        if self.julia_project is not None and not (self.julia_project / "Project.toml").is_file():
            raise ValueError("Julia project is unavailable; supply --hwcd-project")
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / "tmp"
        temporary.mkdir(exist_ok=True)
        env = dict(os.environ, JULIA_NUM_THREADS="1", JULIA_IMAGE_THREADS="1",
                   JULIA_PKG_PRECOMPILE_AUTO="0", OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TMPDIR=str(temporary),
                   TMP=str(temporary), TEMP=str(temporary))
        if self.depot is not None:
            env["JULIA_DEPOT_PATH"] = str(self.depot)
        command = [self.julia, "--startup-file=no", "--compiled-modules=existing",
                   "--pkgimages=existing"]
        if self.julia_project is not None:
            command.append("--project=" + str(self.julia_project))
        command.append(str(self.driver))
        self.log = (self.directory / "julia.log").open("ab")
        try:
            self.process = subprocess.Popen(command, cwd=self.directory, env=env,
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, start_new_session=True)
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.process.stdout, selectors.EVENT_READ)
            self.buffer = b""
            ready = self._readline(self.startup_timeout).split("\t")
            if len(ready) != 4 or ready[:2] != ["READY", VERSION]:
                raise ValueError("Unexpected PhyloNetworks HWCD worker/version response")
            self.runtime = dict(implementation=IMPLEMENTATION, phylonetworks_version=ready[1],
                                julia_version=ready[2], phylonetworks_source_sha256=ready[3],
                                driver_sha256=hashlib.sha256(self.driver.read_bytes()).hexdigest(),
                                log=str(self.directory / "julia.log"))
            if self.julia_project is not None:
                self.runtime["environment_hashes"] = {
                    name: hashlib.sha256((self.julia_project / name).read_bytes()).hexdigest()
                    for name in ("Project.toml", "Manifest.toml")
                    if (self.julia_project / name).is_file()
                }
            (self.directory / "runtime.json").write_text(json.dumps(
                dict(self.runtime, command=command, julia_project=str(self.julia_project) if self.julia_project else None,
                     julia_depot=str(self.depot) if self.depot else os.environ.get("JULIA_DEPOT_PATH")), indent=2) + "\n")
        except BaseException:
            self.close(kill=True, cleanup=False)
            raise

    def _readline(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise ValueError(f"PhyloNetworks HWCD timed out after {timeout:g}s; see {self.directory}")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                raise ValueError(f"PhyloNetworks HWCD worker stopped; see {self.directory / 'julia.log'}")
            self.buffer += block
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode("utf-8")

    def score(self, truth, estimate, *, rooted=False):
        """Return raw HWCD, reticulation counts and implementation provenance."""
        if not isinstance(rooted, bool):
            raise ValueError("rooted must be boolean")
        if not isinstance(truth, str) or not isinstance(estimate, str):
            raise ValueError("HWCD inputs must be Newick strings")
        self.sequence += 1
        attempt = self.directory / f"{self.sequence:06d}"
        attempt.mkdir(parents=True)
        for name, value in (("truth.nwk", truth), ("estimate.nwk", estimate)):
            (attempt / name).write_text(value)
        receipt = dict(status="evaluation_failed", rooted=rooted, timeout_seconds=self.timeout,
                       truth_sha256=hashlib.sha256(truth.encode()).hexdigest(),
                       estimate_sha256=hashlib.sha256(estimate.encode()).hexdigest())
        began = time.monotonic()
        try:
            if self.process is None:
                self._start()
            request = f"{self.sequence}\t{str(rooted).lower()}\t{truth.encode().hex()}\t{estimate.encode().hex()}\n"
            try:
                self.process.stdin.write(request.encode())
                self.process.stdin.flush()
                result = self._readline(self.timeout).split("\t")
                if result[:2] != ["RESULT", str(self.sequence)]:
                    raise ValueError("Unexpected HWCD response identity")
                if len(result) == 4 and result[2] == "error":
                    raise _PairError(bytes.fromhex(result[3]).decode("utf-8"))
                if len(result) != 6 or result[2] != "ok":
                    raise ValueError("Invalid HWCD worker response")
                distance, true_k, inferred_k = map(int, result[3:])
                if min(distance, true_k, inferred_k) < 0:
                    raise ValueError("Invalid negative HWCD result")
            except _PairError as exc:
                raise ValueError(str(exc)) from exc
            except (ValueError, OSError):
                self.close(kill=True, cleanup=False)
                raise
            provenance = dict(self.runtime, rooted=rooted, receipt=str(attempt / "receipt.json"))
            scores = dict(hardwired_distance=distance, true_k=true_k, inferred_k=inferred_k)
            receipt.update(status="ok", scores=scores, implementation=provenance)
            return dict(scores, metric_parameters_json=json.dumps(provenance, sort_keys=True))
        except (ValueError, OSError) as exc:
            receipt["error"] = str(exc)
            raise ValueError(f"{exc}; see {attempt / 'receipt.json'}") from exc
        finally:
            receipt["wall_seconds"] = time.monotonic() - began
            (attempt / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            if self.process is None:
                shutil.rmtree(self.directory / "tmp", ignore_errors=True)

    def close(self, *, kill=False, cleanup=True):
        """Stop the worker before releasing its pipes or log."""
        process = self.process
        if process is not None:
            if process.poll() is None:
                if not kill:
                    try:
                        process.stdin.close()
                        process.wait(timeout=5)
                    except (OSError, subprocess.TimeoutExpired):
                        kill = True
                if kill and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            # Stop any remaining descendants before removing owned temporaries.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for pipe in (process.stdin, process.stdout):
                if pipe is not None and not pipe.closed:
                    pipe.close()
        if self.selector is not None:
            self.selector.close()
        if self.log is not None:
            self.log.close()
        self.process = self.log = self.selector = None
        self.buffer = b""
        if cleanup:
            shutil.rmtree(self.directory / "tmp", ignore_errors=True)


class _PairError(Exception):
    """A bad input pair leaves the Julia protocol usable for later pairs."""

