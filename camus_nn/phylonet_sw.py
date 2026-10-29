"""Call pinned PhyloNet CmpNets -m cluster for rooted softwired FN/FP/error.

This backend preserves the supplied roots. It is distinct from the historical
OUT-oriented split scorer; no Python cluster extraction computes its scores.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from .topology import parse_network, network_newick

VERSION = "3.8.2"
JAR_SHA256 = "eefabc198f113d3c835d67a44206203382e99a582d199478cc53f0bb746658fb"
METRIC_VERSION = "phylonet-sw/v1-rooted"
NORMALIZATION = "Shared taxon tokens; rooted topology retained; branch lengths, inheritance values and internal annotations omitted"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def topology_inputs(truth, estimate):
    """Make safe topology-only NEXUS inputs without rerooting either network."""
    graphs = [parse_network(text, allow_parallel=True) for text in (truth, estimate)]
    taxa = [{g.nodes[v]["label"] for v in g if g.out_degree(v) == 0} for g in graphs]
    if taxa[0] != taxa[1]:
        raise ValueError("Inferred and true network taxa differ")
    # Both the Rich Newick command parser and its legacy ExNewickReader see
    # only safe identifiers. This also handles quoted/Unicode taxon names.
    tokens = {name: f"SW_T{i:06d}" for i, name in enumerate(sorted(taxa[0]))}
    reticulations, normalized = [], []
    for graph in graphs:
        reticulations.append(sum(graph.in_degree(v) == 2 for v in graph))
        for v in graph:
            if graph.out_degree(v) == 0:
                graph.nodes[v]["label"] = tokens[graph.nodes[v]["label"]]
        normalized.append(network_newick(graph))
    return normalized, reticulations, {token: name for name, token in tokens.items()}


def parse_result(text):
    matches = re.findall(
        r"^The cluster-based distance between two networks:\s*(\S+)\s+(\S+)\s+(\S+)\s*$",
        text, re.MULTILINE,
    )
    if len(matches) != 1:
        raise ValueError("PhyloNet did not return exactly one CmpNets cluster result")
    fn, fp, error = map(float, matches[0])
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (fn, fp, error)):
        raise ValueError("PhyloNet returned invalid softwired rates")
    if not math.isclose(error, (fn + fp) / 2, rel_tol=0, abs_tol=1e-12):
        raise ValueError("PhyloNet returned an inconsistent mean softwired error")
    return dict(softwired_fn=fn, softwired_fp=fp, softwired_error=error)


class PhyloNetSW:
    """Score saved Newick strings with the actual PhyloNet command.

    Each pair has its own NEXUS, logs and receipt in the chosen work directory. A timeout
    kills the entire Java process group before failure is reported. No partial
    result is accepted and no inference artifact or previous score is changed.
    """

    def __init__(self, *, workdir, jar=None, java=None, heap="2g", timeout=3600):
        self.jar = Path(jar or "vendor/PhyloNet-3.8.2.jar").resolve()
        selected_java = str(java or shutil.which("java") or "java")
        self.java = str(Path(selected_java).resolve()) if os.sep in selected_java else selected_java
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("PhyloNet scoring timeout must be positive and finite")
        if not re.fullmatch(r"[1-9][0-9]*[kKmMgG]", heap):
            raise ValueError("Java heap must be a positive size such as 2g")
        base = Path(workdir).resolve()
        self.directory = base / f"run-{time.time_ns()}-{os.getpid()}"
        self.heap, self.timeout = heap, timeout
        self.sequence = 0
        self.runtime = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass  # score() always waits for its Java process to stop.

    def _check_runtime(self):
        if self.runtime:
            return
        if not self.jar.is_file():
            raise ValueError("PhyloNet 3.8.2 scoring JAR is unavailable; supply --sw-jar")
        if digest(self.jar) != JAR_SHA256:
            raise ValueError("PhyloNet scoring JAR does not match the pinned 3.8.2 checksum")
        self.runtime = dict(implementation="PhyloNet.CmpNets.cluster", phylonet_version=VERSION,
                            phylonet_jar=str(self.jar), phylonet_jar_sha256=JAR_SHA256,
                            wrapper_sha256=digest(__file__), rooted=True,
                            normalization=NORMALIZATION)

    def score(self, truth, estimate):
        if not isinstance(truth, str) or not isinstance(estimate, str):
            raise ValueError("PhyloNet SW inputs must be Newick strings")
        self.sequence += 1
        attempt = self.directory / f"{self.sequence:06d}"
        attempt.mkdir(parents=True)
        for name, text in (("truth-original.nwk", truth), ("estimate-original.nwk", estimate)):
            (attempt / name).write_text(text)
        receipt = dict(status="evaluation_failed", rooted=True, timeout_seconds=self.timeout,
                       truth_sha256=hashlib.sha256(truth.encode()).hexdigest(),
                       estimate_sha256=hashlib.sha256(estimate.encode()).hexdigest(),
                       started_utc=time.time())
        started = time.monotonic()
        process = None
        try:
            self._check_runtime()
            normalized, reticulations, tokens = topology_inputs(truth, estimate)
            (attempt / "taxon-map.json").write_text(json.dumps(tokens, indent=2) + "\n")
            for name, text in zip(("truth-normalized.nwk", "estimate-normalized.nwk"), normalized):
                (attempt / name).write_text(text + "\n")
            nexus = ("#NEXUS\nBEGIN NETWORKS;\nNetwork truth = " + normalized[0]
                     + "\nNetwork estimate = " + normalized[1]
                     + "\nEND;\nBEGIN PHYLONET;\nCmpNets truth estimate -m cluster;\nEND;\n")
            inputs = attempt / "input.nex"
            inputs.write_text(nexus)
            temporary = attempt / "tmp"
            temporary.mkdir()
            home = attempt / "home"
            home.mkdir()
            command = [self.java, "-showversion", "-XX:ActiveProcessorCount=1", "-Xmx" + self.heap,
                       "-Djava.io.tmpdir=" + str(temporary), "-Duser.home=" + str(home),
                       "-jar", str(self.jar), str(inputs)]
            env = dict(os.environ, TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
                       XDG_CACHE_HOME=str(home / ".cache"))
            receipt.update(command=command, implementation=self.runtime,
                           normalized_truth_sha256=digest(attempt / "truth-normalized.nwk"),
                           normalized_estimate_sha256=digest(attempt / "estimate-normalized.nwk"))
            (attempt / "command.json").write_text(json.dumps(receipt, indent=2) + "\n")
            with (attempt / "stdout.log").open("wb") as stdout, (attempt / "stderr.log").open("wb") as stderr:
                process = subprocess.Popen(command, cwd=attempt, env=env, stdout=stdout,
                                           stderr=stderr, start_new_session=True)
                receipt["pid"] = process.pid
                try:
                    process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    receipt["timed_out"] = True
                    raise ValueError(f"PhyloNet SW scoring timed out after {self.timeout:g}s")
            receipt["exit_code"] = process.returncode
            if process.returncode != 0:
                raise ValueError(f"PhyloNet scoring exited with status {process.returncode}")
            scores = parse_result((attempt / "stdout.log").read_text())
            receipt.update(status="ok", scores=scores)
            versions = re.findall(r'(?m)^(?:openjdk|java) version .+$',
                                  (attempt / "stderr.log").read_text())
            provenance = dict(self.runtime, java=self.java,
                              java_version=versions[0] if versions else None,
                              receipt=str(attempt / "receipt.json"))
            return dict(scores, true_k=reticulations[0], inferred_k=reticulations[1],
                        metric_parameters_json=json.dumps(provenance, sort_keys=True))
        except (ValueError, OSError) as exc:
            receipt["error"] = str(exc)
            raise ValueError(f"{exc}; see {attempt / 'receipt.json'}") from exc
        finally:
            # Also stop the group if parsing/validation is interrupted.
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            if process is not None:
                receipt["exit_code"] = process.returncode
            receipt.update(wall_seconds=time.monotonic() - started, finished_utc=time.time())
            (attempt / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            # Processes have stopped and the permanent receipt is saved.
            for name in ("tmp", "home"):
                shutil.rmtree(attempt / name, ignore_errors=True)

