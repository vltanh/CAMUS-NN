"""Optional upstream scorers: metric semantics, provenance, and process cleanup."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from camus_nn.evaluate import main
from camus_nn.phylonet_sw import PhyloNetSW, parse_result
from camus_nn.phylonetworks_hwcd import PhyloNetworksHWCD

NETWORK = "(O,(((D,C),(A,B)#H1),(#H1,E)));"
REROOTED = "(E,(#H1,(O,((D,C),(A,B)#H1))));"
JAR = os.environ.get("CAMUS_NN_PHYLONET_JAR")
JULIA = os.environ.get("CAMUS_NN_JULIA")
JULIA_PROJECT = os.environ.get("CAMUS_NN_JULIA_PROJECT")
JULIA_DEPOT = os.environ.get("CAMUS_NN_JULIA_DEPOT")


class ConfigurationTests(unittest.TestCase):
    def test_ambiguous_or_inapplicable_options_are_rejected(self):
        cases = [
            (["--metric", "phylonet-sw"], "require --output"),
            (["--metric", "phylonetworks-hwcd"], "require --output"),
            (["--metric", "phylonet-sw", "--output", "RESULT", "--outgroup", "O"], "--outgroup applies"),
            (["--metric", "phylonetworks-hwcd", "--output", "RESULT", "--max-switchings", "1"], "--max-switchings applies"),
            (["--metric", "both", "--sw-heap", "2g"], "--sw-* options require"),
            (["--metric", "phylonet-sw", "--output", "RESULT", "--hwcd-timeout", "1"], "--hwcd-* options require"),
        ]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            network = root / "network.nwk"
            network.write_text(NETWORK)
            for flags, expected in cases:
                flags = [str(root / "unused.json") if value == "RESULT" else value for value in flags]
                error = StringIO()
                with self.subTest(flags=flags), redirect_stderr(error), self.assertRaises(SystemExit):
                    main(["--truth", str(network), "--estimate", str(network), *flags])
                self.assertIn(expected, error.getvalue())
                self.assertFalse((root / "unused.json").exists())

    def test_default_python_scoring_does_not_start_external_processes(self):
        with tempfile.TemporaryDirectory() as name:
            network = Path(name) / "network.nwk"
            network.write_text("(OUT,((A,B),C));")
            output = StringIO()
            with patch("subprocess.Popen", side_effect=AssertionError("Unexpected external runtime")), redirect_stdout(output):
                main(["--truth", str(network), "--estimate", str(network)])
            record = json.loads(output.getvalue())
            self.assertEqual(record["scores"]["softwired_error"], 0)
            self.assertEqual(record["scores"]["hardwired_distance"], 0)
            self.assertEqual(record["outgroup"], "OUT")

    def test_phylonet_output_requires_one_consistent_finite_result(self):
        line = "The cluster-based distance between two networks: 0.25 0.0 0.125\n"
        self.assertEqual(parse_result(line), dict(softwired_fn=.25, softwired_fp=0, softwired_error=.125))
        for text in ("", line + line, line.replace("0.25", "NaN"),
                     line.replace("0.25", "1.25"), line.replace("0.125", "0.5")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_result(text)

    def test_wrong_jar_is_rejected_with_a_failure_receipt(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            jar = root / "wrong.jar"
            jar.write_bytes(b"wrong implementation")
            with PhyloNetSW(jar=jar, workdir=root / "scores") as scorer:
                with self.assertRaisesRegex(ValueError, "checksum"):
                    scorer.score(NETWORK, NETWORK)
            receipt = json.loads((scorer.directory / "000001/receipt.json").read_text())
            self.assertEqual(receipt["status"], "evaluation_failed")

    def test_java_timeout_stops_children_and_cleans_owned_temporaries(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            jar = root / "fixture.jar"
            jar.write_bytes(b"test fixture")
            java = root / "fake-java"
            java.write_text("#!/usr/bin/env python3\nimport subprocess,sys,time\n"
                            "from pathlib import Path\n"
                            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
                            "Path('child.pid').write_text(str(p.pid))\ntime.sleep(60)\n")
            java.chmod(0o755)
            with patch("camus_nn.phylonet_sw.JAR_SHA256", hashlib.sha256(jar.read_bytes()).hexdigest()):
                with PhyloNetSW(jar=jar, java=java, timeout=.3, workdir=root / "scores") as scorer:
                    with self.assertRaisesRegex(ValueError, "timed out"):
                        scorer.score(NETWORK, NETWORK)
            attempt = scorer.directory / "000001"
            receipt = json.loads((attempt / "receipt.json").read_text())
            self.assertTrue(receipt["timed_out"])
            self.assertLess(receipt["exit_code"], 0)
            self.assertFalse((attempt / "tmp").exists())
            self.assertFalse((attempt / "home").exists())
            for pid in (receipt["pid"], int((attempt / "child.pid").read_text())):
                stat = Path(f"/proc/{pid}/stat")
                self.assertTrue(not stat.exists() or stat.read_text().split()[2] == "Z")

    def test_julia_timeout_stops_worker_and_retains_receipts(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            julia = root / "fake-julia"
            julia.write_text("#!/usr/bin/env python3\nimport sys,time\n"
                             "print('READY\\t1.4.0\\t1.11.6\\t'+'0'*64,flush=True)\n"
                             "for line in sys.stdin: time.sleep(60)\n")
            julia.chmod(0o755)
            with PhyloNetworksHWCD(julia=julia, workdir=root / "scores", timeout=.05, startup_timeout=10) as scorer:
                with self.assertRaisesRegex(ValueError, "timed out"):
                    scorer.score("(A,B,C);", "(A,B,C);")
                self.assertIsNone(scorer.process)
            self.assertFalse((scorer.directory / "tmp").exists())
            receipt = json.loads((scorer.directory / "000001/receipt.json").read_text())
            self.assertEqual(receipt["status"], "evaluation_failed")
            self.assertTrue((scorer.directory / "julia.log").is_file())


@unittest.skipUnless(JAR and Path(JAR).is_file() and shutil.which("java"),
                     "Set CAMUS_NN_PHYLONET_JAR for upstream PhyloNet integration tests")
class PhyloNetIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.scorer = PhyloNetSW(jar=JAR, workdir=self.root / "scores")

    def tearDown(self):
        self.temporary.cleanup()

    def test_root_dependence_and_upstream_provenance(self):
        self.assertEqual(self.scorer.score(NETWORK, NETWORK)["softwired_error"], 0)
        result = self.scorer.score(NETWORK, REROOTED)
        self.assertEqual((result["softwired_fn"], result["softwired_fp"], result["softwired_error"]), (.4, .4, .4))
        parameters = json.loads(result["metric_parameters_json"])
        self.assertTrue(parameters["rooted"])
        self.assertEqual(parameters["phylonet_version"], "3.8.2")
        attempt = Path(parameters["receipt"]).parent
        self.assertIn("CmpNets truth estimate -m cluster;", (attempt / "input.nex").read_text())
        self.assertFalse((attempt / "tmp").exists())

    def test_truth_order_annotations_and_safe_taxon_tokens(self):
        truth = "(OUT,((a,(b)#H1),(#H1,(c,d))));"
        tree = "(OUT,((a,b),(c,d)));"
        a = self.scorer.score(truth, tree)
        b = self.scorer.score(tree, truth)
        self.assertEqual((a["softwired_fn"], a["softwired_fp"], a["softwired_error"]), (.25, 0, .125))
        self.assertEqual((b["softwired_fn"], b["softwired_fp"]), (0, .25))
        weighted = "('OUT β':9e-20,((A:1,(B:2)#H1:1e-15::0.7),(C:3,#H1:2::0.3)));"
        plain = "('OUT β',((A,(B)#H1),(C,#H1)));"
        self.assertEqual(self.scorer.score(weighted, plain)["softwired_error"], 0)

    def test_cli_version_roots_and_input_preservation(self):
        truth, estimate, output = self.root / "truth.nwk", self.root / "estimate.nwk", self.root / "scores.json"
        truth.write_text(NETWORK)
        estimate.write_text(REROOTED)
        before = (truth.read_bytes(), estimate.read_bytes())
        with redirect_stdout(StringIO()):
            main(["--truth", str(truth), "--estimate", str(estimate), "--metric", "phylonet-sw",
                  "--sw-jar", JAR, "--output", str(output)])
        record = json.loads(output.read_text())
        self.assertEqual(record["metric_version"], "phylonet-sw/v1-rooted")
        self.assertEqual(record["scores"]["softwired_error"], .4)
        self.assertIsNone(record["outgroup"])
        self.assertIsNone(record["max_switchings"])
        self.assertEqual(before, (truth.read_bytes(), estimate.read_bytes()))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(["--truth", str(truth), "--estimate", str(estimate), "--metric", "phylonet-sw",
                  "--sw-jar", JAR, "--output", str(output)])


@unittest.skipUnless(JULIA and shutil.which(JULIA),
                     "Set CAMUS_NN_JULIA and its environment for upstream PhyloNetworks tests")
class PhyloNetworksIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.options = dict(julia=JULIA, julia_project=JULIA_PROJECT, depot=JULIA_DEPOT)
        cls.scorer = PhyloNetworksHWCD(workdir=cls.root / "scores", **cls.options)

    @classmethod
    def tearDownClass(cls):
        directory = cls.scorer.directory
        cls.scorer.close()
        assert not (directory / "tmp").exists()
        cls.temporary.cleanup()

    def test_semidirected_and_rooted_conventions(self):
        self.assertEqual(self.scorer.score(NETWORK, REROOTED)["hardwired_distance"], 0)
        result = self.scorer.score(NETWORK, REROOTED, rooted=True)
        self.assertEqual(result["hardwired_distance"], 4)
        parameters = json.loads(result["metric_parameters_json"])
        self.assertEqual(parameters["phylonetworks_version"], "1.4.0")
        self.assertIs(parameters["rooted"], True)
        self.assertEqual(len(parameters["phylonetworks_source_sha256"]), 64)

    def test_bad_pair_recovers_and_hybrid_information_is_retained(self):
        self.scorer.score(NETWORK, NETWORK)
        worker = self.scorer.process
        with self.assertRaisesRegex(ValueError, "taxa differ"):
            self.scorer.score("(A,B,C);", "(A,B,D);")
        self.assertEqual(self.scorer.score(NETWORK, NETWORK)["hardwired_distance"], 0)
        self.assertIs(self.scorer.process, worker)
        tree = "(O,((D,C),((A,B),E)));"
        result = self.scorer.score(NETWORK, tree)
        self.assertGreater(result["hardwired_distance"], 0)
        self.assertEqual((result["true_k"], result["inferred_k"]), (1, 0))

    def test_cli_semidirected_metadata(self):
        truth, estimate, output = self.root / "truth.nwk", self.root / "estimate.nwk", self.root / "cli.json"
        truth.write_text(NETWORK)
        estimate.write_text(REROOTED)
        flags = ["--hwcd-julia", JULIA]
        if JULIA_PROJECT:
            flags += ["--hwcd-project", JULIA_PROJECT]
        if JULIA_DEPOT:
            flags += ["--hwcd-depot", JULIA_DEPOT]
        with redirect_stdout(StringIO()):
            main(["--truth", str(truth), "--estimate", str(estimate), "--metric", "phylonetworks-hwcd",
                  "--output", str(output), *flags])
        record = json.loads(output.read_text())
        self.assertEqual(record["metric_version"], "phylonetworks-hwcd/v1-semidirected")
        self.assertEqual(record["scores"]["hardwired_distance"], 0)
        self.assertFalse(record["metric_parameters"]["rooted"])
        self.assertNotIn("softwired_error", record["scores"])
        self.assertNotIn("hardwired_error", record["scores"])


if __name__ == "__main__":
    unittest.main()
