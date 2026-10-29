"""Keep the ASTRAL-guide and optional known-root choices explicit."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('brooqs_runner',ROOT/'misc/brooqs.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BrooqsCommandTests(unittest.TestCase):
    def test_blob_search_uses_supplied_astral_and_root_is_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            args=argparse.Namespace(tob_qmc=Path('tree-qmc'),alpha=1e-7,beta=.95,
                polytomy_seed=12345,maxcut_seed=1,starting_tree=Path('astral.nwk'),
                gene_trees=Path('genes.nwk'),tob_search='exhaustive',outgroup=None)
            for outgroup in (None,'OUT'):
                args.outgroup=outgroup
                with patch.object(runner.subprocess,'run') as run:
                    runner.estimate_tree_of_blobs(args,output)
                command=run.call_args.args[0]
                self.assertEqual(command[command.index('--blobsearchonly')+1],'astral.nwk')
                self.assertEqual('--root' in command,outgroup is not None)
                self.assertEqual(command[command.index('--iter_limit_blob')+1],'0')

    def test_brooqs_root_is_not_imposed_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);(output/'network.nwk').write_text('((A,B),(C,D));\n')
            args=argparse.Namespace(brooqs_script=Path('count_quartets.py'),
                gene_trees=Path('genes.nwk'),seed=1142,rooted=False)
            for rooted in (False,True):
                args.rooted=rooted
                with patch.object(runner.subprocess,'run') as run:
                    runner.resolve_tree_of_blobs(args,Path('tob.nwk'),output)
                command=run.call_args.args[0]
                self.assertEqual('--rooted' in command,rooted)
                self.assertEqual(command[command.index('-t')+1],'tob.nwk')


if __name__=='__main__':unittest.main()
