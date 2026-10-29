#!/usr/bin/env python3
"""Repeat the documented commands using the installed package, in fresh outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path,help='Fresh report directory outside the checkout')
    parser.add_argument('--bin-dir',type=Path,help='Optional explicit native-program directory')
    parser.add_argument('--training',action='store_true',help='Also repeat example training and compare CSV/direct input')
    args=parser.parse_args();output=args.out.resolve()
    if output.is_relative_to(ROOT):parser.error('Keep generated reports outside the source checkout')
    output.mkdir(parents=True,exist_ok=False);(output/'tmp').mkdir();(output/'cache').mkdir()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(output/'tmp'),TMP=str(output/'tmp'),
        TEMP=str(output/'tmp'),XDG_CACHE_HOME=str(output/'cache'),OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    # A source-tree PYTHONPATH would hide an incomplete installation.
    env.pop('PYTHONPATH',None)
    def run(command,log):
        with (output/log).open('w') as stream:
            subprocess.run(command,cwd=output,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    probe=('import json,platform,camus_nn,numpy,torch,networkx; '
        'from camus_nn.infer import DEFAULT_BIN_DIR,DEFAULT_MODEL; '
        'print(json.dumps(dict(package=camus_nn.__file__,python=platform.python_version(),'
        'numpy=numpy.__version__,torch=torch.__version__,networkx=networkx.__version__,'
        'bin_dir=str(DEFAULT_BIN_DIR),model=str(DEFAULT_MODEL))))')
    installed=json.loads(subprocess.check_output([sys.executable,'-B','-c',probe],cwd=output,env=env,text=True))
    if Path(installed['package']).resolve().is_relative_to(ROOT):
        raise ValueError('Run this check against an installed wheel, rather than an editable source checkout')
    assert sha(installed['model'])==sha(ROOT/'models/all-frequency.json'),'Bundled model differs from checkout'
    report=dict(schema='camus-nn-reproducibility/v1',installed=installed,checks=[],
        scope='Exact repeatability on this host; not a guarantee across all hardware/library versions',
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')),'unknown'),
        inputs={},model_sha256=sha(installed['model']))
    def checkpoint():
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    datasets=[]
    for name in ('validation','test'):
        folder=ROOT/'examples'/name
        datasets.append((name,folder/'genes.nwk',folder/'guide.nwk','outgroup',('orig','lqf','lqw','iter')))
    renamed=output/'supplied-inputs';renamed.mkdir()
    for name in ('genes.nwk','guide.nwk'):
        (renamed/name).write_text(re.sub(r'\bOUT\b','REFERENCE',(ROOT/'examples/test'/name).read_text()))
    datasets.append(('supplied',renamed/'genes.nwk',renamed/'guide.nwk','supplied',('lqw','iter')))
    for dataset,genes,guide,rooting,methods in datasets:
        report['inputs'][dataset]=dict(genes=sha(genes),guide=sha(guide),guide_rooting=rooting)
        for method in methods:
            runs=[]
            for repeat in (1,2):
                destination=output/f'{dataset}-{method}-{repeat}'
                command=[sys.executable,'-B','-m','camus_nn.infer','--method',method if method!='iter' else 'lqw',
                    '--genes',str(genes),'--guide',str(guide),'--guide-rooting',rooting,
                    '--sampler','all','--threads','1','--case','reproducibility/'+dataset,
                    '--out',str(destination)]
                if method!='orig':command+=['--model',str(ROOT/'models/all-frequency.json')]
                if method=='iter':command+=['--iterate','--inner-solver','per-k-stopping','--max-guide-runs','5']
                if args.bin_dir:command+=['--bin-dir',str(args.bin_dir.resolve())]
                run(command,destination.name+'.log');runs.append(destination)
            left,right=runs
            for name in ('network.nwk','network.csv'):
                assert (left/name).read_bytes()==(right/name).read_bytes(),(dataset,method,name)
            a,b=[json.loads((d/'prediction.json').read_text()) for d in runs]
            for key in ('network','k','objective','method','sampler','search_cap','guide_rooting','outgroup_anchored'):
                assert a[key]==b[key],(dataset,method,key)
            if method=='iter':
                iterations=[json.loads((d/'iteration.json').read_text()) for d in runs]
                signature=lambda it:dict(winner=it['winner'],trials=[{k:t[k] for k in
                    ('guide','network','k','guide_score','added_gain','total_score')} for t in it['trials']])
                assert signature(iterations[0])==signature(iterations[1]),(dataset,method,'trial sequence')
            assert all(not (d/'tmp').exists() for d in runs),'Intermediate-table cleanup failed'
            report['checks'].append(dict(dataset=dataset,method=method,exact_curve=True,
                exact_network=True,temporary_tables_cleaned=True,network_sha256=sha(left/'network.nwk'),
                curve_sha256=sha(left/'network.csv')))
            checkpoint();print(dataset,method,'exact repeat',flush=True)
    if args.training:
        table=output/'training.csv';common=[sys.executable,'-B','-m']
        extra=['--bin-dir',str(args.bin_dir.resolve())] if args.bin_dir else []
        cases=ROOT/'examples/training/cases.json'
        run([*common,'camus_nn.prepare','--training-cases',str(cases),'--output-table',str(table),*extra],'prepare.log')
        models=[]
        for name,mode,source in [('fit-1','quartet-table',table),('fit-2','quartet-table',table),
                                  ('fit-direct','training-cases',cases)]:
            model=output/(name+'.json');models.append(model)
            command=[*common,'camus_nn.train','--'+mode,str(source),'--output-model',str(model),'--seed','42']
            if mode=='training-cases':command+=extra
            run(command,name+'.log')
        assert len({sha(p) for p in models})==1,'Example fits differ'
        report['training']=dict(example_only=True,exact_seeded_repeats=True,exact_csv_vs_direct=True,
                                model_sha256=sha(models[0]),recreates_bundled_model=False)
    report['passed']=True;checkpoint();print(output/'report.json',flush=True)


if __name__=='__main__':main()
