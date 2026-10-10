"""Collect immutable completed jobs; optionally release the dedicated VM at end.

Run from the repository root with the training environment. Observation failures
never cause a training restart or VM release. Checkpoints remain in W&B; local
archives include final adapters, manifests, metrics, and raw evaluations.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

GROUP = 'controlled-l4-fla-v1'
JOBS = [f'{reward}-{seed}' for seed in (42,43,44) for reward in ('binary','shaped')]
PREFIX = 'COUNTDOWN_JSON '


def remote(session, code):
    result = subprocess.run(['colab','exec','-s',session,'--timeout','30'],input=code,
                            capture_output=True,text=True,timeout=55)
    if result.returncode:
        raise RuntimeError(f'Colab observation/action exited {result.returncode}; inspect the same session')
    lines = [line[len(PREFIX):] for line in result.stdout.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise RuntimeError('Colab did not return exactly one structured result')
    return json.loads(lines[0])


def inspect_remote(session):
    return remote(session, f'''
import json
from pathlib import Path
metadata=json.loads(Path('/content/study-process.json').read_text())
if metadata['group'] != {GROUP!r}:
    raise RuntimeError('Session now belongs to a different study')
root=Path(metadata['root'])/'outputs'/metadata['group']
jobs={{}}
for name in {JOBS!r}:
    path=root/name/'status.json'
    jobs[name]=json.loads(path.read_text()) if path.exists() else {{'state':'pending'}}
stat_path=Path('/proc')/str(metadata['pid'])/'stat'
process_state=stat_path.read_text().split(') ',1)[1].split()[0] if stat_path.exists() else None
print({PREFIX!r}+json.dumps({{'pid':metadata['pid'],'controller_running':process_state not in (None,'Z'),'jobs':jobs}}))
''')


def verify_job_files(path):
    required = ['status.json','evaluation.json','training/config.json','training/metrics.jsonl',
                'training/train-puzzles.json','training/holdout.json','training/wandb-run.json',
                'training/adapter/adapter_config.json','training/adapter/adapter_model.safetensors']
    if any(not (path/name).is_file() for name in required):
        raise ValueError('Completed-job archive lacks required evidence or final adapter')
    if json.loads((path/'status.json').read_text())['state'] != 'complete':
        raise ValueError('Job is not complete')
    metrics = [json.loads(line) for line in (path/'training/metrics.jsonl').read_text().splitlines()]
    if ([row['step'] for row in metrics if 'reward' in row] != list(range(1,301))
            or 'train_runtime' not in metrics[-1]):
        raise ValueError('Training did not finish all 300 steps')


def install_archive(archive, checksum, destination):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != checksum:
        raise ValueError('Downloaded archive checksum differs')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.collect-',dir=destination.parent) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                parts = Path(member.name).parts
                if (not parts or parts[0] != destination.name or '..' in parts
                        or Path(member.name).is_absolute() or not (member.isfile() or member.isdir())):
                    raise ValueError('Unexpected archive member')
            handle.extractall(staging,filter='data')
        candidate = staging/destination.name
        verify_job_files(candidate)
        (candidate/'collection.json').write_text(json.dumps({'sha256':checksum},indent=2)+'\n')
        candidate.rename(destination)


def collect_job(session, name, root, archives):
    destination = root/name
    if not destination.exists():
        info = remote(session, f'''
import hashlib,json,tarfile
from pathlib import Path
metadata=json.loads(Path('/content/study-process.json').read_text())
if metadata['group'] != {GROUP!r}:
    raise RuntimeError('Wrong study')
source=Path(metadata['root'])/'outputs'/metadata['group']/{name!r}
if json.loads((source/'status.json').read_text())['state'] != 'complete':
    raise RuntimeError('Refusing to archive an unfinished job')
archive=Path('/content')/({GROUP!r}+'-'+{name!r}+'.tar.gz')
def include(member):
    return None if any(p.startswith('checkpoint-') for p in Path(member.name).parts) else member
with tarfile.open(archive,'w:gz') as handle:
    handle.add(source,arcname={name!r},filter=include)
print({PREFIX!r}+json.dumps({{'path':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}}))
''')
        archive = archives/f'{name}.tar.gz'
        partial = archives/f'{name}-{time.time_ns()}.partial'
        try:
            subprocess.run(['colab','download','-s',session,info['path'],str(partial)],check=True,timeout=300)
            install_archive(partial,info['sha256'],destination)
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
        (destination/'collection.json').write_text(json.dumps(info,indent=2)+'\n')
    verify_job_files(destination)
    if not (destination/'collection.json').is_file():
        raise ValueError('Existing local job lacks a verified collection receipt; inspect before proceeding')
    provenance = destination/'training/provenance.json'
    if not provenance.exists():
        import wandb
        reference = json.loads((destination/'training/wandb-run.json').read_text())
        run = wandb.Api(timeout=30).run(f"{reference['entity']}/{reference['project']}/{reference['id']}")
        if run.state != 'finished' or run.group != GROUP:
            raise ValueError('W&B training run is not finished in the expected group')
        config = json.loads((destination/'training/config.json').read_text())
        if any(run.config.get(key) != value for key,value in config.items()):
            raise ValueError('W&B config differs from the training manifest')
        with tempfile.TemporaryDirectory() as temporary:
            file = run.file('wandb-metadata.json').download(root=temporary,replace=True)
            metadata = json.loads(Path(file.name).read_text())
            requirements_file = run.file('requirements.txt').download(root=temporary,replace=True)
            requirements = Path(requirements_file.name).read_text()
        record = {'training_url':run.url,'state':run.state,'group':run.group,
                  'captured_at':datetime.now(timezone.utc).isoformat(),
                  'config':{key:run.config.get(key) for key in
                            ('git_commit','git_dirty','packages','optional_kernel_packages','python_version')},
                  'hardware':{key:metadata.get(key) for key in ('gpu','gpu_count','cpu_count','cudaVersion','python','os')}}
        record['config']['requirements_sha256'] = hashlib.sha256('\n'.join(sorted(requirements.splitlines())).encode()).hexdigest()
        (destination/'training/requirements.txt').write_text(requirements)
        provenance.write_text(json.dumps(record,indent=2)+'\n')
    print(f'Collected and checked {name}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',default='countdown-study')
    parser.add_argument('--interval',type=int,default=300)
    parser.add_argument('--once',action='store_true',help='Observe/collect once without running final analysis')
    parser.add_argument('--release-when-complete',action='store_true')
    args = parser.parse_args()
    if args.interval < 30:
        parser.error('--interval must be at least 30 seconds')
    root = Path('outputs')/GROUP
    archives = Path('outputs/colab-archives')/GROUP
    archives.mkdir(parents=True,exist_ok=True)
    status = archives/'status.json'
    while True:
        try:
            snapshot = inspect_remote(args.session)
            for name in JOBS:
                if snapshot['jobs'][name]['state'] == 'complete':
                    collect_job(args.session,name,root,archives)
            status.write_text(json.dumps({'state':'waiting','snapshot':snapshot},indent=2)+'\n')
            if args.once:
                print(json.dumps(snapshot),flush=True)
                return
            if all(snapshot['jobs'][name]['state'] == 'complete' for name in JOBS) and not snapshot['controller_running']:
                # Run the scientific gates against the collected data before release.
                from analyze_controlled_study import checked_run
                for name in JOBS:
                    reward,seed = name.split('-')
                    checked_run(root/name,0.0 if reward == 'binary' else 0.05,int(seed))
                confirmation = inspect_remote(args.session)
                if confirmation != snapshot:
                    raise RuntimeError('Remote state changed during final collection; recheck before release')
                if args.release_when_complete:
                    subprocess.run(['colab','stop','-s',args.session],check=True,timeout=55)
                status.write_text(json.dumps({'state':'collected','gpu_released':args.release_when_complete},indent=2)+'\n')
                break
            if not snapshot['controller_running'] and any(snapshot['jobs'][n]['state']=='failed' for n in JOBS):
                status.write_text(json.dumps({'state':'training_failed','snapshot':snapshot},indent=2)+'\n')
                raise SystemExit('Training queue failed; completed jobs collected, session retained for diagnosis')
        except Exception as error:
            status.write_text(json.dumps({'state':'observation_or_collection_error','error':str(error)},indent=2)+'\n')
            print(f'Collection deferred: {error}',flush=True)
            if args.once:
                raise
        time.sleep(args.interval)
    code = subprocess.run([sys.executable,'scripts/analyze_controlled_study.py',str(root),'--wandb']).returncode
    status.write_text(json.dumps({'state':'complete' if code == 0 else 'analysis_failed',
                                  'gpu_released':args.release_when_complete,'analysis_exit_code':code},indent=2)+'\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
