"""Wait for downloads, audit full selected-frame coverage, then train and evaluate."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
state_path=ROOT/'audit/pipeline_status.json'


def status(stage,**extra):
    state_path.write_text(json.dumps(dict(stage=stage,updated=time.strftime('%Y-%m-%d %H:%M:%S'),**extra),indent=2))
    print(stage,extra,flush=True)


def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],check=True,cwd=ROOT)


try:
    status('waiting_for_downloads')
    while True:
        states=[]
        for archive in ('scenes.tar','rect_labels.tar'):
            p=ROOT/'audit'/(archive+'.download.json')
            try:states.append(json.loads(p.read_text()))
            except (FileNotFoundError,json.JSONDecodeError):states.append({})
        if all(s.get('complete') for s in states):break
        # A stopped downloader needs manual resume; no silent infinite wait.
        stamps=[(ROOT/'audit'/(a+'.download.json')).stat().st_mtime for a in ('scenes.tar','rect_labels.tar')
                if (ROOT/'audit'/(a+'.download.json')).exists()]
        if stamps and time.time()-max(stamps)>600:
            raise RuntimeError('Download progress has stopped for 10 minutes; inspect download logs and resume')
        time.sleep(30)
    status('checking_data')
    split=ROOT/'split.json'
    if not split.exists():run('prepare_split.py','--data',str(ROOT/'data'),'--out',str(split),'--stride','16')
    run('audit_data.py')
    run('smoke_real.py')
    status('training_then_evaluating',models=['grconvnet','ggcnn2'])
    run('run_experiments.py')
    status('complete')
except Exception as e:
    status('failed',error=str(e))
    raise
