"""Paired, separate augmentation experiments; stop on errors, resume completed jobs."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=['baseline','mild_missing','mild_gaussian','mild_irregular','mild_smooth','mild_edge','mild_occlusion','dropout']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='auto')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--warmup-epochs',type=int,default=10)
    p.add_argument('--size',type=int,default=320)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument("--train-only",action="store_true")
    a=p.parse_args()
    split=ROOT/'split.json'
    if not split.exists(): raise FileNotFoundError('Prepare and audit the complete selected-frame split first')
    out=ROOT/'runs'; out.mkdir(exist_ok=True)
    source={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in ROOT.rglob('*.py') if (f.parent == ROOT or 'models' in f.relative_to(ROOT).parts or 'third_party' in f.relative_to(ROOT).parts) and '__pycache__' not in f.parts}
    protocol=dict(vars(a),models=['grconvnet','ggcnn2'],variants=VARIANTS,
                  checkpoint='last.pt after equal fine-tuning epochs',
                  split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),source_sha256=source)
    protocol.pop("train_only",None)
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Existing protocol differs; do not mix experiments')
    path.write_text(json.dumps(protocol,indent=2))
    common=['--data',str(ROOT/'data'),'--split',str(split),'--device',a.device,'--size',str(a.size),
            '--batch',str(a.batch),'--loss-profile','foreground_balanced','--lr','0.0003']
    def call(args, directory):
        directory.mkdir(parents=True,exist_ok=True)
        with (ROOT/'audit'/'experiments.log').open('a') as log:
            log.write('\n'+' '.join(args)+'\n');log.flush()
            subprocess.run([sys.executable,str(ROOT/'run.py')]+args,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    def train(model,seed,variant,dest,epochs,initial=None):
        marker=dest/'complete.json'
        if marker.exists():
            if not (dest/'last.pt').exists(): raise RuntimeError('Completion marker without checkpoint')
            return
        resume = dest.exists() and any(dest.iterdir())
        if resume and not (dest/'last.pt').exists():
            raise RuntimeError('No epoch checkpoint for incomplete run: '+str(dest))
        args=['train']+common+['--out',str(dest),'--model',model,'--seed',str(seed),'--variant',variant,'--epochs',str(epochs)]
        if resume: args+=['--resume']
        if initial: args+=['--init-checkpoint',str(initial)]
        call(args,dest)
        marker.write_text(json.dumps({'epochs':epochs,'last_sha256':hashlib.sha256((dest/'last.pt').read_bytes()).hexdigest()}))
    for model in protocol['models']:
        for seed in a.seeds:
            warm=out/model/('warmup_s%d'%seed)
            train(model,seed,'baseline',warm,a.warmup_epochs)
            history=json.loads((warm/'history.json').read_text())
            if max(r['val_accuracy'] for r in history) <= 0:
                raise RuntimeError('Warmup validation accuracy remained zero; diagnose before augmentation experiments: '+model)
            for variant in VARIANTS:
                dest=out/model/('%s_s%d'%(variant,seed))
                train(model,seed,variant,dest,a.epochs,warm/'best.pt')
    if a.train_only:
        print('Training completed; full test-label audit required before evaluation',flush=True)
        return
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_data.py')],check=True,cwd=ROOT)
    # All model/augmentation training finishes before opening final test results.
    for model in protocol['models']:
        for seed in a.seeds:
            for variant in VARIANTS:
                dest=out/model/('%s_s%d'%(variant,seed))
                for part in ('test_seen','test_similar','test_novel'):
                    ev=dest/part
                    if (ev/'complete.json').exists(): continue
                    call(['evaluate']+common+['--out',str(ev),'--checkpoint',str(dest/'last.pt'),
                          '--part',part,'--test-suite','extended'],ev)
                    (ev/'complete.json').write_text(json.dumps({'complete':True}))
    subprocess.run([sys.executable,str(ROOT/'scripts/report_results.py')],check=True,cwd=ROOT)
    print('All experiments complete',flush=True)


if __name__=='__main__': main()
