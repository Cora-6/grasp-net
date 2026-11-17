"""Summarize actual evaluation files; never fill missing results with estimates."""
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
rows=[]
for path in sorted((ROOT/'runs').glob('*/*/test_*/metrics.json')):
    if not (path.parent/'complete.json').exists(): continue
    model=path.parents[2].name
    variant,seed=path.parents[1].name.rsplit('_s',1)
    for item in json.loads(path.read_text()):
        rows.append(dict(model=model,variant=variant,seed=int(seed),test_group=path.parent.name,**item))
if not rows: raise SystemExit('No completed evaluations. No results report generated.')
report=ROOT/'results';report.mkdir(exist_ok=True)
with (report/'per_seed.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
groups={}
for r in rows:
    key=tuple(r[k] for k in ('model','variant','test_group','condition','severity'))
    groups.setdefault(key,[]).append(r)
summary=[]
for key,rr in sorted(groups.items()):
    summary.append(dict(zip(('model','variant','test_group','condition','severity'),key),
                        seeds=len(rr),mean_accuracy_percent=100*float(np.mean([r['accuracy'] for r in rr])),
                        std_accuracy_percent=100*float(np.std([r['accuracy'] for r in rr]))))
baselines={(r['model'],r['test_group'],r['condition'],r['severity']):r['mean_accuracy_percent'] for r in summary if r['variant']=='baseline'}
for r in summary:
    baseline=baselines.get((r['model'],r['test_group'],r['condition'],r['severity']))
    r['delta_vs_baseline_pp']=None if baseline is None else r['mean_accuracy_percent']-baseline
with (report/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(report/'README.md').write_text('# Results\n\nActual completed runs only. See summary.csv and per_seed.csv. '
    'Accuracy is the project planar IoU/angle metric, not official GraspNet 3D AP. '
    'Seed standard deviation is not a confidence interval. Keep the three test groups and each corruption separate.\n')
print('Saved',len(rows),'evaluation rows')
