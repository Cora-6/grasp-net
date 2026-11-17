"""Real training-frame decoding and one optimizer step per architecture, NOT evaluation."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import GraspNet
from models import build_model,model_loss,metadata
from run import device_of

rows=[]
for rgb in sorted((ROOT/'data/scenes').glob('scene_*/realsense/rgb/*.png')):
    sid=int(rgb.parents[2].name[6:])
    if sid>=100 or sid%10==9: continue
    camera=rgb.parents[1]
    if (camera/'depth'/rgb.name).exists() and (camera/'rect'/(rgb.stem+'.npy')).exists():
        rows.append(dict(sample=str(camera.relative_to(ROOT/'data')/rgb.stem),object_id='scene_%04d'%sid))
        if len(rows)==2:break
if not rows:raise SystemExit('No complete training frames yet')
audit=ROOT/'audit';audit.mkdir(exist_ok=True)
manifest=audit/'smoke_only_split.json';manifest.write_text(json.dumps({'train':rows}))
ds=GraspNet(ROOT/'data',manifest,'train',320)
x,y,rs,rgb,d=ds.sample(0)
device=device_of('auto');torch.set_num_threads(4)
result=dict(purpose='interface smoke test only; not a trained model result',sample=rows[0],
            real_data=True,device=str(device),size=320,grasp_labels=len(rs),
            valid_depth_fraction=float((d>0).mean()),models={})
for name in ('grconvnet','ggcnn2'):
    model=build_model(name,'foreground_balanced').to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.0003)
    xt=torch.from_numpy(x[None]).to(device);yt=torch.from_numpy(y[None]).to(device)
    start=time.time();loss=model_loss(model,model(xt),yt)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    loss.backward();opt.step()
    result['models'][name]=dict(metadata(model),parameters=sum(p.numel() for p in model.parameters()),
                               one_step_loss=float(loss.detach().cpu()),elapsed_seconds=time.time()-start)
fig,ax=plt.subplots(1,3,figsize=(12,4))
ax[0].imshow(rgb);ax[0].set_title('Real RGB frame')
ax[1].imshow(np.ma.masked_where(d<=0,d),cmap='viridis');ax[1].set_title('Raw depth (metres)')
ax[2].imshow(rgb);ax[2].set_title('Provided labels (up to 50)')
for r in rs[::max(1,len(rs)//50)][:50]:
    pts=np.vstack([r,r[0]]);ax[2].plot(pts[:,0],pts[:,1],color='lime',lw=.6)
for a in ax:a.axis('off')
fig.tight_layout();fig.savefig(audit/'real_sample.png',dpi=160);plt.close(fig)
(audit/'real_smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
