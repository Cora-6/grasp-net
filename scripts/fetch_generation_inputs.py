"""Fetch official 6D labels and scene metadata for missing planar test labels.

Only supplied annotation inputs are downloaded here; no model predictions are
used to create test truth. Generation is a separate audited step.
"""
import hashlib,json,time
from pathlib import Path
from zipfile import ZipFile
from fetch_sample import RemoteZip

ROOT=Path(__file__).resolve().parents[1]
SCENES=set(range(120,190))|{17}  # scene 17 checks agreement with supplied labels.

class CachedZip(RemoteZip):
    def __init__(self,fid):
        super().__init__(fid);self.cache=b'';self.start=-1
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n<=0:return b''
        if not (self.start<=self.pos and self.pos+n<=self.start+len(self.cache)):
            self.start=self.pos
            self.cache=self.fetch(self.pos,min(self.size-1,self.pos+max(n,65536)-1))
        start=self.pos-self.start;self.pos+=n
        return self.cache[start:start+n]

def destination(name,kind):
    parts=Path(name).parts
    if '..' in parts:return None
    if kind=='grasp' and name.endswith('_labels.npz'):
        return Path('grasp_label')/parts[-1]
    scene=next((p for p in parts if p.startswith('scene_') and p[6:].isdigit()),None)
    if not scene or int(scene[6:]) not in SCENES:return None
    if kind=='collision' and name.endswith('.npz'):
        return Path('collision_label')/scene/parts[-1]
    tail=parts[parts.index(scene)+1:]
    if tail==('object_id_list.txt',):return Path('scenes')/scene/tail[0]
    if not tail or tail[0]!='realsense':return None
    if len(tail)==2 and tail[1] in ('camera_poses.npy','cam0_wrt_table.npy','camK.npy'):
        return Path('scenes')/scene/Path(*tail)
    if len(tail)==3 and tail[1]=='annotations' and Path(tail[2]).suffix=='.xml' and int(Path(tail[2]).stem)%16==0:
        return Path('scenes')/scene/Path(*tail)
    return None

sources=[('grasp','1FCV6j2J2eQpVk_ddJXljJvjRT1KU3sJ6'),
         ('collision','1p43sntiN9HJZRDFDNpzaEaEYoPY6IWsu'),
         ('metadata','1_nxiCmHhtsjCgA1IKJn3AuMq4SH_fseW'),
         ('metadata','1njgthC-uUvTXgG99qq1fjS-fzofttFms'),
         ('metadata','1xixvgY0yK7TEALq3k7JcJk2_SP_6r8nk'),
         ('metadata','1wQx8IJ_Lok3hVK_nchQUzgw88QBGZ5iq')]

def main():
    audit=ROOT/'audit';state_path=audit/'generation_inputs_status.json'
    for kind,fid in sources:
        last=None
        for attempt in range(3):
            try:
                with ZipFile(CachedZip(fid)) as archive:
                    selected=[(m,destination(m.filename,kind)) for m in archive.infolist() if not m.is_dir()]
                    selected=[(m,p) for m,p in selected if p is not None]
                    if not selected:raise RuntimeError('No expected inputs in official archive '+fid)
                    for index,(member,relative) in enumerate(selected):
                        path=ROOT/'data'/relative
                        if not path.exists() or path.stat().st_size!=member.file_size:
                            payload=archive.read(member)  # ZIP CRC checked by zipfile.
                            path.parent.mkdir(parents=True,exist_ok=True)
                            tmp=path.with_suffix(path.suffix+'.part');tmp.write_bytes(payload);tmp.replace(path)
                            with (audit/'generation_input_files.jsonl').open('a') as log:
                                log.write(json.dumps(dict(path=str(relative),source_id=fid,member=member.filename,
                                    sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))+'\n')
                        status=dict(stage='downloading',kind=kind,source=fid,files=index+1,total=len(selected))
                        state_path.write_text(json.dumps(status,indent=2));print(status,flush=True)
                last=None;break
            except Exception as e:
                last=e;print('Input download failed',repr(e),flush=True);time.sleep(2)
        if last:
            state_path.write_text(json.dumps(dict(stage='blocked',source=fid,error=str(last)),indent=2))
            raise last
    state_path.write_text(json.dumps(dict(stage='inputs_downloaded',generation_complete=False),indent=2))

if __name__=='__main__':main()
