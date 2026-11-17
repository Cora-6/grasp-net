"""Stream the public mirror, keeping selected RealSense frames, with restart offsets.

Archives are not stored. Checkpoint after each tar member; interrupted members
are replaced on retry. Dataset license and mirror identity are recorded.
"""
import argparse, hashlib, json, shutil, tarfile, time
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

REV = '1f9db26044cd2a7e629bbb465be3665364cbf2bf'
BASE = 'https://huggingface.co/datasets/DravenALG/GraspNet-1Billion/resolve/' + REV


def selected(name, stride):
    parts = PurePosixPath(name).parts
    scene = next((i for i,p in enumerate(parts) if p.startswith('scene_') and p[6:].isdigit()), None)
    if scene is None:
        return None
    tail = parts[scene:]
    if len(tail) == 3 and tail[-1].endswith('.npy'):
        sid, camera, fn = tail
        folder = 'rect'
    elif len(tail) == 4:
        sid, camera, folder, fn = tail
    else:
        return None
    if camera != 'realsense' or folder not in ('rgb','depth','rect'):
        return None
    p = Path(fn)
    if not p.stem.isdigit() or int(p.stem) % stride:
        return None
    if p.suffix != ('.npy' if folder == 'rect' else '.png'):
        return None
    return Path('scenes') / sid / camera / folder / fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stride', type=int, default=16)
    p.add_argument('--archives', nargs='+', choices=['scenes.tar','rect_labels.tar'], default=['scenes.tar','rect_labels.tar'])
    a = p.parse_args()
    if a.stride < 1 or a.stride > 256:
        p.error('stride must be 1..256')
    root = a.root.resolve()
    audit = root / 'audit'; audit.mkdir(parents=True,exist_ok=True)
    data = root / 'data'; data.mkdir(exist_ok=True)
    for archive in a.archives:
        state_path = audit / (archive + '.download.json')
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(
            archive=archive, revision=REV, stride=a.stride, next_offset=0, files=0, complete=False)
        if state['stride'] != a.stride or state['revision'] != REV:
            raise ValueError('Download protocol changed; use a new project directory')
        if state['complete']:
            continue
        for attempt in range(4):
            offset = state['next_offset']
            req = Request(BASE + '/' + archive + '?download=true&offset=%d' % offset,
                          headers={'Range':'bytes=%d-' % offset})
            try:
                with urlopen(req, timeout=60) as response:
                    if offset and (response.status != 206 or not response.headers.get('Content-Range','').startswith('bytes %d-' % offset)):
                        raise RuntimeError('Resume range not honored')
                    if 'text/' in response.headers.get('Content-Type',''):
                        raise RuntimeError('Download returned text instead of tar')
                    last = time.time()
                    with tarfile.open(fileobj=response,mode='r|') as tar:
                        for member in tar:
                            relative = selected(member.name,a.stride) if member.isfile() else None
                            if relative:
                                dest=data/relative; dest.parent.mkdir(parents=True,exist_ok=True)
                                if shutil.disk_usage(data).free < member.size + 2*1024**3:
                                    raise RuntimeError('Insufficient disk space')
                                digest=hashlib.sha256()
                                with tar.extractfile(member) as src, dest.with_suffix(dest.suffix+'.part').open('wb') as out:
                                    while True:
                                        block=src.read(1024**2)
                                        if not block: break
                                        digest.update(block); out.write(block)
                                tmp=dest.with_suffix(dest.suffix+'.part')
                                if tmp.stat().st_size != member.size:
                                    raise RuntimeError('Truncated member')
                                tmp.replace(dest)
                                with (audit/'downloaded_files.jsonl').open('a') as log:
                                    log.write(json.dumps(dict(path=str(relative),bytes=member.size,sha256=digest.hexdigest(),archive=archive))+'\n')
                                state['files']+=1
                            state['next_offset']=offset + member.offset_data + ((member.size+511)//512)*512
                            # Persist member position; tar advances over skipped contents on next iteration.
                            tmp=state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(state,indent=2)); tmp.replace(state_path)
                            if time.time()-last>20:
                                print(archive, 'read GB',round(state['next_offset']/1e9,2), 'kept files',state['files'],flush=True)
                                last=time.time()
                state['complete']=True
                state_path.write_text(json.dumps(state,indent=2))
                print(archive,'complete',state['files'],flush=True)
                break
            except Exception as e:
                state['last_error']=str(e)
                state_path.write_text(json.dumps(state,indent=2))
                print('Download attempt failed:',repr(e),flush=True)
                if attempt==3: raise
                time.sleep(2)


if __name__=='__main__': main()
