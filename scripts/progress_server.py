"""Read-only local live progress dashboard; no task control or remote access."""
import json, subprocess, time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def status():
    now=time.time()
    try:
        ps=subprocess.check_output(['ps','-axo','pid=,command='],text=True,timeout=3)
        processes=[line for line in ps.splitlines() if str(ROOT) in line and 'python' in line.lower()]
    except Exception:processes=[]
    download_live=any('download_data.py' in p for p in processes)
    training_live=any('run_experiments.py' in p or '/run.py train ' in p for p in processes)
    dl=[]
    for archive,label,total,pattern in [('scenes.tar','RGB 与深度图片',6080,'*.png'),('rect_labels.tar','抓取标注',3040,'*.npy')]:
        path=ROOT/'audit'/(archive+'.download.json');s=read(path)
        count=sum(1 for _ in (ROOT/'data/scenes').rglob(pattern))
        age=now-path.stat().st_mtime if path.exists() else None
        live=any('download_data.py' in p and ('--archives '+archive in p or '--archives' not in p) for p in processes)
        state=('已完成' if count>=total else '数据包已读完，标注仍不齐') if s.get('complete') else ('正在下载' if live and age is not None and age<120 else '等待网络响应' if live else '已停止')
        dl.append(dict(label=label,done=count,total=total,percent=min(100,count/total*100),state=state,age_seconds=age,read_gb=round(s.get('next_offset',0)/1e9,2)))
    histories=[]
    for p in (ROOT/'runs').glob('*/*/history.json'):
        h=read(p)
        if isinstance(h,list) and h:
            histories.append(dict(name=str(p.parent.relative_to(ROOT/'runs')),epochs=len(h),mtime=p.stat().st_mtime))
    epochs=sum(h['epochs'] for h in histories)
    active=[]
    for p in (ROOT/'runs').glob('*/*/progress.json'):
        item=read(p)
        if item and now-p.stat().st_mtime<1800:
            active.append((p.stat().st_mtime,str(p.parent.relative_to(ROOT/'runs')),item))
    current=max(active) if active else None
    jobs=len(list((ROOT/'runs').glob('*/*/complete.json')))
    evaluations=len(list((ROOT/'runs').glob('*/*/test_*/complete.json')))
    old=read(ROOT/'audit/pipeline_status.json')
    if download_live: stage='正在下载数据'
    elif training_live: stage='正在训练或评测'
    elif old.get('stage')=='complete': stage='全部实验已完成'
    elif all(d['state']=='已完成' for d in dl) and old.get('stage')=='checking_data':stage='正在检查数据'
    else:stage='任务当前未运行或正在切换阶段'
    return dict(updated=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),stage=stage,
                downloads=dl,training=dict(done=epochs,total=540,jobs=jobs,total_jobs=54,
                current=(current[1]+' · 第 '+str(current[2].get('epoch',0))+' 轮 · 批次 '+str(current[2].get('batch',0))+'/'+str(current[2].get('batches',0))) if current and training_live else max(histories,key=lambda h:h['mtime'])['name'] if histories else '等待训练开始'),
                evaluation=dict(done=evaluations,total=144),
                error=old.get('error') if not download_live and not training_live and old.get('stage')=='failed' else None)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path=self.path.split('?')[0]
        if path=='/':
            content=(ROOT/'progress.html').read_bytes();kind='text/html; charset=utf-8'
        elif path=='/status.json':
            content=json.dumps(status(),ensure_ascii=False).encode();kind='application/json; charset=utf-8'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(content)))
        self.end_headers();self.wfile.write(content)
    def log_message(self,*args):pass

if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',18765),Handler)
    print('http://127.0.0.1:18765',flush=True)
    server.serve_forever()
