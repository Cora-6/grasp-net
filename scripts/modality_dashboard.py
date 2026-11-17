"""Read-only progress page for the three additional modality experiments."""
import ast
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
JOBS = [('GG-CNN RGB-D', 'ggcnn_modalities', 'ggcnn'),
        ('GG-CNN2 Depth', 'ggcnn_modalities', 'ggcnn2_depth'),
        ('GG-CNN2 RGB-D', 'ggcnn2_comparison', 'ggcnn2')]
PARTS = [('test_seen', '已见物体'), ('test_similar', '相似物体'), ('test_novel', '新物体')]


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def status():
    jobs = []
    for label, queue, model in JOBS:
        base = ROOT / 'outputs' / queue
        folder = base / (model + '_s42')
        progress = read(folder / 'progress.json', {})
        history = read(folder / 'history.json', [])
        config = read(folder / 'config.json', {})
        total = config.get('epochs', 20)
        trained = len(history) >= total
        completed = len(history)
        fraction = max(completed, progress.get('epoch', 1) - 1 + progress.get('batch', 0) / max(1, progress.get('batches', 1)))
        # Reserve the last fraction until validation and checkpoint saving finish.
        percent = 100 if trained else min(99.9, 100 * fraction / total)
        evaluations = []
        logs = [base / (model + '_s42_train.log')]
        for part, title in PARTS:
            log = base / f'{model}_s42_{part}.log'
            logs.append(log)
            rows = read(folder / part / 'metrics.json', [])
            done = (folder / part / 'metrics.csv').is_file() and len(rows) == 11
            observed = set()
            if log.exists():
                for line in log.read_text(errors='replace').splitlines():
                    if not line.startswith("{'condition':"):
                        continue
                    try:
                        item = ast.literal_eval(line)
                        observed.add((item['condition'], item['severity']))
                    except (ValueError, SyntaxError, KeyError):
                        pass
            clean = next((r['accuracy'] for r in rows if r['condition'] == 'clean'), None)
            evaluations.append(dict(title=title, done=done, conditions=11 if done else min(11, len(observed)), clean=clean))
        all_done = all(e['done'] for e in evaluations)
        paths = [folder / 'progress.json', folder / 'history.json', *logs]
        mtimes = [p.stat().st_mtime for p in paths if p.exists()]
        age = time.time() - max(mtimes) if mtimes else None
        error = None
        for log in logs:
            if log.exists():
                tail = log.read_text(errors='replace').splitlines()[-25:]
                if any('Traceback (most recent call last)' in s for s in tail):
                    error = tail[-1] if tail else '运行异常'
        stage = '全部完成' if all_done else '训练完成 · 等待评测' if trained else '训练中' if progress else '排队中'
        if trained and any(e['conditions'] for e in evaluations) and not all_done:
            stage = '评测中'
        if error:
            stage = '运行异常'
        elif age is not None and age > 300 and not all_done and not trained:
            stage = '进度超过 5 分钟未更新'
        jobs.append(dict(label=label, stage=stage, percent=round(percent, 1), completed=completed,
                         total=total, epoch=progress.get('epoch', 0), batch=progress.get('batch', 0),
                         batches=progress.get('batches', 180), history=history, evaluations=evaluations,
                         age=round(age) if age is not None else None, error=error,
                         done=all_done, device=config.get('device', 'cpu')))
    return dict(updated=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S'), jobs=jobs)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/':
            content = (ROOT / 'modality_progress.html').read_bytes()
            kind = 'text/html; charset=utf-8'
        elif path == '/status.json':
            content = json.dumps(status(), ensure_ascii=False).encode()
            kind = 'application/json; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 18767), Handler)
    print('http://127.0.0.1:18767', flush=True)
    server.serve_forever()
