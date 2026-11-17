"""Run from the Mac terminal to migrate the active training queue to MPS."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/graspnet_main"
LAUNCHER = ROOT / "scripts/run_model_comparison.py"
LOG = ROOT / "outputs/graspnet_main_launcher.log"
DATASET = Path(os.environ.get("GRASPNET_PROJECT", str(ROOT)))

sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib"))

import torch
from models import build_model, model_loss


def check_gpu():
    if not torch.backends.mps.is_available():
        raise SystemExit("这个终端也无法使用 Mac GPU。现有训练未动。")
    for name in ("ggcnn_depth", "grconvnet", "dual_concat", "early_concat"):
        model = build_model(name, "original").to("mps")
        image = torch.zeros((1, 4, 300, 300), device="mps")
        output = model(image)
        loss = model_loss(model, output, torch.zeros_like(output))
        loss.backward()
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"GPU check failed: {name}")
    print("Mac GPU 检查通过。", flush=True)


def running_queue():
    rows = subprocess.check_output(["ps", "-axo", "pid=,ppid=,command="], text=True)
    processes = []
    for line in rows.splitlines():
        pid, parent, command = line.split(None, 2)
        processes.append((int(pid), int(parent), command))
    holders = subprocess.run(["lsof", "-t", str(LOG)], text=True, capture_output=True)
    ids = {int(pid) for pid in holders.stdout.split()}
    launchers = [(pid, command) for pid, _, command in processes
                 if pid in ids and "run_model_comparison.py" in command]
    if len(launchers) > 1:
        raise SystemExit("发现多个训练队列，保留现有进程。")
    if not launchers:
        return []
    pid, command = launchers[0]
    if "--device mps" in command:
        raise SystemExit("队列已经设置为使用 Mac GPU。")
    children = [child for child, parent, command in processes
                if parent == pid and "run.py train " in command]
    return [pid, *children]


def switch():
    check_gpu()
    pids = running_queue()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(20):
        if not pids:
            break
        states = subprocess.run(["ps", "-o", "stat=", "-p", ",".join(map(str, pids))],
                                text=True, capture_output=True)
        if not any(not state.startswith("Z") for state in states.stdout.split()):
            break
        time.sleep(0.5)
    else:
        raise SystemExit("旧训练进程尚未退出，请稍后重试。")

    command = [
        sys.executable, "-u", str(LAUNCHER), "--research-plan", "--train-only", "--resume",
        "--device", "mps", "--reuse-grconvnet", str(DATASET / "runs/grconvnet"),
        "--data", str(DATASET / "data"), "--split", str(DATASET / "split.json"),
        "--out", str(OUT),
    ]
    with LOG.open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
    time.sleep(2)
    if process.poll() is not None:
        raise SystemExit(f"GPU 队列启动失败，请查看 {LOG}")
    print(f"GPU 训练已启动（进程 {process.pid}）。")


if __name__ == "__main__":
    switch()
