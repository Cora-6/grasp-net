"""Read-only local dashboard for the four-model GraspNet training run."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/graspnet_main"
OLD = ROOT / "runs/grconvnet"


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def status():
    plan = read_json(RUN / "comparison_protocol.json", {})
    models = [*plan.get("main_models", []), *plan.get("ablation", [])[1:]]
    jobs = []
    for seed in plan.get("seeds", []):
        for model in models:
            name = f"{model}_s{seed}"
            folder = RUN / name
            log = RUN / f"{name}_train.log"
            jobs.append({
                "name": name, "model": model, "seed": seed,
                "progress": read_json(folder / "progress.json", {}),
                "history": read_json(folder / "history.json", []),
                "device": read_json(folder / "config.json", {}).get("device"),
                "has_checkpoint": (folder / "last.pt").is_file(),
                "started": log.is_file(),
                "reused": name in plan.get("reused", {}),
                "excluded": name in plan.get("excluded", []),
            })
    old = {
        str(seed): read_json(OLD / f"baseline_s{seed}" / "history.json", [])
        for seed in (42, 43)
    }
    log = ROOT / "outputs/graspnet_main_launcher.log"
    lines = log.read_text().splitlines() if log.exists() else []
    return {"plan": plan, "jobs": jobs, "historical": old, "launcher": lines[-14:]}


def write_dashboard():
    template = (ROOT / "main_training.html").read_text()
    payload = json.dumps(status(), ensure_ascii=False).replace("<", "\\u003c")
    html = template.replace("__TRAINING_STATUS__", payload, 1)
    output = RUN / "training_dashboard.html"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    while True:
        write_dashboard()
        time.sleep(5)
