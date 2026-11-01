"""Build item response matrices from a checkout of github.com/SWE-bench/experiments.

    git clone --filter=blob:none --no-checkout https://github.com/SWE-bench/experiments swe-exp
    cd swe-exp && git sparse-checkout set --no-cone \
        'evaluation/*/*/results/results.json' 'evaluation/*/*/per_instance_details.json' \
        'evaluation/*/*/metadata.y*ml' && git checkout
    uv run python data/build_swebench.py path/to/swe-exp

Writes src/benchmeasure/data/swebench_{split}.csv.gz (one row per submission, one 0/1 column per
task) and swebench_{split}_meta.csv beside it. A task missing from a submission's
resolved list counts as unresolved, which is how the leaderboard scores it.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

SPLITS = {"verified": 500, "multilingual": 300}
OUT = Path(__file__).resolve().parents[1] / "src" / "benchmeasure" / "data"


def resolved_ids(sub: Path):
    res = sub / "results" / "results.json"
    if res.exists():
        return set(json.loads(res.read_text())["resolved"])
    det = sub / "per_instance_details.json"
    if det.exists():
        return {k for k, v in json.loads(det.read_text()).items() if v.get("resolved")}
    return None


def metadata(sub: Path):
    for name in ("metadata.yaml", "metadata.yml"):
        f = sub / name
        if f.exists():
            return yaml.safe_load(f.read_text()) or {}
    return {}


def task_universe(root: Path, split: str):
    """Task ids seen in at least two submissions' files, which drops malformed one-off ids."""
    seen = {}
    for sub in (root / "evaluation" / split).iterdir():
        ids = set()
        det = sub / "per_instance_details.json"
        if det.exists():
            ids |= set(json.loads(det.read_text()))
        res = sub / "results" / "results.json"
        if res.exists():
            for v in json.loads(res.read_text()).values():
                if isinstance(v, list):
                    ids |= set(v)
        for i in ids:
            seen[i] = seen.get(i, 0) + 1
    return sorted(i for i, n in seen.items() if n >= 2)


def build(root: Path, split: str):
    tasks = task_universe(root, split)
    assert len(tasks) == SPLITS[split], (split, len(tasks))
    rows, meta = {}, []
    for sub in sorted((root / "evaluation" / split).iterdir()):
        ids = resolved_ids(sub)
        if ids is None:
            continue
        unknown = ids - set(tasks)
        if unknown:
            print(f"  {sub.name}: dropped unknown ids {sorted(unknown)}")
        m = metadata(sub)
        info, tags = m.get("info", {}) or {}, m.get("tags", {}) or {}
        reported = info.get("resolved")
        if reported is not None and abs(100 * len(ids & set(tasks)) / len(tasks) - float(reported)) > 1:
            print(f"  {sub.name}: dropped, per-task file gives {100 * len(ids) / len(tasks):.1f}"
                  f" but metadata reports {reported}")
            continue
        rows[sub.name] = [int(t in ids) for t in tasks]
        meta.append({
            "submission": sub.name,
            "date": sub.name[:8],
            "name": info.get("name", ""),
            "reported_resolved": info.get("resolved"),
            "model_org": tags.get("model_org", tags.get("org", "")),
            "os_model": tags.get("os_model"),
            "agent": tags.get("agent", ""),
            "reasoning_effort": tags.get("reasoning_effort", ""),
            "mini_swe_agent": "_mini-" in sub.name,
            "model": ",".join(tags.get("model") or []),
        })
    X = pd.DataFrame.from_dict(rows, orient="index", columns=tasks)
    X.index.name = "submission"
    X.to_csv(OUT / f"swebench_{split}.csv.gz")
    pd.DataFrame(meta).to_csv(OUT / f"swebench_{split}_meta.csv", index=False)
    return X


if __name__ == "__main__":
    root = Path(sys.argv[1])
    commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    for split in SPLITS:
        X = build(root, split)
        print(f"{split}: {X.shape[0]} submissions x {X.shape[1]} tasks")
    (OUT / "SOURCE").write_text(f"github.com/SWE-bench/experiments {commit}\n")
