import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmeasure.cli import main
from benchmeasure.report import load_matrix


def test_demo_runs(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "173 models x 500 tasks" in out and "coverage C" in out


def test_report_long_format_with_groups_and_criterion(tmp_path, capsys):
    rng = np.random.default_rng(0)
    models = [f"m{i}" for i in range(40)]
    theta = rng.normal(0, 1, 40)
    rows = []
    for name, extra in (("a", 0), ("b", 1)):
        b = rng.normal(0, 1, 60)
        p = 1 / (1 + np.exp(-(theta[:, None] - b)))
        for run in range(2):
            x = rng.random(p.shape) < p
            rows += [{"model": m, "task": f"{name}{j}", "score": int(x[i, j]), "run": run}
                     for i, m in enumerate(models) for j in range(60)]
    df = pd.DataFrame(rows)
    df[df.task.str.startswith("a")].to_csv(tmp_path / "a.csv", index=False)
    df[df.task.str.startswith("b")].to_json(tmp_path / "b.jsonl", orient="records", lines=True)
    pd.DataFrame({"model": models, "family": ["x", "y"] * 20}).to_csv(tmp_path / "g.csv", index=False)

    X = load_matrix(tmp_path / "a.csv")
    assert X.shape == (40, 60) and set(np.unique(X.to_numpy())) <= {0, 0.5, 1}

    assert main(["report", str(tmp_path / "a.csv"), "--groups", str(tmp_path / "g.csv"),
                 "--group-col", "family", "--criterion", str(tmp_path / "b.jsonl"), "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["n_models"] == 40 and "invariance" in res and "validity" in res
    assert res["validity"]["r_true"] > res["validity"]["r_observed"]


def test_console_script_installed():
    out = subprocess.run([sys.executable, "-m", "benchmeasure.cli", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "report" in out.stdout
