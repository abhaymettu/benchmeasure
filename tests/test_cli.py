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


def _write(path, A):
    pd.DataFrame(A, index=[f"m{i}" for i in range(len(A))]).rename_axis("model").to_csv(path)
    return str(path)


def test_degenerate_inputs_are_reported_not_printed_as_nan(tmp_path, capsys):
    import pytest
    with pytest.raises(SystemExit, match="same score on every task"):
        main(["report", _write(tmp_path / "const.csv", np.ones((6, 20), int))])
    with pytest.raises(SystemExit, match="No such file"):
        main(["report", str(tmp_path / "missing.csv")])
    noise = _write(tmp_path / "noise.csv", np.random.default_rng(1).integers(0, 2, (8, 40)))
    assert main(["report", noise]) == 0
    out = capsys.readouterr().out
    assert "nan" not in out and "not reliable" in out
    assert main(["report", noise, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["reliability"]["all"]["tasks_for_g90"] is None
    groups = tmp_path / "groups.csv"
    pd.DataFrame({"model": [f"m{i}" for i in range(8)], "group": [1] + [0] * 7}).to_csv(groups, index=False)
    with pytest.raises(SystemExit, match="at least 2 models"):
        main(["report", noise, "--groups", str(groups)])
