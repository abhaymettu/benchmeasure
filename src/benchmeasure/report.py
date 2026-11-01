"""One call that answers the three questions for any results matrix.

1. Reliability: does the ranking replicate, for the whole board and for its top?
2. Invariance: do tasks mean the same for two groups of models?
3. Validity ceiling: how much of a second benchmark can this one predict?
"""

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from .ceiling import ceiling, estimate_coverage
from .compare import adjacent_pairs, min_detectable_difference
from .invariance import excess_gap_variance, heldout_subset_swing, random_subset_stability
from .reliability import gstudy


def load_matrix(path) -> pd.DataFrame:
    """Read a results file into a (models x tasks) 0/1 or score matrix.

    Accepts a wide CSV (first column the model, one column per task) or a long
    CSV/JSONL with columns model, task, score. Repeated (model, task) rows, as
    from several runs, are averaged.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if {"model", "task", "score"} <= set(cols):
        df = df.rename(columns={cols["model"]: "model", cols["task"]: "task", cols["score"]: "score"})
        X = df.pivot_table(index="model", columns="task", values="score", aggfunc="mean")
    else:
        X = df.set_index(df.columns[0])
    X = X.apply(pd.to_numeric, errors="coerce")
    missing = X.isna().any(axis=0)
    if missing.any():
        X = X.loc[:, ~missing]
    if X.shape[0] < 3 or X.shape[1] < 2:
        raise ValueError(f"need at least 3 models and 2 complete tasks, got {X.shape}")
    X.attrs["dropped_tasks"] = int(missing.sum())
    return X


def example_path(name: str = "swebench_verified.csv.gz") -> Path:
    """Path to a bundled public results matrix (SWE-bench Verified by default)."""
    return Path(str(resources.files("benchmeasure") / "data" / name))


def top_k_reliability(X: pd.DataFrame, k: int, reps: int = 50, seed: int = 0) -> dict:
    """G coefficient of the top k, chosen on half the tasks and measured on the rest."""
    rng = np.random.default_rng(seed)
    n = X.shape[1]
    vm, vr = [], []
    for _ in range(reps):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        top = X.iloc[:, a].mean(axis=1).sort_values(ascending=False).index[:k]
        g = gstudy(X.loc[top].iloc[:, b].to_numpy())
        vm.append(g.var_model)
        vr.append(g.var_residual)
    vm, vr = float(np.median(vm)), float(np.median(vr))
    g = vm / (vm + vr / n) if vm + vr > 0 else float("nan")
    need = int(np.ceil(9 * vr / vm)) if vm > 0 else None
    return {"k": k, "g": g, "sd_model": np.sqrt(vm), "sem": np.sqrt(vr / n), "tasks_for_g90": need}


def report(X: pd.DataFrame, group=None, criterion: pd.DataFrame = None, top=(10, 20)) -> dict:
    out = {"n_models": X.shape[0], "n_tasks": X.shape[1]}
    A = X.to_numpy(float)
    g = gstudy(A)
    out["reliability"] = {"all": {"g": g.g_coef, "phi": g.phi, "sd_model": np.sqrt(g.var_model),
                                  "sem": g.sem, "tasks_for_g90": g.tasks_for(0.9)},
                          "top": [top_k_reliability(X, k) for k in top if 3 <= k < len(X)]}
    binary = np.isin(A, (0, 1)).all()
    if binary:
        k = max([k for k in top if k < len(X)], default=len(X))
        lead = X.loc[X.mean(axis=1).sort_values(ascending=False).index[:k]]
        ap = adjacent_pairs(lead.to_numpy(), list(lead.index))
        disc = float(ap.n_discordant.median()) / X.shape[1]
        out["pairs"] = {"k": k, "adjacent_separable": int(ap.separable.sum()),
                        "adjacent_pairs": len(ap),
                        "mdd": min_detectable_difference(disc, X.shape[1]),
                        "table": ap}
    if group is not None:
        grp = np.asarray(group).astype(bool)
        kk = max(5, min(50, X.shape[1] // 10))
        out["invariance"] = {**excess_gap_variance(A, grp),
                             "heldout_swing": heldout_subset_swing(A, grp, kk)["mean_swing"],
                             "random_subset": random_subset_stability(A, kk, n_draws=400, group=grp),
                             "subset_k": kk}
    if criterion is not None:
        common = X.index.intersection(criterion.index)
        if len(common) < 4:
            raise ValueError(f"criterion shares only {len(common)} models with the benchmark; need 4")
        c = estimate_coverage(X.loc[common].to_numpy(float), criterion.loc[common].to_numpy(float))
        out["validity"] = {**c.as_dict(), "ceiling_r2": ceiling(c.rho_x, c.rho_y, c.coverage)}
    return out


def _pct(x):
    return f"{100 * x:.1f} pp"


def render(res: dict, name: str = "results") -> str:
    lines = [f"benchmeasure report: {name}",
             f"{res['n_models']} models x {res['n_tasks']} tasks", ""]
    rel = res["reliability"]
    lines += ["1. Reliability (generalizability theory, models x tasks)",
              f"   whole board   G = {rel['all']['g']:.3f}   model SD {_pct(rel['all']['sd_model'])}"
              f"   score SEM {_pct(rel['all']['sem'])}"]
    for t in rel["top"]:
        need = "n/a" if t["tasks_for_g90"] is None else f"{t['tasks_for_g90']:,}"
        lines.append(f"   top {t['k']:<3d}       G = {t['g']:.3f}   model SD {_pct(t['sd_model'])}"
                     f"   tasks for G = 0.9: {need}")
    lines.append("   (top k chosen on half the tasks and scored on the other half, median of 50 splits)")
    if "pairs" in res:
        p = res["pairs"]
        lines += ["", f"   leaderboard neighbours in the top {p['k']}: {p['adjacent_separable']} of "
                  f"{p['adjacent_pairs']} separable by exact McNemar at 0.05;"
                  f" smallest gap detectable at 80% power about {_pct(p['mdd'])}"]
    if "invariance" in res:
        v = res["invariance"]
        lines += ["", "2. Invariance (group 1 vs group 0)",
                  f"   test-level gap {_pct(v['test_gap'])}; per-task gap variance {v['var_ratio']:.2f}x"
                  f" sampling noise ({100 * max(v['item_specific_share'], 0):.0f}% task-specific)",
                  f"   a {v['subset_k']}-task subset chosen on other models moves the gap by"
                  f" {_pct(v['heldout_swing'])}",
                  f"   random {v['subset_k']}-task subsets reverse the gap's sign"
                  f" {100 * v['random_subset']['sign_flip_rate']:.1f}% of the time"]
    if "validity" in res:
        c = res["validity"]
        lines += ["", f"3. Validity ceiling against the criterion ({c['n_models']} shared models)",
                  f"   observed r = {c['r_observed']:.3f}; reliabilities {c['rho_x']:.3f} and {c['rho_y']:.3f}",
                  f"   true-score r = {c['r_true']:.3f}; coverage C = {c['coverage']:.3f}"
                  f" (95% CI {c['ci_low']:.2f} to {c['ci_high']:.2f})",
                  f"   ceiling R^2 <= rho_x * rho_y * C = {c['ceiling_r2']:.3f}"]
    return "\n".join(lines)
