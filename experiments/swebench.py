"""Every SWE-bench number in the paper, from the matrices committed in src/benchmeasure/data/.

    uv run python experiments/swebench.py          # writes results/swebench.json and figures/

Seeds are fixed, so the JSON is byte-for-byte reproducible (tests/test_reproduce.py).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmeasure.ceiling import decompose, estimate_coverage
from benchmeasure.compare import adjacent_pairs, min_detectable_difference, separable_fraction
from benchmeasure.invariance import (dif, excess_gap_variance, heldout_subset_swing,
                                     random_subset_stability, split_correlation)
from benchmeasure.reliability import bootstrap_g, gstudy
from benchmeasure.report import top_k_reliability

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, FIG = ROOT / "src" / "benchmeasure" / "data", ROOT / "results", ROOT / "figures"

# Bjarnason, Silva and Monperrus (2026), Table 1: run-to-run SD over 10 full runs
# of SWE-bench Verified, 12 model x scaffold x temperature configurations.
RUN_SD_PP = [0.7, 1.4, 1.0, 1.5, 1.1, 1.5, 1.2, 1.8, 1.0, 1.5, 1.6, 1.7]
TOP_K = [173, 100, 50, 30, 20, 10]


def load():
    X = pd.read_csv(DATA / "swebench_verified.csv.gz", index_col=0)
    meta = pd.read_csv(DATA / "swebench_verified_meta.csv").set_index("submission").loc[X.index]
    Y = pd.read_csv(DATA / "swebench_multilingual.csv.gz", index_col=0)
    mini = (DATA / "swebench_verified_mini_ids.txt").read_text().split()
    return X, meta, Y, mini


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def population_curve(X):
    """Reliability of the ranking as the compared set narrows to the top k.

    `naive_g` chooses and measures the top k on the same 500 tasks, which is
    what a reader of the leaderboard implicitly does. `g` is the split-task
    estimate from `top_k_reliability`, which is the one to believe.
    """
    score = X.mean(axis=1)
    rows = []
    for k in TOP_K:
        top = score.sort_values(ascending=False).index[:k]
        naive = gstudy(X.loc[top].to_numpy())
        t = top_k_reliability(X, k)
        rows.append({"k": k, "score_min": r(score[top].min()), "score_max": r(score[top].max()),
                     "naive_g": r(naive.g_coef), "g": r(t["g"]), "g_p10": r(t["g_p10"]),
                     "g_p90": r(t["g_p90"]), "halves_r_500": r(t["halves_r"]),
                     "sd_model_pp": r(100 * t["sd_model"], 3),
                     "sem_pp": r(100 * t["sem"], 3),
                     "tasks_for_g80": t["tasks_for_g80"], "tasks_for_g90": t["tasks_for_g90"]})
    return rows


def fixed_populations(X):
    """Frontier sets defined without looking at the scores, so no selection bias."""
    sets = {
        "mini_swe_agent_v2_batch": [i for i in X.index if "_mini-v2.0.0" in i],
        "submitted_2026": [i for i in X.index if i[:8] >= "20260101"],
        "submitted_since_2025_10": [i for i in X.index if i[:8] >= "20251001"],
    }
    out = {}
    score = X.mean(axis=1)
    for name, idx in sets.items():
        g = gstudy(X.loc[idx].to_numpy())
        out[name] = {"n": len(idx), "score_min": r(score[idx].min()), "score_max": r(score[idx].max()),
                     "sd_model_pp": r(100 * np.sqrt(g.var_model), 3), "sem_pp": r(100 * g.sem, 3),
                     "g": r(g.g_coef), "g_ci": [r(v) for v in bootstrap_g(X.loc[idx].to_numpy(), 1000)]}
    return out


def runs_projection(X, k):
    """How far repeated runs could lift top-k reliability, bounded by published run noise.

    One run per task cannot split the residual into stable model x task
    interaction and run noise. Bjarnason et al. measured run noise directly; if
    a single-run score has run SD s, then s^2 of the residual error variance is
    run noise and averaging n runs divides that part by n.
    """
    t = top_k_reliability(X, k)
    vm, err = t["var_model"], t["sem"] ** 2
    out = {}
    for label, sd in (("low", min(RUN_SD_PP)), ("median", float(np.median(RUN_SD_PP))),
                      ("high", max(RUN_SD_PP))):
        run = min((sd / 100) ** 2, err)
        out[label] = {"run_sd_pp": sd, "run_share_of_error": r(run / err, 3),
                      **{f"g_runs_{n}": r(vm / (vm + err - run + run / n)) for n in (1, 3, 5, 10)}}
    return {"k": k, "single_run_sem_pp": r(100 * np.sqrt(err), 3), **out}


def pairs(X):
    out = {}
    score = X.mean(axis=1).sort_values(ascending=False)
    for k in (20, 30):
        top = score.index[:k]
        ap = adjacent_pairs(X.loc[top].to_numpy(), list(top))
        disc = float(ap.n_discordant.median()) / X.shape[1]
        out[f"top{k}"] = {"adjacent_separable": int(ap.separable.sum()), "adjacent_pairs": len(ap),
                          "all_pairs_separable_share": r(separable_fraction(X.loc[top].to_numpy()), 3),
                          "median_adjacent_gap_pp": r(100 * ap.gap.median(), 2),
                          "median_discordant": float(ap.n_discordant.median()),
                          "mdd_pp": r(100 * min_detectable_difference(disc, X.shape[1]), 2),
                          "spread_pp": r(100 * (score[top].max() - score[top].min()), 1)}
    return out


def matched_multilingual(X, Y):
    """Same models, same scaffold (mini-SWE-agent v2), scored on Verified and Multilingual."""
    def key(s):
        return s.split("_", 2)[2].replace("-high", "").lower()
    v = {key(s): s for s in X.index if "_mini-v2" in s}
    y = {key(s): s for s in Y.index}
    common = sorted(set(v) & set(y))
    return common, X.loc[[v[k] for k in common]].to_numpy(), Y.loc[[y[k] for k in common]].to_numpy()


def coverage(X, Y):
    common, xa, ya = matched_multilingual(X, Y)
    c = estimate_coverage(xa, ya, n_boot=4000, seed=0)
    loo = []
    for i, name in enumerate(common):
        keep = np.arange(len(common)) != i
        ci = estimate_coverage(xa[keep], ya[keep], n_boot=1, seed=0)
        loo.append({"dropped": name, "r_observed": r(ci.r_observed), "r_true": r(ci.r_true),
                    "rho_x": r(ci.rho_x), "rho_y": r(ci.rho_y)})
    weakest = int(np.argmin(xa.mean(1)))
    keep = np.arange(len(common)) != weakest
    c11 = estimate_coverage(xa[keep], ya[keep], n_boot=4000, seed=0)

    def split(ce):
        d = decompose(ce.rho_x, ce.rho_y, ce.coverage)
        return {**{k: r(v) for k, v in ce.as_dict().items()},
                **{k: r(v) for k, v in d.as_dict().items()},
                "r_observed_squared": r(ce.r_observed ** 2)}

    return {"models": common,
            "verified": [r(v) for v in xa.mean(1)], "multilingual": [r(v) for v in ya.mean(1)],
            **{k: r(v) for k, v in c.as_dict().items()},
            "decomposition": split(c),
            "without_weakest": {"dropped": common[weakest], **split(c11)},
            "leave_one_out": loo}


def invariance(X, meta, mini):
    repo = np.array([t.split("__")[0] for t in X.columns])
    score = X.mean(axis=1).sort_values(ascending=False)
    out = {"repo_counts": pd.Series(repo).value_counts().to_dict()}
    rng = np.random.default_rng(0)
    for name, k in (("all", len(X)), ("top50", 50)):
        Xk = X.loc[score.index[:k]].to_numpy()
        s = split_correlation(Xk, repo == "django")
        boots = []
        for _ in range(1000):
            rows = rng.integers(0, k, k)
            with np.errstate(invalid="ignore", divide="ignore"):
                boots.append(split_correlation(Xk[rows], repo == "django")["r_true"])
        s = {kk: r(v) for kk, v in s.items()}
        s["r_true_ci"] = [r(v) for v in np.nanquantile(boots, [0.025, 0.975])]
        out[f"django_vs_rest_{name}"] = s

    g = meta.os_model.astype(bool).to_numpy()
    Xa = X.to_numpy()
    # base model as the cluster, so one model under two scaffolds stays on one side of a split
    base = meta.model.fillna(pd.Series(meta.index, index=meta.index)).to_numpy()
    flags = dif(Xa, g.astype(int))
    out["open_vs_closed"] = {
        "raw": {k: r(v) for k, v in excess_gap_variance(Xa, g).items()},
        "score_banded": {k: r(v) for k, v in excess_gap_variance(Xa, g, n_strata=5).items()},
        "dif_flagged_q05": int(flags.flag.sum()),
        "n_base_models": int(len(np.unique(base))),
        "heldout_swing_k50": {k: r(v) for k, v in
                              heldout_subset_swing(Xa, g, 50, seed=0, clusters=base).items()},
        "random_k50": {k: r(v) for k, v in random_subset_stability(Xa, 50, group=g, seed=0).items()},
    }
    # the published 50-task Mini against the full 500, on every submission and on the top 20
    m = X.columns.isin(mini)
    full, sub = X.mean(axis=1), X.loc[:, m].mean(axis=1)
    top20 = score.index[:20]
    rng = np.random.default_rng(1)
    base_sp = []
    for _ in range(2000):
        cols = rng.choice(X.shape[1], int(m.sum()), replace=False)
        base_sp.append(full[top20].corr(X.loc[top20].iloc[:, cols].mean(axis=1), method="spearman"))
    t20 = top_k_reliability(X, 20)
    out["verified_mini"] = {
        "n_tasks": int(m.sum()),
        "spearman_all": r(full.corr(sub, method="spearman")),
        "spearman_top20": r(full[top20].corr(sub[top20], method="spearman")),
        "random50_spearman_top20": [r(v) for v in np.nanquantile(base_sp, [0.1, 0.5, 0.9])],
        "projected_g_top20_at_50_tasks": r(t20["var_model"] / (t20["var_model"] + t20["var_residual"] / 50)),
        "open_closed_gap_full": r(full[g].mean() - full[~g].mean()),
        "open_closed_gap_mini": r(sub[g].mean() - sub[~g].mean()),
    }
    return out


def figures(curve, cov, X):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted,
                         "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    FIG.mkdir(exist_ok=True)

    # 1. reliability against the compared population
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=200)
    ks = [c["k"] for c in curve]
    ax.fill_between(ks, [c["g_p10"] for c in curve], [c["g_p90"] for c in curve],
                    color=blue, alpha=0.15, lw=0)
    ax.plot(ks, [c["g"] for c in curve], color=blue, lw=2, marker="o", ms=5,
            label="top k chosen on half the tasks, scored on the rest")
    ax.plot(ks, [c["naive_g"] for c in curve], color=orange, lw=2, marker="s", ms=5,
            label="chosen and scored on the same tasks (biased)")
    ax.set_xscale("log")
    ax.set_xticks(ks, [("all " + str(k)) if k == 173 else f"top {k}" for k in ks])
    ax.minorticks_off()
    ax.invert_xaxis()
    ax.set_ylim(-0.02, 1.08)
    ax.axhline(0.8, color=grid, lw=1, zorder=0)
    ax.text(ks[-1], 0.81, "0.8", color=muted, fontsize=8, va="bottom", ha="right")
    ax.set_ylabel("G coefficient of the ranking")
    ax.grid(axis="y", color=grid, lw=0.6)
    for c in curve:
        ax.annotate(f"{c['g']:.2f}", (c["k"], c["g"]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_reliability_curve.png")
    plt.close(fig)

    # 2. verified vs multilingual, matched models
    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=200)
    v, m = np.array(cov["verified"]) * 100, np.array(cov["multilingual"]) * 100
    ax.scatter(v, m, s=36, color=blue, edgecolor="white", linewidth=1.5, zorder=3)
    for name, a, b in zip(cov["models"], v, m):
        ax.annotate(name, (a, b), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=muted)
    ax.set_xlabel("SWE-bench Verified, % resolved")
    ax.set_ylabel("SWE-bench Multilingual, % resolved")
    ax.grid(color=grid, lw=0.6)
    ax.set_title(f"Same scaffold, {len(v)} models: r = {cov['r_observed']:.2f}, "
                 f"true-score r = {cov['r_true']:.2f}", loc="left", fontsize=9, color=ink)
    fig.tight_layout()
    fig.savefig(FIG / "fig_multilingual.png")
    plt.close(fig)


def main():
    X, meta, Y, mini = load()
    curve = population_curve(X)
    cov = coverage(X, Y)
    res = {
        "source": (DATA / "SOURCE").read_text().strip(),
        "n_submissions": int(X.shape[0]), "n_tasks": int(X.shape[1]),
        "population_curve": curve,
        "fixed_populations": fixed_populations(X),
        "runs_projection_top20": runs_projection(X, 20),
        "pairs": pairs(X),
        "coverage_multilingual": cov,
        "invariance": invariance(X, meta, mini),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "swebench.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item()) + "\n")
    figures(curve, cov, X)
    print(json.dumps({"curve": [(c["k"], c["naive_g"], c["g_p10"], c["g"], c["g_p90"]) for c in curve],
                      "fixed": {k: v["g"] for k, v in res["fixed_populations"].items()},
                      "runs": res["runs_projection_top20"],
                      "pairs": res["pairs"], "coverage": {k: cov[k] for k in ("r_observed", "r_true", "coverage", "ci_low", "ci_high")}},
                     indent=1))


if __name__ == "__main__":
    main()
