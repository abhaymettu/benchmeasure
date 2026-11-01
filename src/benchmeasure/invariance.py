"""Does a task mean the same thing for every group of models?

Differential item functioning (DIF) is a task that is easier for one group of
models than another at the same overall ability. The item level tests live in
benchdif (github.com/abhaymettu/benchdif); this module holds the test level
consequences that matter for a leaderboard: how much per-task group gap there
is beyond sampling noise, whether it cancels in the full score, and how far a
task subset can move a group comparison when chosen on other models.
"""

import numpy as np
import pandas as pd
from scipy import stats


def excess_gap_variance(X, group, n_strata: int = 1) -> dict:
    """Variance of the per-task group gap against what sampling noise alone gives.

    Under no DIF and equal ability distributions, the per-task gap between
    group means varies only by binomial noise, p_a(1 - p_a)/n_a +
    p_b(1 - p_b)/n_b. The ratio of observed to that is 1 without DIF, and
    `item_specific_share` is the part of the observed variance it cannot
    explain. If the groups differ in ability, gaps on easy and hard tasks
    differ even without DIF (the probability scale is not linear), which
    inflates the ratio. `n_strata` > 1 compares the groups within bands of
    total score instead, which removes most of that; bands where either group
    has fewer than two models are skipped.
    """
    X = np.asarray(X, float)
    g = np.asarray(group).astype(bool)
    if g.all() or not g.any():
        raise ValueError("group must contain both 0 and 1")
    total = X.mean(1)
    edges = np.quantile(total, np.linspace(0, 1, n_strata + 1))
    band = np.clip(np.searchsorted(edges, total, side="right") - 1, 0, n_strata - 1)
    obs, null, used = [], [], 0
    for b in range(n_strata):
        A, B = X[(band == b) & g], X[(band == b) & ~g]
        if len(A) < 2 or len(B) < 2:
            continue
        pa, pb = A.mean(0), B.mean(0)
        obs.append((pa - pb).var(ddof=1))
        null.append((pa * (1 - pa) / len(A) + pb * (1 - pb) / len(B)).mean())
        used += len(A) + len(B)
    ratio = np.mean(obs) / np.mean(null) if null and np.mean(null) > 0 else np.nan
    return {"n_a": int(g.sum()), "n_b": int((~g).sum()), "n_strata": n_strata, "n_used": used,
            "test_gap": float(X[g].mean() - X[~g].mean()),
            "var_ratio": float(ratio), "item_specific_share": float(1 - 1 / ratio)}


def heldout_subset_swing(X, group, k: int, n_splits: int = 20, seed: int = 0, clusters=None) -> dict:
    """How much the group gap depends on which k tasks are kept, out of sample.

    Split each group's models in half. On the first half pick the k tasks that
    most favour group A and the k that most favour group B. On the second half,
    score the gap on each subset. `mean_swing` is the difference between those
    two held-out gaps: the range spanned by the two most adversarial k-task
    subsets. `mean_max_shift` is the larger of the two one-sided moves away
    from the held-out full-test gap, which is how far a single subset moves it.
    `clusters` (one label per model, e.g. the base model) keeps every model of
    a cluster in the same half, so the same base model under two scaffolds
    cannot be both selector and judge.
    """
    X = np.asarray(X, float)
    g = np.asarray(group).astype(bool)
    if g.all() or not g.any():
        raise ValueError("group must contain both 0 and 1")
    labels = np.arange(len(X)) if clusters is None else np.asarray(clusters)
    rng = np.random.default_rng(seed)
    swings, shifts = [], []
    for _ in range(n_splits):
        first = np.zeros(len(X), bool)
        for side in (g, ~g):
            cl = np.unique(labels[side])
            pick = rng.permutation(cl)[: len(cl) // 2]
            first |= side & np.isin(labels, pick)
        a1, b1 = X[first & g], X[first & ~g]
        a2, b2 = X[~first & g], X[~first & ~g]
        if min(len(a1), len(b1), len(a2), len(b2)) == 0:
            continue
        order = np.argsort(a1.mean(0) - b1.mean(0), kind="stable")
        lo, hi = order[:k], order[-k:]
        full = a2.mean() - b2.mean()
        g_hi = a2[:, hi].mean() - b2[:, hi].mean()
        g_lo = a2[:, lo].mean() - b2[:, lo].mean()
        swings.append(g_hi - g_lo)
        shifts.append(max(abs(g_hi - full), abs(g_lo - full)))
    swings = np.asarray(swings)
    return {"k": k, "test_gap": float(X[g].mean() - X[~g].mean()),
            "mean_swing": float(swings.mean()), "min_swing": float(swings.min()),
            "mean_max_shift": float(np.mean(shifts))}


def random_subset_stability(X, k: int, n_draws: int = 1000, group=None, seed: int = 0) -> dict:
    """Rank agreement between k random tasks and the full benchmark.

    Reports the median Spearman correlation of model scores and, when a group
    is given, how often the subset reverses the sign of the full group gap.
    """
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    full = X.mean(1)
    rhos, flips = [], 0
    g = None if group is None else np.asarray(group).astype(bool)
    full_gap = None if g is None else full[g].mean() - full[~g].mean()
    for _ in range(n_draws):
        cols = rng.choice(X.shape[1], k, replace=False)
        sub = X[:, cols].mean(1)
        rhos.append(stats.spearmanr(sub, full).statistic)
        if g is not None:
            flips += np.sign(sub[g].mean() - sub[~g].mean()) != np.sign(full_gap)
    out = {"k": k, "median_spearman": float(np.median(rhos)),
           "p05_spearman": float(np.quantile(rhos, 0.05))}
    if g is not None:
        out["sign_flip_rate"] = flips / n_draws
    return out


def split_correlation(X, mask) -> dict:
    """Disattenuated correlation between two task subsets of one benchmark.

    `mask` marks tasks in part one. If the two parts measure one ability the
    true score correlation is near 1; well below 1 means the leaderboard
    ranking depends on the task mix (a model x task cluster interaction).
    """
    from .reliability import gstudy

    X = np.asarray(X, float)
    m = np.asarray(mask).astype(bool)
    x, y = X[:, m], X[:, ~m]
    r = float(np.corrcoef(x.mean(1), y.mean(1))[0, 1])
    rx, ry = gstudy(x).g_coef, gstudy(y).g_coef
    return {"n_tasks_a": int(m.sum()), "n_tasks_b": int((~m).sum()), "r_observed": r,
            "rho_a": rx, "rho_b": ry, "r_true": r / np.sqrt(rx * ry)}


def dif(X, group, q: float = 0.05) -> pd.DataFrame:
    """Per-task logistic regression DIF (uniform plus non-uniform), BH-adjusted.

    Thin wrapper over benchdif.logistic, which matches R's glm to machine
    precision. Matching is on total score, so it needs enough models per
    group (benchdif.check_adequacy warns when it does not).
    """
    from benchdif import adjust, logistic

    res = logistic(np.asarray(X, int), np.asarray(group, int), kind="both")
    res["q_value"] = adjust(res["p_value"].to_numpy(), method="bh")
    res["flag"] = res["q_value"] < q
    return res
