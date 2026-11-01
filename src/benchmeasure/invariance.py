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


def _split(X, group):
    X = np.asarray(X, float)
    g = np.asarray(group).astype(bool)
    if g.all() or not g.any():
        raise ValueError("group must contain both 0 and 1")
    return X[g], X[~g]


def excess_gap_variance(X, group) -> dict:
    """Variance of the per-task group gap against what sampling noise alone gives.

    Under no DIF, the per-task gap between group means varies only by binomial
    noise, p_a(1 - p_a)/n_a + p_b(1 - p_b)/n_b. The ratio of observed to that
    is 1 without DIF; `item_specific_share` is the part of the observed
    variance it cannot explain. Group ability differences shift every task's
    gap by about the same amount, so they do not inflate the variance.
    """
    A, B = _split(X, group)
    pa, pb = A.mean(0), B.mean(0)
    gap = pa - pb
    null = (pa * (1 - pa) / len(A) + pb * (1 - pb) / len(B)).mean()
    ratio = gap.var(ddof=1) / null if null > 0 else np.nan
    return {"n_a": len(A), "n_b": len(B), "test_gap": float(A.mean() - B.mean()),
            "var_ratio": float(ratio), "item_specific_share": float(1 - 1 / ratio)}


def heldout_subset_swing(X, group, k: int, n_splits: int = 20, seed: int = 0) -> dict:
    """How far a k-task subset moves the group gap on models it was not chosen on.

    Split each group's models in half. On the first half pick the k tasks that
    most favour group A and the k that most favour group B. Score both subsets
    on the second half. The swing is the difference between the two held-out
    gaps: how much the group comparison depends on which k tasks were kept,
    with selection and measurement on different models so the swing is not
    circular.
    """
    A, B = _split(X, group)
    rng = np.random.default_rng(seed)
    swings = []
    for _ in range(n_splits):
        ia, ib = rng.permutation(len(A)), rng.permutation(len(B))
        a1, a2 = A[ia[: len(A) // 2]], A[ia[len(A) // 2:]]
        b1, b2 = B[ib[: len(B) // 2]], B[ib[len(B) // 2:]]
        order = np.argsort(a1.mean(0) - b1.mean(0), kind="stable")
        lo, hi = order[:k], order[-k:]
        swings.append((a2[:, hi].mean() - b2[:, hi].mean())
                      - (a2[:, lo].mean() - b2[:, lo].mean()))
    swings = np.asarray(swings)
    return {"k": k, "test_gap": float(A.mean() - B.mean()),
            "mean_swing": float(swings.mean()), "min_swing": float(swings.min())}


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
