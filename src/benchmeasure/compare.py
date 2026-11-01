"""Can two models on the same tasks be told apart?

Two models that attempted the same tasks give a paired comparison. Only the
discordant tasks (one solved, the other not) carry information, so the right
test is McNemar's, exact when the discordant count is small (McNemar 1947).
"""

import numpy as np
import pandas as pd
from scipy import stats


def mcnemar(a, b):
    """Exact two-sided McNemar test for paired 0/1 vectors. Returns (diff, n_discordant, p)."""
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(stats.binomtest(only_a, n, 0.5).pvalue)
    return (a.mean() - b.mean(), n, p)


def min_detectable_difference(discordance: float, n_tasks: int,
                              alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true score gap a paired test detects with the given power.

    Normal approximation: with discordant fraction d, the paired difference has
    standard error close to sqrt(d / n) when the gap is small next to d.
    """
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(discordance / n_tasks))


def adjacent_pairs(X, labels=None, alpha: float = 0.05):
    """Test each leaderboard neighbour pair (sorted by score) with exact McNemar.

    Returns a DataFrame, best model first, with the gap to the next model down,
    the discordant count, the p value and whether the gap is significant.
    """
    X = np.asarray(X)
    labels = np.arange(len(X)) if labels is None else np.asarray(labels)
    order = np.argsort(-X.mean(axis=1), kind="stable")
    rows = []
    for hi, lo in zip(order[:-1], order[1:]):
        diff, n, p = mcnemar(X[hi], X[lo])
        rows.append({"model": labels[hi], "next": labels[lo], "gap": diff,
                     "n_discordant": n, "p_value": p, "separable": p < alpha})
    return pd.DataFrame(rows)


def separable_fraction(X, alpha: float = 0.05) -> float:
    """Share of all model pairs whose difference is significant by exact McNemar (no correction)."""
    X = np.asarray(X)
    n = len(X)
    hits = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            hits += mcnemar(X[i], X[j])[2] < alpha
            total += 1
    return hits / total if total else np.nan
