"""Generalizability theory for benchmark score matrices.

A benchmark run is a crossed design: every model (the object of measurement)
attempts every task. With one attempt per cell the score of model m on task i
decomposes into

    x_mi = mu + model_m + task_i + (model x task)_mi,e

and the leaderboard score is the row mean. The model x task term carries both
real model-by-task interaction and run-to-run noise; with one run they cannot
be separated, and both are error when the claim is about tasks like these
rather than these exact tasks. Variance components come from the two-way ANOVA
mean squares (Cronbach et al. 1972; Brennan 2001).

    G coefficient (relative)   E rho^2 = s2_model / (s2_model + s2_mt / n_tasks)
    dependability (absolute)   Phi     = s2_model / (s2_model + (s2_task + s2_mt) / n_tasks)

E rho^2 is the reliability of the ranking and equals Cronbach's alpha (KR-20
for 0/1 scores, up to the variance divisor).
Phi is the reliability of the score as an absolute number. Both depend on the
spread of the models compared, which is the point: the same benchmark is
reliable for one comparison and not for another.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GStudy:
    n_models: int
    n_tasks: int
    var_model: float
    var_task: float
    var_residual: float
    g_coef: float
    phi: float
    sem: float

    def as_dict(self):
        return asdict(self)

    def g_at(self, n_tasks: int) -> float:
        """Projected G coefficient with a different number of tasks (D study)."""
        return self.var_model / (self.var_model + self.var_residual / n_tasks)

    def tasks_for(self, target: float):
        """Tasks needed for the G coefficient to reach `target` at this model spread.

        None when the models show no spread beyond noise: no number of tasks gets there.
        """
        if self.var_model <= 0:
            return None
        return int(np.ceil(target / (1 - target) * self.var_residual / self.var_model))


def gstudy(X) -> GStudy:
    """Variance components of a complete (models x tasks) score matrix.

    Scores may be 0/1 or continuous. Negative component estimates are set to
    zero, the usual convention.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) < 2:
        raise ValueError("need a (models x tasks) matrix with at least 2 of each")
    if np.isnan(X).any():
        raise ValueError("matrix has missing cells; the ANOVA estimators need a complete design")
    n_p, n_i = X.shape
    grand = X.mean()
    row = X.mean(axis=1)
    col = X.mean(axis=0)
    ms_p = n_i * ((row - grand) ** 2).sum() / (n_p - 1)
    ms_i = n_p * ((col - grand) ** 2).sum() / (n_i - 1)
    resid = X - row[:, None] - col[None, :] + grand
    ms_r = (resid ** 2).sum() / ((n_p - 1) * (n_i - 1))
    s2_p = max((ms_p - ms_r) / n_i, 0.0)
    s2_i = max((ms_i - ms_r) / n_p, 0.0)
    s2_r = ms_r
    rel_err = s2_r / n_i
    g = s2_p / (s2_p + rel_err) if s2_p + rel_err > 0 else np.nan
    phi = s2_p / (s2_p + (s2_i + s2_r) / n_i) if s2_p + s2_i + s2_r > 0 else np.nan
    return GStudy(n_p, n_i, s2_p, s2_i, s2_r, g, phi, float(np.sqrt(rel_err)))


def gstudy_runs(X):
    """Variance components for a complete (models x tasks x runs) array.

    Separates run-to-run noise from model x task interaction, which a single
    run cannot do. Returns a dict of the seven p x i x r components plus the G
    coefficient for the given number of tasks and runs.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or min(X.shape) < 2:
        raise ValueError("need a (models x tasks x runs) array with at least 2 of each")
    n_p, n_i, n_r = X.shape
    g = X.mean()
    mp, mi, mr = X.mean((1, 2)), X.mean((0, 2)), X.mean((0, 1))
    mpi, mpr, mir = X.mean(2), X.mean(1), X.mean(0)
    ss = {
        "p": n_i * n_r * ((mp - g) ** 2).sum(),
        "i": n_p * n_r * ((mi - g) ** 2).sum(),
        "r": n_p * n_i * ((mr - g) ** 2).sum(),
        "pi": n_r * ((mpi - mp[:, None] - mi[None, :] + g) ** 2).sum(),
        "pr": n_i * ((mpr - mp[:, None] - mr[None, :] + g) ** 2).sum(),
        "ir": n_p * ((mir - mi[:, None] - mr[None, :] + g) ** 2).sum(),
    }
    resid = (X - mpi[:, :, None] - mpr[:, None, :] - mir[None, :, :]
             + mp[:, None, None] + mi[None, :, None] + mr[None, None, :] - g)
    ss["pir"] = (resid ** 2).sum()
    df = {"p": n_p - 1, "i": n_i - 1, "r": n_r - 1, "pi": (n_p - 1) * (n_i - 1),
          "pr": (n_p - 1) * (n_r - 1), "ir": (n_i - 1) * (n_r - 1),
          "pir": (n_p - 1) * (n_i - 1) * (n_r - 1)}
    ms = {k: ss[k] / df[k] for k in ss}
    v = {"pir": ms["pir"]}
    v["pi"] = (ms["pi"] - ms["pir"]) / n_r
    v["pr"] = (ms["pr"] - ms["pir"]) / n_i
    v["ir"] = (ms["ir"] - ms["pir"]) / n_p
    v["p"] = (ms["p"] - ms["pi"] - ms["pr"] + ms["pir"]) / (n_i * n_r)
    v["i"] = (ms["i"] - ms["pi"] - ms["ir"] + ms["pir"]) / (n_p * n_r)
    v["r"] = (ms["r"] - ms["pr"] - ms["ir"] + ms["pir"]) / (n_p * n_i)
    v = {k: max(val, 0.0) for k, val in v.items()}
    err = v["pi"] / n_i + v["pr"] / n_r + v["pir"] / (n_i * n_r)
    v["g_coef"] = v["p"] / (v["p"] + err) if v["p"] + err > 0 else np.nan
    return v


def bootstrap_g(X, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """Percentile CI for the G coefficient, resampling models and tasks independently.

    Two-way resampling double counts the interaction and so runs wide; read the
    interval as conservative.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    n_p, n_i = X.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        rows = rng.integers(0, n_p, n_p)
        cols = rng.integers(0, n_i, n_i)
        out[b] = gstudy(X[np.ix_(rows, cols)]).g_coef
    a = (1 - level) / 2
    return tuple(np.nanquantile(out, [a, 1 - a]))
