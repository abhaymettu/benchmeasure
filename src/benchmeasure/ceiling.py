"""Validity ceilings: how much of a criterion a benchmark score can predict.

Classical attenuation (Spearman 1904): unreliability in either measure bounds
the correlation they can show.

    r_observed <= sqrt(rho_X * rho_Y) * r_true

Writing the true score correlation squared as coverage C, the share of the
criterion's true variance the benchmark's construct shares, gives the ceiling
on explained variance

    R^2 <= rho_X * rho_Y * C

and an exact split of the unit total into four slices (benchmark noise,
criterion noise, uncovered variance, reachable ceiling). When the same models
are scored on a second benchmark, C can be estimated instead of assumed: it is
the disattenuated correlation squared.
"""

from dataclasses import asdict, dataclass

import numpy as np


def ceiling(rho_x: float, rho_y: float, coverage: float) -> float:
    """Upper bound on R^2 between benchmark and criterion."""
    for name, v in (("rho_x", rho_x), ("rho_y", rho_y), ("coverage", coverage)):
        if not 0 <= v <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    return rho_x * rho_y * coverage


def disattenuate(r_observed: float, rho_x: float, rho_y: float) -> float:
    """Correlation between true scores implied by an observed correlation."""
    return r_observed / np.sqrt(rho_x * rho_y)


@dataclass(frozen=True)
class Decomposition:
    benchmark_noise: float
    criterion_noise: float
    uncovered: float
    reachable: float

    def as_dict(self):
        return asdict(self)


def decompose(rho_x: float, rho_y: float, coverage: float) -> Decomposition:
    """Split the unit total of criterion variance into the four sources of the gap.

    Benchmark noise is taken first (1 - rho_x), then criterion noise on what is
    left, then the uncovered share. The order is a convention; the reachable
    slice does not depend on it.
    """
    reach = ceiling(rho_x, rho_y, coverage)
    bench = 1 - rho_x
    crit = rho_x * (1 - rho_y)
    unc = rho_x * rho_y * (1 - coverage)
    return Decomposition(bench, crit, unc, reach)


@dataclass(frozen=True)
class CoverageEstimate:
    n_models: int
    r_observed: float
    rho_x: float
    rho_y: float
    r_true: float
    coverage: float
    ci_low: float
    ci_high: float

    def as_dict(self):
        return asdict(self)


def estimate_coverage(X, Y, n_boot: int = 4000, seed: int = 0, level: float = 0.95):
    """Coverage of criterion Y by benchmark X from the same models on both.

    X and Y are (models x tasks) matrices with rows in the same model order.
    Reliability of each score is its G coefficient over tasks within these
    models, so the correction uses the noise that is actually in each score.
    The CI resamples models, the scarce side; tasks are held fixed inside it.
    The point estimate of r_true can exceed 1 in small samples; coverage is
    reported clipped to [0, 1] and the raw r_true is kept.
    """
    from .reliability import gstudy

    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if len(X) != len(Y):
        raise ValueError("X and Y need the same models in the same order")

    def one(rows):
        x, y = X[rows].mean(1), Y[rows].mean(1)
        r = np.corrcoef(x, y)[0, 1]
        rx, ry = gstudy(X[rows]).g_coef, gstudy(Y[rows]).g_coef
        return r, rx, ry, r / np.sqrt(rx * ry)

    n = len(X)
    r, rx, ry, rt = one(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        rows = rng.integers(0, n, n)
        if len(np.unique(rows)) < 3:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            boots.append(one(rows)[3])
    boots = np.clip(np.asarray(boots), 0, 1) ** 2
    a = (1 - level) / 2
    lo, hi = np.nanquantile(boots, [a, 1 - a])
    return CoverageEstimate(n, r, rx, ry, rt, float(np.clip(rt, 0, 1) ** 2), lo, hi)
