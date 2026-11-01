import numpy as np
import pytest

from benchmeasure.ceiling import ceiling, decompose, disattenuate, estimate_coverage
from benchmeasure.compare import adjacent_pairs, mcnemar, min_detectable_difference
from benchmeasure.invariance import excess_gap_variance, heldout_subset_swing, split_correlation
from benchmeasure.reliability import gstudy, gstudy_runs


def cronbach_alpha(X):
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(0, ddof=1).sum() / X.sum(1).var(ddof=1))


def test_g_coefficient_equals_cronbach_alpha():
    rng = np.random.default_rng(1)
    theta = rng.normal(0, 1, 120)
    b = rng.normal(0, 1, 60)
    X = (rng.random((120, 60)) < 1 / (1 + np.exp(-(theta[:, None] - b)))).astype(int)
    assert gstudy(X).g_coef == pytest.approx(cronbach_alpha(X), abs=1e-12)


def test_gstudy_recovers_known_components():
    rng = np.random.default_rng(2)
    n_p, n_i = 400, 300
    X = (rng.normal(0, 2, (n_p, 1)) + rng.normal(0, 1, (1, n_i))
         + rng.normal(0, 3, (n_p, n_i)))
    g = gstudy(X)
    assert g.var_model == pytest.approx(4, rel=0.15)
    assert g.var_task == pytest.approx(1, rel=0.2)
    assert g.var_residual == pytest.approx(9, rel=0.05)
    assert g.g_at(300) == pytest.approx(g.g_coef)
    assert g.g_at(g.tasks_for(0.9)) >= 0.9 > g.g_at(g.tasks_for(0.9) - 1)


def test_gstudy_runs_separates_run_noise():
    rng = np.random.default_rng(3)
    n_p, n_i, n_r = 60, 80, 6
    X = (rng.normal(0, 1.0, (n_p, 1, 1)) + rng.normal(0, 1.0, (1, n_i, 1))
         + rng.normal(0, 1.5, (n_p, n_i, 1)) + rng.normal(0, 2.0, (n_p, n_i, n_r)))
    v = gstudy_runs(X)
    assert v["p"] == pytest.approx(1.0, rel=0.35)
    assert v["pi"] == pytest.approx(2.25, rel=0.15)
    assert v["pir"] == pytest.approx(4.0, rel=0.05)
    assert v["pr"] < 0.05 and v["r"] < 0.05


def test_mcnemar_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.contingency_tables")
    rng = np.random.default_rng(4)
    a = rng.random(500) < 0.6
    b = np.where(rng.random(500) < 0.8, a, ~a)
    diff, n, p = mcnemar(a, b)
    table = [[(a & b).sum(), (a & ~b).sum()], [(~a & b).sum(), (~a & ~b).sum()]]
    assert p == pytest.approx(sm.mcnemar(table, exact=True).pvalue, rel=1e-9)
    assert n == (a != b).sum()


def test_adjacent_pairs_sorted_best_first():
    X = np.array([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]])
    ap = adjacent_pairs(X, ["mid", "top", "low"])
    assert list(ap.model) == ["top", "mid"] and list(ap.next) == ["mid", "low"]


def test_min_detectable_difference_scales_with_root_n():
    assert min_detectable_difference(0.1, 400) == pytest.approx(
        2 * min_detectable_difference(0.1, 1600))


def test_ceiling_reproduces_agent_validity_ceiling_numbers():
    # agent-validity-ceiling's frontier row: rho_X 0.941, rho_Y 0.75, C 0.25
    rho_x = 25 / (25 + 1.25 ** 2)
    d = decompose(rho_x, 0.75, 0.25)
    assert sum(d.as_dict().values()) == pytest.approx(1)
    assert d.reachable == pytest.approx(ceiling(rho_x, 0.75, 0.25))
    assert round(d.benchmark_noise, 3) == 0.059
    assert round(d.uncovered, 3) == 0.529
    assert round(disattenuate(0.40, rho_x, 0.75), 3) == 0.476
    with pytest.raises(ValueError):
        ceiling(1.2, 0.5, 0.5)


def test_coverage_recovers_true_correlation():
    rng = np.random.default_rng(5)
    n_m, rt = 300, 0.7
    a = rng.normal(0, 1, n_m)
    b = rt * a + np.sqrt(1 - rt ** 2) * rng.normal(0, 1, n_m)
    X = (rng.random((n_m, 200)) < 1 / (1 + np.exp(-(a[:, None] - rng.normal(0, 1, 200))))).astype(int)
    Y = (rng.random((n_m, 150)) < 1 / (1 + np.exp(-(b[:, None] - rng.normal(0, 1, 150))))).astype(int)
    c = estimate_coverage(X, Y, n_boot=200)
    # logit to proportion is not linear, so the score-scale correlation sits a little below 0.7
    assert 0.6 < c.r_true < 0.75
    assert c.ci_low < c.coverage < c.ci_high
    assert c.r_observed < c.r_true


def test_split_correlation_near_one_for_unidimensional_data():
    rng = np.random.default_rng(6)
    theta = rng.normal(0, 1, 200)
    X = (rng.random((200, 400)) < 1 / (1 + np.exp(-(theta[:, None] - rng.normal(0, 1, 400))))).astype(int)
    s = split_correlation(X, np.arange(400) < 200)
    assert s["r_true"] == pytest.approx(1, abs=0.05)


def test_no_dif_gives_variance_ratio_near_one_and_small_swing():
    rng = np.random.default_rng(7)
    theta = rng.normal(0, 1, 300)
    group = np.arange(300) < 150
    theta[group] += 0.5  # impact, not DIF
    b = rng.normal(0, 1, 500)
    X = (rng.random((300, 500)) < 1 / (1 + np.exp(-(theta[:, None] - b)))).astype(int)
    e = excess_gap_variance(X, group)
    assert 0.8 < e["var_ratio"] < 1.4
    # plant DIF on 100 tasks and the ratio and held-out swing grow
    Xd = X.copy()
    Xd[np.ix_(group, np.arange(100))] = (rng.random((150, 100)) < 0.95).astype(int)
    assert excess_gap_variance(Xd, group)["var_ratio"] > 3 * e["var_ratio"]
    assert (heldout_subset_swing(Xd, group, 50)["mean_swing"]
            > 3 * heldout_subset_swing(X, group, 50)["mean_swing"])


def test_score_bands_remove_ability_inflation():
    rng = np.random.default_rng(8)
    theta = rng.normal(0, 1, 400)
    group = np.arange(400) < 200
    theta[group] += 1.5  # large impact, no DIF
    b = rng.normal(0, 1.5, 400)
    X = (rng.random((400, 400)) < 1 / (1 + np.exp(-(theta[:, None] - b)))).astype(int)
    raw = excess_gap_variance(X, group)["var_ratio"]
    banded = excess_gap_variance(X, group, n_strata=5)["var_ratio"]
    assert raw > 1.5 and 0.7 < banded < 1.3
