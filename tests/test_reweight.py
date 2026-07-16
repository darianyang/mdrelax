"""Maximum-entropy block reweighting (:mod:`mdrelax.reweight`)."""

import numpy as np
import pytest

from mdrelax.reweight import Reweighter, select_theta
from conftest import CH3_EXP, needs_ch3_exp


# ── unit-level: weights, entropy, gradient ─────────────────────────────────

def _toy(seed=0, n_obs=6, n_blocks=40):
    rng = np.random.default_rng(seed)
    calc = rng.normal(10.0, 2.0, size=(n_obs, n_blocks))
    exp = calc.mean(axis=1) + rng.normal(0.0, 0.3, size=n_obs)
    err = np.full(n_obs, 0.5)
    return Reweighter(calc, exp, err)


def test_prior_is_softmax_zero():
    rw = _toy()
    w = rw.weights(np.zeros(rw.n_blocks))
    assert np.allclose(w, rw.w0)
    assert w.sum() == pytest.approx(1.0)


def test_weights_normalised_and_positive():
    rw = _toy()
    w = rw.weights(np.linspace(-3, 3, rw.n_blocks))
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w > 0)


def test_entropy_zero_at_prior_and_negative_elsewhere():
    rw = _toy()
    assert rw.entropy(rw.w0) == pytest.approx(0.0, abs=1e-12)
    w = rw.weights(np.linspace(-3, 3, rw.n_blocks))
    assert rw.entropy(w) < 0.0


def test_analytic_gradient_matches_finite_difference():
    rw = _toy(seed=3)
    rng = np.random.default_rng(7)
    g = rng.normal(0.0, 0.5, size=rw.n_blocks)
    theta = 0.2
    _, grad = rw.objective(g, theta)
    eps = 1e-6
    num = np.zeros_like(g)
    for i in range(rw.n_blocks):
        gp, gm = g.copy(), g.copy()
        gp[i] += eps
        gm[i] -= eps
        num[i] = (rw.objective(gp, theta)[0] - rw.objective(gm, theta)[0]) / (2 * eps)
    assert np.allclose(grad, num, atol=1e-6, rtol=1e-5)


# ── optimisation behaviour ─────────────────────────────────────────────────

def test_reweighting_reduces_chi2():
    rw = _toy(seed=1)
    res, _ = rw.optimize(theta=0.01)
    assert res.success
    assert res.chi2_red < res.chi2_red_prior
    assert 0.0 < res.phi_eff <= 1.0 + 1e-9


def test_large_theta_stays_near_prior():
    rw = _toy(seed=2)
    strong, _ = rw.optimize(theta=1e7)
    assert strong.phi_eff == pytest.approx(1.0, abs=1e-3)
    assert strong.chi2_red == pytest.approx(strong.chi2_red_prior, rel=1e-3)


def test_small_theta_fits_harder_than_large_theta():
    rw = _toy(seed=4)
    weak, _ = rw.optimize(theta=1e-3)
    strong, _ = rw.optimize(theta=1e2)
    assert weak.chi2_red <= strong.chi2_red + 1e-9
    assert weak.phi_eff <= strong.phi_eff + 1e-9


def test_recovers_matching_block():
    # One block reproduces experiment exactly; the rest are far off.  With weak
    # regularisation the reweighter should concentrate weight on that block and
    # drive chi2 to ~0.
    exp = np.array([1.0, 2.0, 3.0])
    err = np.full(3, 0.1)
    good = exp.copy()
    calc = np.column_stack([good] + [good + 5 * (j + 1) for j in range(9)])
    rw = Reweighter(calc, exp, err)
    res, _ = rw.optimize(theta=1e-4)
    assert res.chi2_red < 1e-2
    assert res.weights[0] > 0.9


def test_scan_monotonic_and_knee_between_extremes():
    rw = _toy(seed=5)
    results = rw.scan(thetas=np.logspace(2, -3, 24))
    phi = np.array([r.phi_eff for r in results])       # large->small theta
    chi = np.array([r.chi2_red for r in results])
    # relaxing theta trades entropy for fit: phi_eff falls, chi2 falls
    assert np.all(np.diff(phi) <= 1e-6)
    assert np.all(np.diff(chi) <= 1e-6)
    knee = select_theta(results)
    assert results[-1].theta <= knee.theta <= results[0].theta
    assert knee.chi2_red <= results[0].chi2_red


# ── against the real ch3 experimental data ─────────────────────────────────

@needs_ch3_exp
def test_ch3_experimental_reweighting_improves_fit():
    calc = np.load(CH3_EXP / "md.npy")                 # (3, 73, n_blocks)
    exp = np.load(CH3_EXP / "nmr_rates.npy")           # (3, 73)
    err = np.load(CH3_EXP / "nmr_errors.npy")          # (3, 73)
    n_rates, n_methyls, n_blocks = calc.shape
    rw = Reweighter(calc.reshape(n_rates * n_methyls, n_blocks),
                    exp.reshape(-1), err.reshape(-1))
    # ABSURDer reports theta ~ 1400 optimal with phi_eff ~ 0.2 on this data;
    # the total-chi2 convention must reproduce that (a reduced-chi2 loss would
    # need theta ~ 1400/m and put phi_eff near 1 here instead).
    res, _ = rw.optimize(theta=1400.0)
    assert res.success
    assert res.chi2_red < res.chi2_red_prior     # data agreement improved
    assert 0.15 < res.phi_eff < 0.25             # matches the ABSURDer optimum
    assert res.weights.shape == (n_blocks,)
