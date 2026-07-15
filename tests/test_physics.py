"""Analytic checks of the spectral-density and rate expressions."""

import numpy as np
import pytest

from mdrelax import spectral_density as SD
from mdrelax import rates as R
from mdrelax.constants import CHI_Q


def test_lipari_szabo_rigid_limit():
    # S2 = 1 -> J(0) = (2/5) tau_c ; J(w) = (2/5) tau_c / (1 + (w tau_c)^2)
    tau_c_ps = 8000.0
    tau_c = tau_c_ps * 1e-12
    assert SD.J_lipari_szabo(0.0, 1.0, 50.0, tau_c_ps) == pytest.approx(
        (2 / 5) * tau_c, rel=1e-10)
    w = 2 * np.pi * 60e6
    assert SD.J_lipari_szabo(w, 1.0, 50.0, tau_c_ps) == pytest.approx(
        (2 / 5) * tau_c / (1 + (w * tau_c) ** 2), rel=1e-10)


def test_lipari_szabo_reduces_to_single_lorentzian_when_taue_zero():
    # tau_e -> 0 makes tau' -> 0, second term vanishes
    tau_c_ps = 5000.0
    j_full = SD.J_lipari_szabo(1e8, 0.8, 1e-6, tau_c_ps)
    j_rigid = SD.J_lipari_szabo(1e8, 1.0, 1e-6, tau_c_ps)
    assert j_full == pytest.approx(0.8 * j_rigid, rel=1e-4)


def test_multiexp_matches_lipari_szabo_single_mode():
    # J_multiexp with one internal mode == LS with same S2, tau_e, tau_c
    S2, tau_e_ps, tau_c_ps = 0.85, 40.0, 9000.0
    tau_c = tau_c_ps * 1e-12
    tau_e = tau_e_ps * 1e-12
    tau_red = 1.0 / (1.0 / tau_c + 1.0 / tau_e)
    w = np.array([1.0 - S2])
    omega = np.array([0.0, 1e8, 5e8])
    j_me = SD.J_multiexp(omega, S2, tau_c, np.array([tau_red]), w)
    j_ls = np.array([SD.J_lipari_szabo(o, S2, tau_e_ps, tau_c_ps) for o in omega])
    assert np.allclose(j_me, j_ls, rtol=1e-10)


def test_deuterium_rate_coefficients():
    # With a flat spectral density J0 = JwD = J2wD = J, the ABSURDer coefficients
    # give known combinations.
    J = 1e-9
    R_Dz, R_Dy, R_3Dz2 = R.deuterium_rates(J, J, J, CHI_Q)
    pre2 = CHI_Q ** 2
    assert R_Dz == pytest.approx(3 / 16 * pre2 * 5 * J, rel=1e-12)
    assert R_Dy == pytest.approx(1 / 32 * pre2 * 30 * J, rel=1e-12)
    assert R_3Dz2 == pytest.approx(3 / 16 * pre2 * 3 * J, rel=1e-12)


def test_nh_rates_physical_magnitudes():
    # rigid amide, tau_c = 8 ns, 600 MHz -> R1 ~ 1-2, R2 ~ 10-20, NOE < 0.85
    tau_c_ps = 8000.0
    J = lambda w: SD.J_lipari_szabo(w, 0.85, 50.0, tau_c_ps)
    R1, R2, noe = R.nh_rates_from_J(J, 600.0)
    assert 0.5 < R1 < 3.0
    assert 8.0 < R2 < 25.0
    assert 0.6 < noe < 0.9
    assert R2 > R1


def test_nh_r2_grows_with_tau_c():
    # slower tumbling -> larger R2
    r2_fast = R.nh_rates_from_J(
        lambda w: SD.J_lipari_szabo(w, 0.9, 50.0, 4000.0), 600.0)[1]
    r2_slow = R.nh_rates_from_J(
        lambda w: SD.J_lipari_szabo(w, 0.9, 50.0, 12000.0), 600.0)[1]
    assert r2_slow > r2_fast
