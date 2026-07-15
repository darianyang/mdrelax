"""Relaxation-rate expressions from spectral-density values.

Backbone 15N NH: R1, R2, heteronuclear NOE (dipolar + CSA).
Side-chain methyl 2H: the three experimentally measured quadrupolar rates
R(Dz), R(Dy), R(3Dz^2-2), matching the ABSURDer convention where the 2/5
prefactor already lives inside J.
"""

import numpy as np

from .constants import (DIP_NH2, GAMMA_H, GAMMA_N, CHI_Q,
                        larmor_H, larmor_N, larmor_D, csa_prefactor_N)


# ── Backbone 15N NH ────────────────────────────────────────────────────────

def nh_rates_from_J(J, field_MHz):
    """R1, R2, hetNOE from a spectral-density callable ``J(omega)``.

    ``J`` must return the *reduced* spectral density (the (2/5)-normalized form
    produced by :mod:`mdrelax.spectral_density`).

    Parameters
    ----------
    J : callable            J(omega_rad_s) -> spectral density (s)
    field_MHz : float       1H Larmor frequency (MHz)

    Returns
    -------
    (R1, R2, NOE) : tuple of float   R1, R2 in s^-1, NOE dimensionless
    """
    omega_H = larmor_H(field_MHz)
    omega_N = larmor_N(field_MHz)
    c2 = csa_prefactor_N(field_MHz)

    J0 = J(0.0)
    JwN = J(omega_N)
    JwH = J(omega_H)
    JwHmN = J(omega_H - omega_N)
    JwHpN = J(omega_H + omega_N)

    R1 = (DIP_NH2 / 4.0) * (JwHmN + 3 * JwN + 6 * JwHpN) + c2 * JwN
    R2 = (DIP_NH2 / 8.0) * (4 * J0 + JwHmN + 3 * JwN + 6 * JwH + 6 * JwHpN) \
        + (c2 / 6.0) * (4 * J0 + 3 * JwN)
    NOE = 1.0 + (GAMMA_H / GAMMA_N) * (DIP_NH2 / 4.0) * (6 * JwHpN - JwHmN) / R1
    return R1, R2, NOE


# ── Side-chain methyl 2H ───────────────────────────────────────────────────

def deuterium_rates(J0, JwD, J2wD, chi_q=CHI_Q):
    """Methyl 2H quadrupolar relaxation rates (ABSURDer convention).

    These are the three deuterium single-spin relaxation rates measured
    experimentally for methyl groups (Millet/Kay; Tugarinov):

    R_Dz    : longitudinal decay of Dz       (3/16) chi^2 (J(wD) + 4 J(2wD))
    R_Dy    : transverse / R_Q               (1/32) chi^2 (9 J0 + 15 J(wD) + 6 J(2wD))
    R_3Dz2  : quadrupolar order 3Dz^2-2      (3/16) chi^2 (3 J(wD))

    ``J*`` are values of the reduced multi-exponential spectral density
    (:func:`mdrelax.spectral_density.J_multiexp`) at 0, wD and 2wD.

    Returns
    -------
    (R_Dz, R_Dy, R_3Dz2) : tuple of float (s^-1)
    """
    pre2 = chi_q ** 2
    R_Dz = 3.0 / 16.0 * pre2 * (JwD + 4 * J2wD)
    R_Dy = 1.0 / 32.0 * pre2 * (9 * J0 + 15 * JwD + 6 * J2wD)
    R_3Dz2 = 3.0 / 16.0 * pre2 * (3 * JwD)
    return R_Dz, R_Dy, R_3Dz2


def methyl_omega_D(field_MHz):
    """2H angular Larmor frequency (rad/s) at a given 1H field (MHz)."""
    return larmor_D(field_MHz)
