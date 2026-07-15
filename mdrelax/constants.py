"""Physical constants and derived NMR coupling prefactors.

All values in SI units unless noted. Gyromagnetic ratios in rad s^-1 T^-1.
"""

import numpy as np

# ── Fundamental constants ──────────────────────────────────────────────────
MU_0 = 4.0 * np.pi * 1e-7        # vacuum permeability (T m A^-1)
HBAR = 1.054571817e-34           # reduced Planck constant (J s)

# ── Gyromagnetic ratios (rad s^-1 T^-1) ────────────────────────────────────
GAMMA_H = 2.6752218744e8         # 1H
GAMMA_N = -2.71261804e7          # 15N
GAMMA_C = 6.728284e7             # 13C
GAMMA_D = 4.10662791e7           # 2H

# ── Backbone amide 15N-1H ──────────────────────────────────────────────────
R_NH = 1.02e-10                  # N-H bond length (m)
CSA_N = -160e-6                  # 15N chemical-shift anisotropy (dimensionless)

# 15N-1H dipolar coupling constant  d = (mu0/4pi) * gH gN hbar / r_NH^3  (rad/s)
DIP_NH = (MU_0 / (4.0 * np.pi)) * GAMMA_H * abs(GAMMA_N) * HBAR / R_NH**3
DIP_NH2 = DIP_NH**2

# ── Methyl 2H quadrupolar ──────────────────────────────────────────────────
# Quadrupolar coupling constant chi_Q = e^2 q Q / hbar ~ 2*pi*167 kHz for a
# tetrahedral C-2H bond (Millet et al. 2002; Hoffmann et al. 2018 / ABSURDer).
CHI_Q_HZ = 167e3
CHI_Q = 2.0 * np.pi * CHI_Q_HZ   # rad/s


def larmor_H(field_MHz):
    """1H angular Larmor frequency (rad/s) for a given field in MHz (1H)."""
    return 2.0 * np.pi * field_MHz * 1e6


def larmor_N(field_MHz):
    """15N angular Larmor frequency (rad/s) at a given 1H field (MHz)."""
    return larmor_H(field_MHz) * abs(GAMMA_N) / GAMMA_H


def larmor_D(field_MHz):
    """2H angular Larmor frequency (rad/s) at a given 1H field (MHz)."""
    return larmor_H(field_MHz) * GAMMA_D / GAMMA_H


def csa_prefactor_N(field_MHz):
    """15N CSA relaxation prefactor c^2 = (1/3)(omega_N * CSA)^2 (rad^2/s^2)."""
    omega_N = larmor_N(field_MHz)
    return (1.0 / 3.0) * (omega_N * CSA_N) ** 2
