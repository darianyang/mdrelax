"""Spectral density functions J(omega).

Two families are provided:

* Model-free (Lipari-Szabo, Extended Model-Free), used for backbone NH.  These
  take internal parameters (S2, tau_e) plus an overall tumbling time tau_c.
* Multi-exponential mapping (:func:`J_multiexp`), used for methyl 2H to match
  the ABSURDer spectral-density-mapping approach.

Frequency ``omega`` is angular (rad/s).  All correlation times are in **seconds**
for :func:`J_multiexp` and the aniso helpers, but the model-free convenience
functions accept picoseconds for readability (they are what the NH fitters use).
"""

import numpy as np


def _tau_prime(tau_c_s, tau_e_s):
    """Effective time: 1/tau' = 1/tau_c + 1/tau_e (seconds)."""
    return 1.0 / (1.0 / tau_c_s + 1.0 / tau_e_s)


# ── Model-free (backbone NH), times in picoseconds ─────────────────────────

def J_lipari_szabo(omega, S2, tau_e_ps, tau_c_ps):
    """Lipari-Szabo spectral density (Lipari & Szabo 1982)."""
    tau_c = tau_c_ps * 1e-12
    tau_e = tau_e_ps * 1e-12
    tp = _tau_prime(tau_c, tau_e)
    return (2.0 / 5.0) * (S2 * tau_c / (1 + (omega * tau_c) ** 2)
                          + (1 - S2) * tp / (1 + (omega * tp) ** 2))


def J_extended_mf(omega, S2f, S2s, tau_f_ps, tau_s_ps, tau_c_ps):
    """Extended Model-Free spectral density (Clore et al. 1990)."""
    tau_c = tau_c_ps * 1e-12
    tau_f = tau_f_ps * 1e-12
    tau_s = tau_s_ps * 1e-12
    S2 = S2f * S2s
    tpf = _tau_prime(tau_c, tau_f)
    tps = _tau_prime(tau_c, tau_s)
    return (2.0 / 5.0) * (S2 * tau_c / (1 + (omega * tau_c) ** 2)
                          + (1 - S2f) * tpf / (1 + (omega * tpf) ** 2)
                          + S2f * (1 - S2s) * tps / (1 + (omega * tps) ** 2))


# ── Axially symmetric anisotropic tumbling ─────────────────────────────────

def aniso_taus(D_par_s, D_perp_s):
    """Three tumbling times for an axially symmetric diffusion tensor (s)."""
    return (1.0 / (6.0 * D_perp_s),
            1.0 / (D_par_s + 5.0 * D_perp_s),
            1.0 / (4.0 * D_par_s + 2.0 * D_perp_s))


def aniso_weights(theta_rad):
    """Geometric weights A0, A1, A2 (Woessner 1962)."""
    ct = np.cos(theta_rad)
    st = np.sin(theta_rad)
    return ((3 * ct**2 - 1) ** 2 / 4.0,
            3 * st**2 * ct**2,
            (3.0 / 4.0) * st**4)


def J_aniso_lipari_szabo(omega, S2, tau_e_ps, D_par_s, D_perp_s, theta_rad):
    """Anisotropic Lipari-Szabo spectral density."""
    tau_e = tau_e_ps * 1e-12
    J = 0.0
    for Am, tau_m in zip(aniso_weights(theta_rad), aniso_taus(D_par_s, D_perp_s)):
        tp = _tau_prime(tau_m, tau_e)
        J += Am * (S2 * tau_m / (1 + (omega * tau_m) ** 2)
                   + (1 - S2) * tp / (1 + (omega * tp) ** 2))
    return (2.0 / 5.0) * J


# ── Multi-exponential mapping (methyl 2H, ABSURDer) ────────────────────────

def J_multiexp(omega, S2, tau_R_s, tau_red_s, weights):
    """Multi-exponential spectral density used for methyl 2H relaxation.

    J(w) = 2/5 [ S2 tauR / (1 + (w tauR)^2)
                 + sum_i w_i tau_red,i / (1 + (w tau_red,i)^2) ]

    This is the ABSURDer form (Hoffmann et al. 2018; Kuemmerer et al.), where
    ``tau_red,i`` are the internal times reduced by the overall tumbling and
    ``weights`` are the fitted amplitudes.  All times in seconds.

    Parameters
    ----------
    omega : float or ndarray   angular frequency (rad/s)
    S2 : float                 methyl-axis order parameter (long-time plateau)
    tau_R_s : float            overall (methyl-specific) tumbling time (s)
    tau_red_s : ndarray        reduced internal times (s)
    weights : ndarray          internal amplitudes (same length as tau_red_s)
    """
    omega = np.asarray(omega, dtype=np.float64)
    tmp = S2 * tau_R_s / (1.0 + (omega * tau_R_s) ** 2)
    tau_red_s = np.asarray(tau_red_s)
    weights = np.asarray(weights)
    for w_i, t_i in zip(weights, tau_red_s):
        tmp = tmp + w_i * t_i / (1.0 + (omega * t_i) ** 2)
    return (2.0 / 5.0) * tmp
