"""Correlation-function fitting.

* Model-free fits of the backbone NH internal ACF: Lipari-Szabo and Extended
  Model-Free (:func:`fit_lipari_szabo`, :func:`fit_extended_mf`).
* The ABSURDer methyl internal fit: exponentially spaced time sampling,
  long-time S2 from the tail, and a 6-exponential decomposition of the internal
  correlation function (:func:`exp_time_points`, :func:`estimate_S2_tail`,
  :func:`fit_internal_multiexp`, :func:`reduced_times_and_weights`).
"""

import numpy as np
from scipy.optimize import curve_fit, least_squares


# ── Backbone NH model-free fits (internal ACF, times in ps) ────────────────

def fit_lipari_szabo(acf, t_ps, tau_c_ps, S2_max=1.0):
    """Fit an internal ACF to C(t) = S2 + (1-S2) exp(-t/tau_e)."""
    def model(t, S2, tau_e):
        return S2 + (1.0 - S2) * np.exp(-t / tau_e)
    try:
        popt, _ = curve_fit(model, t_ps, acf, p0=[S2_max * 0.85, 50.0],
                            bounds=([0.0, 0.1], [S2_max, max(tau_c_ps * 2, 1.0)]),
                            maxfev=5000)
        S2, tau_e = float(np.clip(popt[0], 0.0, S2_max)), float(popt[1])
        rmse = float(np.sqrt(np.mean((model(t_ps, S2, tau_e) - acf) ** 2)))
        return dict(S2=S2, tau_e_ps=tau_e, rmse=rmse, converged=True)
    except (RuntimeError, ValueError):
        return dict(S2=np.nan, tau_e_ps=np.nan, rmse=np.nan, converged=False)


def fit_extended_mf(acf, t_ps, tau_c_ps, S2_max=1.0):
    """Fit an internal ACF to the Extended Model-Free form."""
    def model(t, S2f, S2s, tau_f, tau_s):
        S2 = S2f * S2s
        return S2 + S2f * (1 - S2s) * np.exp(-t / tau_s) \
            + (1 - S2f) * np.exp(-t / tau_f)
    try:
        popt, _ = curve_fit(model, t_ps, acf, p0=[0.95, 0.90, 5.0, 500.0],
                            bounds=([0.0, 0.0, 0.1, 1.0],
                                    [1.0, S2_max, max(tau_c_ps * 0.1, 1.0),
                                     max(tau_c_ps * 2, 2.0)]),
                            maxfev=10000)
        S2f, S2s, tau_f, tau_s = [float(x) for x in popt]
        S2 = float(np.clip(S2f * S2s, 0.0, S2_max))
        rmse = float(np.sqrt(np.mean((model(t_ps, *popt) - acf) ** 2)))
        return dict(S2f=S2f, S2s=S2s, S2=S2, tau_f_ps=tau_f, tau_s_ps=tau_s,
                    rmse=rmse, converged=True)
    except (RuntimeError, ValueError):
        return dict(S2f=np.nan, S2s=np.nan, S2=np.nan, tau_f_ps=np.nan,
                    tau_s_ps=np.nan, rmse=np.nan, converged=False)


# ── ABSURDer methyl internal fit ───────────────────────────────────────────

def exp_time_points(simlength, fit_length, accuracy):
    """Exponentially spaced integer time points below ``fit_length``.

    Reproduces the ABSURDer sampling: points ~exp(i/accuracy) up to simlength,
    kept where they fall below fit_length, de-duplicated.
    """
    maxlogtime = int(accuracy * np.round(np.log(simlength)))
    tmp = np.exp(np.linspace(1, maxlogtime, maxlogtime) / accuracy)
    exp_t = np.unique([int(np.round(v)) for v in tmp if v < fit_length])
    return exp_t


def find_nearest(array, value):
    array = np.asarray(array)
    return array[int((np.abs(array - value)).argmin())]


def estimate_S2_tail(exp_t, ct, fit_length, ct_lim):
    """Long-time plateau S2 as the mean of C(t) over the last 1/ct_lim fraction."""
    thr = find_nearest(exp_t, fit_length / ct_lim)
    idx = int(np.where(exp_t == thr)[0][0])
    return float(np.round(np.mean(ct[idx:]), decimals=5))


def fit_internal_multiexp(times, ct, S2, n_free=5, p0=None, bounds=None):
    """Fit the internal methyl C(t) to a (n_free+1)-exponential decomposition.

    Model (n_free = 5, matching ABSURDer):
        C(t) = S2 + sum_{i=0..4} A_i exp(-t/tau_i)
                  + (1 - S2 - sum A_i) exp(-t/tau_5)

    Parameters
    ----------
    times : ndarray   time points (ps)
    ct : ndarray      correlation values at ``times``
    S2 : float        long-time plateau (fixed)
    n_free : int      number of free amplitudes (default 5)
    p0, bounds : optional overrides for the initial guess / bounds

    Returns
    -------
    dict with keys: amplitudes (n_free+1,), taus_ps (n_free+1,), success (bool),
        cost (float), result (scipy OptimizeResult)
    """
    n_tau = n_free + 1
    if p0 is None:
        p0 = [0.5] * n_free + [1.0] * n_tau
    if bounds is None:
        lo = [0.0] * n_free + [0.0] * n_tau
        hi = [1.0] * n_free + [np.inf] * n_tau
        bounds = (lo, hi)

    def model(p, x):
        out = np.full_like(x, S2, dtype=np.float64)
        amp_sum = 0.0
        for i in range(n_free):
            out = out + p[i] * np.exp(-x / p[n_free + i])
            amp_sum += p[i]
        out = out + (1.0 - S2 - amp_sum) * np.exp(-x / p[n_free + n_free])
        return out

    def resid(p, x, y):
        return model(p, x) - y

    res = least_squares(resid, p0, bounds=bounds, args=(times, ct))
    amps_free = res.x[:n_free]
    taus = res.x[n_free:]
    w_last = 1.0 - S2 - float(np.sum(amps_free))
    amplitudes = np.append(amps_free, w_last)
    return dict(amplitudes=amplitudes, taus_ps=taus, success=res.success,
                cost=float(res.cost), result=res)


def reduced_times_and_weights(fit, tau_R_s):
    """Combine the internal fit with an overall tumbling time tau_R.

    Returns
    -------
    (tau_red_s, weights) : reduced internal times (s) and their amplitudes.
    """
    taus_s = np.asarray(fit['taus_ps']) * 1e-12
    tau_red_s = 1.0 / (1.0 / tau_R_s + 1.0 / taus_s)
    return tau_red_s, np.asarray(fit['amplitudes'])
