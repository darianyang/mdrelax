"""Bayesian / Maximum-Entropy reweighting of simulation blocks.

Given a set of observables computed on independent blocks of a trajectory
(here: methyl 2H relaxation rates per block) together with experimental values
and their uncertainties, find a new set of block weights that improves agreement
with experiment while staying as close as possible to the prior (force-field)
weights.  This is the reweighting scheme used by ABSURDer / BME
(Bottaro, Bengtsen & Lindorff-Larsen, Methods Mol. Biol. 2112:219, 2020).

The functional minimised is

    L(w) = 1/2 chi2(w)  -  theta * S_rel(w)

with, over blocks ``b`` and the ``m`` observables ``i`` (a rate at one methyl),

    <r_i>  = sum_b w_b r_ib                          (reweighted average)
    chi2   = sum_i ((<r_i> - exp_i) / err_i)^2       (total, summed over all m)
    S_rel  = - sum_b w_b ln(w_b / w0_b)              (relative entropy, <= 0)

``chi2`` here is the *total* chi-square summed over all ``m`` measurements, not
the reduced ``chi2/m`` -- equivalently ``L = (m/2) chi2_red - theta S_rel``.
This is the BME / ABSURDer convention, and it is what makes ``theta`` comparable
to published values (e.g. ABSURDer's optimum ``theta ~ 1400``, ``phi_eff ~ 0.2``,
reproduced on ``data/ch3/experimental``): normalising by ``m`` instead would
rescale ``theta`` by a factor of ``m``.  ``chi2_red = chi2 / m`` is still
reported as a per-measurement diagnostic.

``theta`` sets the balance: ``theta -> infinity`` pins the result to the prior
``w0``; ``theta -> 0`` fits the data as hard as the ensemble allows (over-fitting).

Rather than optimise ``w`` under the simplex constraint directly, we optimise an
unconstrained vector ``g`` and map it through a prior-anchored softmax

    w_b = w0_b exp(g_b) / sum_c w0_c exp(g_c),

so ``g = 0`` *is* the prior and normalisation/positivity hold automatically.
The gradient of ``L`` w.r.t. ``g`` is available in closed form (see
:meth:`Reweighter.objective`), which makes L-BFGS-B fast and robust even with
thousands of blocks.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class ReweightResult:
    """Outcome of a single reweighting optimisation.

    Attributes
    ----------
    theta : float
        Entropy weight used.
    weights : ndarray, shape (n_blocks,)
        Optimised, normalised block weights.
    avg : ndarray, shape (n_obs,)
        Reweighted observable averages ``sum_b w_b r_ib``.
    chi2_red, chi2_red_prior : float
        Reduced chi-square after / before reweighting.
    phi_eff : float
        Effective fraction of blocks retained, ``exp(S_rel)`` in ``(0, 1]``.
    neff : float
        Kish effective sample size ``1 / sum_b w_b^2`` (an alternative measure).
    s_rel : float
        Relative entropy ``S_rel`` (<= 0).
    success : bool
        Optimiser convergence flag.
    """

    theta: float
    weights: np.ndarray
    avg: np.ndarray
    chi2_red: float
    chi2_red_prior: float
    phi_eff: float
    neff: float
    s_rel: float
    success: bool = True


class Reweighter:
    """Maximum-entropy block reweighter.

    Parameters
    ----------
    calc : array_like, shape (n_obs, n_blocks)
        Simulated observable ``r_ib`` for observable ``i`` on block ``b``.
        Multi-dimensional observables (e.g. rate x methyl) should be flattened
        to a single leading axis; reshape the results back afterwards.
    exp : array_like, shape (n_obs,)
        Experimental values.
    err : array_like, shape (n_obs,)
        Experimental uncertainties (standard deviations, > 0).
    w0 : array_like, shape (n_blocks,), optional
        Prior block weights.  Defaults to uniform ``1 / n_blocks``.  Need not be
        normalised; it is normalised on input.
    """

    def __init__(self, calc, exp, err, w0=None):
        self.calc = np.asarray(calc, dtype=float)
        if self.calc.ndim != 2:
            raise ValueError("calc must be 2-D (n_obs, n_blocks); flatten first")
        self.exp = np.asarray(exp, dtype=float).ravel()
        self.err = np.asarray(err, dtype=float).ravel()
        self.n_obs, self.n_blocks = self.calc.shape
        if self.exp.shape != (self.n_obs,) or self.err.shape != (self.n_obs,):
            raise ValueError("exp and err must have length n_obs = calc.shape[0]")
        if np.any(self.err <= 0):
            raise ValueError("all err must be positive")

        if w0 is None:
            w0 = np.full(self.n_blocks, 1.0 / self.n_blocks)
        w0 = np.asarray(w0, dtype=float).ravel()
        if w0.shape != (self.n_blocks,):
            raise ValueError("w0 must have length n_blocks = calc.shape[1]")
        if np.any(w0 < 0):
            raise ValueError("prior weights w0 must be non-negative")
        self.w0 = w0 / w0.sum()
        self._logw0 = np.log(np.where(self.w0 > 0, self.w0, 1.0))
        self.inv_err2 = 1.0 / self.err ** 2

    # -- core quantities ----------------------------------------------------

    def weights(self, g):
        """Prior-anchored softmax ``w_b = w0_b e^{g_b} / Z`` of logits ``g``."""
        g = np.asarray(g, dtype=float)
        z = g - g.max()                       # stabilise the exponential
        w = self.w0 * np.exp(z)
        return w / w.sum()

    def chi2(self, weights, reduced=True):
        """(Reduced) chi-square of the weighted averages against experiment."""
        avg = self.calc @ weights
        chi2 = np.sum((avg - self.exp) ** 2 * self.inv_err2)
        return chi2 / self.n_obs if reduced else chi2

    def entropy(self, weights):
        """Relative entropy ``S_rel = -sum w ln(w/w0)`` (<= 0)."""
        w = weights
        mask = w > 0
        return -np.sum(w[mask] * (np.log(w[mask]) - self._logw0[mask]))

    def objective(self, g, theta):
        """Return ``(L, dL/dg)`` for logits ``g`` at entropy weight ``theta``.

        ``L = 1/2 chi2 - theta S_rel``, with ``chi2`` the total (summed over all
        ``m`` observables).  With ``w_b`` the softmax of ``g``,

            d chi2  / dg_a = w_a (c_a - <c>_w),   c_b = 2 sum_i r_ib s_i,
            d S_rel / dg_a = -w_a (ln(w_a/w0_a) + S_rel),

        where ``s_i = (<r_i> - exp_i) / err_i^2`` and ``<c>_w = sum_b w_b c_b``.
        """
        w = self.weights(g)
        avg = self.calc @ w
        resid = avg - self.exp
        chi2 = np.sum(resid ** 2 * self.inv_err2)       # total, over all m obs
        s_rel = self.entropy(w)
        L = 0.5 * chi2 - theta * s_rel

        # chi2 gradient
        s = resid * self.inv_err2
        c = 2.0 * (self.calc.T @ s)                     # d chi2 / d w_b
        grad_chi2 = w * (c - w @ c)
        # entropy gradient
        lnratio = np.where(w > 0, np.log(np.where(w > 0, w, 1.0)) - self._logw0, 0.0)
        grad_s = -w * (lnratio + s_rel)
        grad = 0.5 * grad_chi2 - theta * grad_s
        return L, grad

    # -- optimisation -------------------------------------------------------

    def optimize(self, theta, g0=None, maxiter=5000):
        """Minimise ``L`` at a single ``theta``; return a :class:`ReweightResult`."""
        if g0 is None:
            g0 = np.zeros(self.n_blocks)          # start at the prior
        res = minimize(self.objective, g0, args=(theta,), jac=True,
                       method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": 10 * maxiter,
                                "ftol": 1e-12, "gtol": 1e-8})
        w = self.weights(res.x)
        s_rel = self.entropy(w)
        return ReweightResult(
            theta=float(theta),
            weights=w,
            avg=self.calc @ w,
            chi2_red=self.chi2(w),
            chi2_red_prior=self.chi2(self.w0),
            phi_eff=float(np.exp(s_rel)),
            neff=float(1.0 / np.sum(w ** 2)),
            s_rel=float(s_rel),
            success=bool(res.success),
        ), res.x

    def scan(self, thetas=None, warm_start=True):
        """Optimise across a ladder of ``theta`` for the L-curve.

        Returns a list of :class:`ReweightResult`, ordered from strongest to
        weakest regularisation (large -> small ``theta``).  With
        ``warm_start`` the solution at each ``theta`` seeds the next, which is
        both faster and smoother along the curve.
        """
        if thetas is None:
            # theta scales with the total chi2, i.e. with the number of
            # measurements m; span from near-prior down to heavy reweighting.
            thetas = np.logspace(np.log10(200 * self.n_obs),
                                 np.log10(0.02 * self.n_obs), 40)
        thetas = np.sort(np.asarray(thetas, dtype=float))[::-1]   # large -> small
        results, g = [], None
        for th in thetas:
            r, g = self.optimize(th, g0=g if warm_start else None)
            results.append(r)
        return results


def select_theta(results):
    """Pick the L-curve knee from :meth:`Reweighter.scan` output.

    The knee is the point on the ``(phi_eff, chi2_red)`` curve furthest from the
    chord joining its two endpoints -- the standard L-curve corner, the balance
    beyond which giving up more of the prior (lower ``phi_eff``) buys little
    further reduction in ``chi2_red``.  Returns the chosen :class:`ReweightResult`.
    """
    results = list(results)
    if len(results) < 3:
        return results[len(results) // 2]
    x = np.array([r.phi_eff for r in results])
    y = np.array([r.chi2_red for r in results])
    # normalise both axes to [0, 1] so the corner is scale-independent
    xn = (x - x.min()) / (np.ptp(x) or 1.0)
    yn = (y - y.min()) / (np.ptp(y) or 1.0)
    p0, p1 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    chord = p1 - p0
    length = np.hypot(*chord) or 1.0
    # perpendicular distance of each point from the endpoint chord
    dist = np.abs(chord[0] * (p0[1] - yn) - chord[1] * (p0[0] - xn)) / length
    return results[int(np.argmax(dist))]
