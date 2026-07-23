"""Overall tumbling: isotropic tau_c and the rotational diffusion tensor.

The diffusion-tensor path is a pure-Python replacement for the pdbinertia +
quadric_diffusion pipeline used by ABSURDer:

1. Per-residue backbone tumbling times tau_M are obtained by fitting the
   backbone N-H time-correlation functions (which retain overall tumbling) to
   simple and extended Lipari-Szabo models (:func:`fit_backbone_tauM`).
2. A diffusion tensor is fit to those tau_M with the quadric equation
   (:func:`fit_diffusion`), in any of three models -- ``iso``, ``axial`` or
   ``aniso``.  The axis orientation is a free parameter, so the result is
   independent of the input coordinate frame and pdbinertia is not needed.
3. Per-methyl tumbling times tau_R follow from the C-C(methyl) axis orientation
   relative to the fitted tensor (:func:`tau_R_from_fit`).

The quadric equation (Bruschweiler et al., Science 268:886, 1995; Lee et al.,
J. Biomol. NMR 9:287, 1997) relates the tumbling time of a bond vector to the
diffusion tensor via the vector's direction cosines u_j in the principal frame::

    1 / (6 tau_i) = 1/2 sum_j D_j (1 - u_ij^2)          j = x, y, z

which for an axially symmetric tensor reduces to the familiar
``Diso - P2(cos a_i) (Dpar - Dper) / 3``.

Validated against the reference programs on ubiquitin: see
``tests/test_quadric_reference.py``.
"""

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.stats import f as f_dist


# ── Isotropic tau_c (for backbone NH model-free) ───────────────────────────

def estimate_tau_c(acfs, dt_ps, fit_fraction=0.5):
    """Estimate an isotropic tau_c (ns) from averaged internal+overall ACFs.

    Fits the mean ACF to a single exponential ``A exp(-t/tau)``.  Only reliable
    when the ACF is computed on an un-aligned trajectory that still contains
    overall tumbling and the trajectory is several times longer than tau_c.
    """
    acf = np.asarray(acfs)
    if acf.ndim == 2:
        acf = acf.mean(axis=1)
    n = len(acf)
    t = np.arange(n) * dt_ps
    n_fit = max(10, int(n * fit_fraction))
    popt, _ = curve_fit(lambda t, A, tau: A * np.exp(-t / tau),
                        t[:n_fit], acf[:n_fit], p0=[1.0, 5000.0],
                        bounds=([0.1, 1.0], [2.0, 1e7]), maxfev=10000)
    return popt[1] / 1000.0


# ── Backbone tau_M from N-H TCFs ───────────────────────────────────────────

def _exp_points(simlength, fit_length, accuracy):
    maxlogtime = int(accuracy * np.round(np.log(simlength)))
    tmp = np.exp(np.linspace(1, maxlogtime, maxlogtime) / accuracy)
    return np.unique([int(np.round(v)) for v in tmp if v < fit_length])


def fit_backbone_tauM(tcf_bb, n_nh, l_block_ps, dt_ps=1.0, start_tau_ps=10000.0,
                      accuracy=100):
    """Per-residue backbone tumbling times tau_M (ps) from N-H TCFs.

    Ports the ABSURDer simple/extended Lipari-Szabo fit and model-selection
    logic.  ``tcf_bb`` is a single block array with time in column 0 and one
    N-H correlation function per subsequent column.

    ``dt_ps`` is the spacing between successive rows of ``tcf_bb``.  The fit
    points are chosen in ps (they come from ``l_block_ps``) but have to be read
    out as row indices, so the two only coincide when one row is one ps --
    ABSURDer's case, and the default here for backwards compatibility.  Passing
    the real spacing keeps the fit on the rows it actually meant to sample.

    Returns
    -------
    tauM : ndarray (n_nh,)   per-residue tumbling times (ps); NaN entries are
        replaced with the residue-set mean.
    """
    tcf_bb = np.asarray(tcf_bb)
    fit_length = int(l_block_ps / 2)
    exp_t = _exp_points(l_block_ps, fit_length, accuracy)
    # ps -> row index, de-duplicated (coarse dt maps several early points onto
    # the same row) and clipped to the rows that exist
    rows = np.unique(np.round(exp_t / dt_ps).astype(int))
    rows = rows[rows < tcf_bb.shape[0]]
    t = rows.astype(float) * dt_ps
    exp_t = rows

    delta = start_tau_ps
    lo_a, hi_a = start_tau_ps - delta, start_tau_ps + delta
    tauM = np.full(n_nh, np.nan)

    def ls_simple(x, a, b):
        return np.exp(-x / a) * b

    def ls_ext(x, a, b, c, d):
        return np.exp(-x / a) * (b + (d - b) * np.exp(-x / c))

    for j in range(n_nh):
        ct = tcf_bb[exp_t, j + 1]
        try:
            p1, _ = curve_fit(ls_simple, t, ct, p0=[start_tau_ps, 0.8],
                              bounds=([max(lo_a, 1.0), 0.0], [hi_a, 1.0]),
                              maxfev=5000)
        except Exception:
            p1 = [0.0, 0.0]
        try:
            p2, _ = curve_fit(ls_ext, t, ct, p0=[start_tau_ps, 0.8, 50.0, 1.0],
                              bounds=([max(lo_a, 1.0), 0.0, 5.0, 0.0],
                                      [hi_a, 1.0, 100.0, 1.0]), maxfev=5000)
        except Exception:
            p2 = [0.0, 0.0, 0.0, 0.0]

        # model selection (ABSURDer)
        if ((0 < p2[2] < 200) and (lo_a + 100 < p2[0] < hi_a - 100)
                and ((abs(p2[0] - start_tau_ps) > 10) or (abs(p2[1] - 0.8) > 0.1))
                and (0 < p2[1] < 1.5) and (0 < p2[3] < 1.5)):
            tauM[j] = p2[0]
        elif ((lo_a + 100 < p1[0] < hi_a - 100)
              and ((abs(p1[0] - start_tau_ps) > 10) or (abs(p1[1] - 0.8) > 0.1))
              and (0 < p1[1] < 1.5)):
            tauM[j] = p1[0]
    tauM[np.isnan(tauM)] = np.nanmean(tauM)
    return tauM


# ── Rotational diffusion tensor (quadric_diffusion replacement) ────────────

MODELS = ("iso", "axial", "aniso")
_N_PARAMS = {"iso": 1, "axial": 4, "aniso": 6}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _zyz(phi, theta, psi):
    """Rotation matrix in the z-y-z Euler convention used by quadric."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return (np.array([[cph, -sph, 0.0], [sph, cph, 0.0], [0.0, 0.0, 1.0]])
            @ np.array([[cth, 0.0, sth], [0.0, 1.0, 0.0], [-sth, 0.0, cth]])
            @ np.array([[cps, -sps, 0.0], [sps, cps, 0.0], [0.0, 0.0, 1.0]]))


def _euler_zyz(axes):
    """(phi, theta, psi) of a frame whose principal axes are the columns of `axes`."""
    theta = np.arccos(np.clip(axes[2, 2], -1.0, 1.0))
    if abs(np.sin(theta)) < 1e-8:               # gimbal lock: phi/psi degenerate
        return float(np.arctan2(axes[1, 0], axes[0, 0])), float(theta), 0.0
    return (float(np.arctan2(axes[1, 2], axes[0, 2])), float(theta),
            float(np.arctan2(axes[2, 1], -axes[2, 0])))


def _axial_D(Diso, Dratio):
    """(Dpar, Dper) from the isotropic value and the ratio Dpar/Dper."""
    return 3.0 * Diso / (1.0 + 2.0 / Dratio), 3.0 * Diso / (2.0 + Dratio)


def _aniso_D(Diso, zz_ratio, xy_ratio):
    """(Dxx, Dyy, Dzz) from Diso, 2Dzz/(Dxx+Dyy) and Dxx/Dyy."""
    s = 3.0 * Diso / (1.0 + zz_ratio / 2.0)           # Dxx + Dyy
    Dyy = s / (1.0 + xy_ratio)
    return xy_ratio * Dyy, Dyy, zz_ratio * s / 2.0


def local_D(D_principal, axes, vectors):
    """Effective local diffusion constant of each bond vector (quadric equation).

    ``1/(6 tau_i) = 1/2 sum_j D_j (1 - u_ij^2)``, with u_i the unit bond vector
    expressed in the principal diffusion frame.

    Parameters
    ----------
    D_principal : (3,)    principal values (Dxx, Dyy, Dzz), s^-1
    axes : (3, 3)         principal axes as *columns*, in the molecular frame
    vectors : (N, 3)      bond vectors in the molecular frame (need not be unit)

    Returns
    -------
    ndarray (N,)          local diffusion constant of each vector (s^-1)
    """
    u = _unit(np.atleast_2d(vectors)) @ np.asarray(axes, float)
    return 0.5 * np.sum(np.asarray(D_principal, float) * (1.0 - u ** 2), axis=1)


def inertia_principal_axes(coords, masses=None, sort=True):
    """Principal axes of the mass-weighted inertia tensor (pdbinertia).

    Parameters
    ----------
    coords : (N, 3)          atomic coordinates
    masses : (N,), optional  atomic masses; all-equal if omitted
    sort : bool              order the axes by *descending* moment, as pdbinertia
                             reports them.  ``False`` keeps ``numpy.linalg.eigh``
                             order (ascending).

    Returns
    -------
    (axes, moments, com) : the principal axes as *columns* of a (3, 3) array,
        the corresponding principal moments, and the centre of mass.

    Note the axes carry an arbitrary per-column sign, so individual columns may
    appear negated relative to pdbinertia's rotation matrix (whose *rows* are
    these axes).
    """
    coords = np.asarray(coords, dtype=float)
    if masses is None:
        masses = np.ones(len(coords))
    masses = np.asarray(masses, float)
    com = np.average(coords, axis=0, weights=masses)
    r = coords - com
    # I = sum_a m_a (|r_a|^2 delta_ij - r_ai r_aj)
    I = (np.eye(3) * np.sum(masses * np.sum(r ** 2, axis=1))
         - np.einsum("a,ai,aj->ij", masses, r, r))
    moments, axes = np.linalg.eigh(I)
    if sort:
        order = np.argsort(moments)[::-1]
        moments, axes = moments[order], axes[:, order]
    return axes, moments, com


def fit_diffusion(tauM_ps, vectors, model="axial", sigma_tauM_ps=None,
                  init_axis=None):
    """Fit a rotational diffusion tensor to per-residue tau_M (quadric).

    Pure-Python replacement for quadric_diffusion.  The tensor orientation is a
    free parameter of the fit, so -- unlike quadric -- the input structure need
    not be pre-aligned to its inertia frame by pdbinertia.

    Parameters
    ----------
    tauM_ps : (N,)               per-residue tumbling times (ps)
    vectors : (N, 3)             the corresponding bond vectors (e.g. N-H), in
                                 the molecular frame; need not be unit vectors
    model : {'iso', 'axial', 'aniso'}
        ``iso``   isotropic, 1 parameter (Diso).
        ``axial`` axially symmetric, 4 parameters (Diso, Dpar/Dper, theta, phi).
        ``aniso`` fully anisotropic, 6 parameters (Diso, 2Dzz/(Dxx+Dyy),
                  Dxx/Dyy, phi, theta, psi).
    sigma_tauM_ps : (N,), optional
        Uncertainties on tau_M.  When given, the fit is weighted by 1/dD^2 as
        quadric does, and ``chi2`` is a true chi-square.  Otherwise all residues
        carry equal weight and ``chi2`` is only meaningful up to a constant
        factor (which cancels in :func:`f_statistic`).
    init_axis : (3,), optional   starting guess for the unique axis

    Returns
    -------
    dict with keys
        model, Diso, Dxx, Dyy, Dzz   principal values (s^-1), ascending
        axes                         principal axes as columns, molecular frame
        axis                         'axial': the unique (symmetry) axis;
                                     'aniso': the z axis; None for 'iso'
        tau_c_ns                     1 / (6 Diso), in ns
        chi2, chi2_red, n_params, n_points, cost
    plus, for 'axial': Dpar, Dper, Dratio, theta, phi
    and,  for 'aniso': zz_ratio (2Dzz/(Dxx+Dyy)), xy_ratio (Dxx/Dyy),
                       phi, theta, psi

    Principal values follow quadric's convention Dxx <= Dyy <= Dzz.  The axial
    model is free to come back oblate (Dratio < 1), in which case the unique
    axis is x rather than z; use ``axis``, which accounts for that, rather than
    assuming z.  ``theta``/``phi`` are the polar/azimuthal angles of ``axis``.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")

    tauM_s = np.asarray(tauM_ps, float) * 1e-12
    y = 1.0 / (6.0 * tauM_s)                       # observed local D (s^-1)
    u = _unit(np.asarray(vectors, float))
    if len(u) != len(y):
        raise ValueError(f"got {len(y)} tau_M but {len(u)} vectors")

    n_pts = len(y)
    n_par = _N_PARAMS[model]
    scale = float(np.median(y))                    # ~1e7, conditions the fit

    # Unweighted, sigma is a flat `scale`: that keeps residuals O(1) and, since
    # chi2 then carries a constant factor, leaves the F-test between models
    # unchanged.  chi2 is only a true chi-square when sigma_tauM_ps is given.
    if sigma_tauM_ps is None:
        sigma = np.full_like(y, scale)
    else:                                          # dD = D dtau / tau
        sigma = y * np.asarray(sigma_tauM_ps, float) / np.asarray(tauM_ps, float)

    def chi2_of(D, axes):
        return float(np.sum(((local_D(D, axes, u) - y) / sigma) ** 2))

    if model == "iso":
        # the weighted mean has a closed form; no optimiser needed
        wt = 1.0 / sigma ** 2
        Diso = float(np.sum(y * wt) / np.sum(wt))
        D = np.array([Diso] * 3)
        axes = np.eye(3)
        chi2 = chi2_of(D, axes)
        extra = {}
    else:
        if model == "axial":
            def unpack(p):
                Diso = p[0] * scale
                Dpar, Dper = _axial_D(Diso, p[1])
                return (Dper, Dper, Dpar), _zyz(p[3], p[2], 0.0)

            p0_shape = [1.2]
            lo, hi = [0.2], [5.0]
        else:
            def unpack(p):
                Diso = p[0] * scale
                return _aniso_D(Diso, p[1], p[2]), _zyz(p[3], p[4], p[5])

            p0_shape = [1.15, 1.0]
            lo, hi = [0.2, 0.2], [5.0, 5.0]

        def resid(p):
            Dvals, R = unpack(p)
            return (local_D(Dvals, R, u) - y) / sigma

        n_ang = n_par - 1 - len(p0_shape)
        bounds = ([0.1] + lo + [-np.inf] * n_ang,
                  [10.0] + hi + [np.inf] * n_ang)

        if init_axis is not None:
            a = _unit(np.asarray(init_axis, float))
            seed_ang = [(float(np.arccos(np.clip(a[2], -1, 1))),
                         float(np.arctan2(a[1], a[0])))]
        else:
            seed_ang = []
        # multi-start: the angular part is multi-modal
        seed_ang += [(t, p) for t in (0.3, 0.9, 1.6, 2.3)
                     for p in (0.0, 1.6, 3.1, 4.7)]

        best = None
        for th, ph in seed_ang:
            for ps in ((0.0,) if model == "axial" else (-1.5, 0.0, 1.5)):
                p0 = ([1.0] + p0_shape
                      + ([th, ph] if model == "axial" else [ph, th, ps]))
                res = least_squares(resid, p0, bounds=bounds)
                if best is None or res.cost < best.cost:
                    best = res

        Dvals, axes = unpack(best.x)
        D = np.array(Dvals)
        chi2 = chi2_of(D, axes)
        Diso = float(np.mean(D))

    # quadric convention: ascending principal values, right-handed frame
    order = np.argsort(D)
    D, axes = D[order], axes[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1.0
    Dxx, Dyy, Dzz = (float(v) for v in D)

    if model == "iso":
        axis = None
    elif model == "axial":
        Dpar, Dper = _axial_D(Diso, best.x[1])
        # Sorting put the *unique* axis on z for a prolate tensor (Dpar > Dper)
        # but on x for an oblate one, where z is one of the degenerate pair.
        axis = axes[:, 2] if Dpar >= Dper else axes[:, 0]
        extra = dict(Dpar=Dpar, Dper=Dper, Dratio=float(best.x[1]),
                     theta=float(np.arccos(np.clip(axis[2], -1, 1))),
                     phi=float(np.arctan2(axis[1], axis[0])))
    else:
        axis = axes[:, 2]           # no unique axis exists; report z
        phi, theta, psi = _euler_zyz(axes)
        extra = dict(zz_ratio=2.0 * Dzz / (Dxx + Dyy), xy_ratio=Dxx / Dyy,
                     phi=phi, theta=theta, psi=psi)

    return dict(model=model, Diso=Diso, Dxx=Dxx, Dyy=Dyy, Dzz=Dzz, axes=axes,
                axis=axis,
                tau_c_ns=1.0 / (6.0 * Diso) * 1e9,
                chi2=chi2, chi2_red=chi2 / max(n_pts - n_par, 1),
                n_params=n_par, n_points=n_pts,
                cost=float(0.5 * chi2), **extra)


def describe(fit):
    """One-line human-readable summary of a :func:`fit_diffusion` result."""
    s = (f"{fit['model']}: Diso={fit['Diso'] * 1e-7:.3f}e7 s^-1"
         f"  tau_c={fit['tau_c_ns']:.2f} ns")
    if fit["model"] == "axial":
        s += f"  Dpar/Dper={fit['Dratio']:.3f}"
    elif fit["model"] == "aniso":
        s += (f"  2Dzz/(Dxx+Dyy)={fit['zz_ratio']:.3f}"
              f"  Dxx/Dyy={fit['xy_ratio']:.3f}")
    return s + f"  chi2_red={fit['chi2_red']:.4g}"


def f_statistic(fit_simple, fit_complex):
    """F-statistic comparing two nested diffusion fits (quadric's model test).

    ``F = ((X2_s - X2_c) / (p_c - p_s)) / (X2_c / (n - p_c))``; a large F favours
    the more complex model.  Both fits must come from the same data set.
    """
    if fit_simple["n_points"] != fit_complex["n_points"]:
        raise ValueError("fits must be over the same data set")
    n = fit_complex["n_points"]
    dp = fit_complex["n_params"] - fit_simple["n_params"]
    if dp <= 0:
        raise ValueError("fit_complex must have more parameters than fit_simple")
    return (((fit_simple["chi2"] - fit_complex["chi2"]) / dp)
            / (fit_complex["chi2"] / (n - fit_complex["n_params"])))


def select_diffusion_model(tauM_ps, vectors, sigma_tauM_ps=None, alpha=0.05,
                           init_axis=None):
    """Fit all three models and keep the simplest one the data justify.

    Walks the nested ladder iso -> axial -> aniso, taking each step only when an
    F-test says the extra parameters buy a significantly better fit at the
    ``alpha`` level; this is the criterion quadric's output is designed for.
    Fitting all three costs well under a second, so this is cheap next to
    computing the correlation functions they are fit to.

    Beware that the F-test assumes independent, normally distributed residuals.
    Per-residue tau_M are neither -- they share a structure and inherit the
    systematic error of the local-diffusion approximation -- so treat the choice
    as a well-founded default rather than a definitive answer, and pass an
    explicit model when you know better.

    Parameters
    ----------
    alpha : float    significance level for accepting a more complex model

    Returns
    -------
    (fit, trials) : the winning :func:`fit_diffusion` result, and {model: fit}
        for all three so the losing fits can be inspected.

    The winning fit carries a ``selection`` key: one ``(simpler, complex, F, p,
    accepted)`` tuple per rung of the ladder.
    """
    trials = {m: fit_diffusion(tauM_ps, vectors, model=m,
                               sigma_tauM_ps=sigma_tauM_ps, init_axis=init_axis)
              for m in MODELS}
    chosen, steps = "iso", []
    for nxt in ("axial", "aniso"):
        F = f_statistic(trials[chosen], trials[nxt])
        dp = trials[nxt]["n_params"] - trials[chosen]["n_params"]
        df2 = trials[nxt]["n_points"] - trials[nxt]["n_params"]
        p = float(f_dist.sf(F, dp, df2)) if F > 0 else 1.0
        accepted = p < alpha
        steps.append((chosen, nxt, float(F), p, accepted))
        if accepted:
            chosen = nxt

    fit = trials[chosen]
    fit["selection"] = steps
    return fit, trials


def fit_axial_diffusion(tauM_ps, nh_vectors, init_axis=None):
    """Fit an axially symmetric diffusion tensor to per-residue tau_M.

    Thin wrapper over ``fit_diffusion(..., model='axial')``, kept for callers
    that only ever want the axial model.
    """
    return fit_diffusion(tauM_ps, nh_vectors, model="axial",
                         init_axis=init_axis)


def tau_R_from_fit(vectors, fit):
    """Per-vector tumbling time tau_R (ns) for any diffusion model.

    Parameters
    ----------
    vectors : (N, 3)   bond vectors (e.g. methyl C-C axes) in the same molecular
                       frame as the vectors the tensor was fit to
    fit : dict         a :func:`fit_diffusion` result

    Returns
    -------
    ndarray (N,)       tau_R of each vector, in ns
    """
    D = (fit["Dxx"], fit["Dyy"], fit["Dzz"])
    return 1.0 / (6.0 * local_D(D, fit["axes"], vectors)) * 1e9


def methyl_tau_R(cc_vectors, Diso, Dpar, Dper, axis):
    """Per-methyl tumbling time tau_R (ns) from C-C axis orientation.

    tau_R,i = 1 / (6 (Diso - P2(cos theta_i) (Dpar - Dper)/3)),
    theta_i = angle between the C-C(methyl) axis and the unique diffusion axis.

    Axially symmetric special case of :func:`tau_R_from_fit`.
    """
    v = _unit(np.asarray(cc_vectors, float))
    axis = _unit(np.asarray(axis, float))
    P2 = 0.5 * (3.0 * (v @ axis) ** 2 - 1.0)
    Di = Diso - P2 * (Dpar - Dper) / 3.0
    return 1.0 / (6.0 * Di) * 1e9
