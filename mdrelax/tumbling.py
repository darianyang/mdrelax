"""Overall tumbling: isotropic tau_c and the axially symmetric diffusion tensor.

The diffusion-tensor path is a pure-Python replacement for the pdbinertia +
quadric_diffusion pipeline used by ABSURDer:

1. Per-residue backbone tumbling times tau_M are obtained by fitting the
   backbone N-H time-correlation functions (which retain overall tumbling) to
   simple and extended Lipari-Szabo models (:func:`fit_backbone_tauM`).
2. An axially symmetric diffusion tensor (Diso, Dpar/Dper, unique axis) is fit
   to those tau_M using the local-diffusion approximation
   ``1/(6 tau_M,i) = Diso - P2(cos a_i) (Dpar - Dper)/3`` (:func:`fit_axial_diffusion`).
   The axis orientation is a free parameter, so the result is independent of the
   input coordinate frame and pdbinertia is not needed.
3. Per-methyl tumbling times tau_R follow from the C-C(methyl) axis orientation
   relative to the unique axis (:func:`methyl_tau_R`).
"""

import numpy as np
from scipy.optimize import curve_fit, least_squares


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


def fit_backbone_tauM(tcf_bb, n_nh, l_block_ps, start_tau_ps=10000.0,
                      accuracy=100):
    """Per-residue backbone tumbling times tau_M (ps) from N-H TCFs.

    Ports the ABSURDer simple/extended Lipari-Szabo fit and model-selection
    logic.  ``tcf_bb`` is a single block array with time in column 0 and one
    N-H correlation function per subsequent column.

    Returns
    -------
    tauM : ndarray (n_nh,)   per-residue tumbling times (ps); NaN entries are
        replaced with the residue-set mean.
    """
    tcf_bb = np.asarray(tcf_bb)
    fit_length = int(l_block_ps / 2)
    exp_t = _exp_points(l_block_ps, fit_length, accuracy)
    t = exp_t.astype(float)

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


# ── Axially symmetric diffusion tensor ─────────────────────────────────────

def _axis_from_angles(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])


def _D_components(Diso, Dratio):
    """Return (Dpar, Dper) for isotropic value Diso and ratio Dpar/Dper."""
    Dpar = 3.0 * Diso / (1.0 + 2.0 / Dratio)
    Dper = 3.0 * Diso / (2.0 + Dratio)
    return Dpar, Dper


def inertia_principal_axes(coords, masses=None):
    """Principal axes (columns of returned matrix) of the mass-weighted inertia
    tensor.  Used only to seed the diffusion-axis fit."""
    coords = np.asarray(coords, dtype=float)
    if masses is None:
        masses = np.ones(len(coords))
    masses = np.asarray(masses, float)
    com = np.average(coords, axis=0, weights=masses)
    r = coords - com
    I = np.zeros((3, 3))
    for m, ri in zip(masses, r):
        I += m * (np.eye(3) * (ri @ ri) - np.outer(ri, ri))
    _, eigvecs = np.linalg.eigh(I)
    return eigvecs


def fit_axial_diffusion(tauM_ps, nh_vectors, init_axis=None):
    """Fit an axially symmetric diffusion tensor to per-residue tau_M.

    Model: ``1/(6 tau_M,i) = Diso - P2(cos a_i) (Dpar - Dper)/3`` where a_i is
    the angle between N-H vector i and the unique diffusion axis.

    Parameters
    ----------
    tauM_ps : ndarray (N,)     per-residue tumbling times (ps)
    nh_vectors : ndarray (N,3) N-H bond vectors in the molecular frame
    init_axis : ndarray (3,), optional  starting guess for the unique axis

    Returns
    -------
    dict: Diso (s^-1), Dratio, Dpar, Dper, axis (unit vec), tau_c_ns, cost
    """
    tauM_s = np.asarray(tauM_ps, float) * 1e-12
    y = 1.0 / (6.0 * tauM_s)                       # target local D (s^-1)
    u = np.asarray(nh_vectors, float)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)

    scale = np.median(y)                           # ~1e7, condition the fit
    if init_axis is None:
        init_axis = np.array([0.0, 0.0, 1.0])
    init_axis = init_axis / np.linalg.norm(init_axis)
    theta0 = np.arccos(np.clip(init_axis[2], -1, 1))
    phi0 = np.arctan2(init_axis[1], init_axis[0])

    def resid(p):
        Diso, Dratio, theta, phi = p
        Diso *= scale
        Dpar, Dper = _D_components(Diso, Dratio)
        axis = _axis_from_angles(theta, phi)
        cosa = u @ axis
        P2 = 0.5 * (3.0 * cosa**2 - 1.0)
        pred = Diso - P2 * (Dpar - Dper) / 3.0
        return (pred - y) / scale

    best = None
    # try a few axis seeds to avoid local minima
    seeds = [(theta0, phi0), (0.0, 0.0), (np.pi / 2, 0.0), (np.pi / 2, np.pi / 2)]
    for th, ph in seeds:
        p0 = [1.0, 1.2, th, ph]
        res = least_squares(resid, p0,
                            bounds=([0.1, 1.0, -np.inf, -np.inf],
                                    [10.0, 3.0, np.inf, np.inf]))
        if best is None or res.cost < best.cost:
            best = res

    Diso = best.x[0] * scale
    Dratio = best.x[1]
    Dpar, Dper = _D_components(Diso, Dratio)
    axis = _axis_from_angles(best.x[2], best.x[3])
    return dict(Diso=Diso, Dratio=Dratio, Dpar=Dpar, Dper=Dper, axis=axis,
                tau_c_ns=1.0 / (6.0 * Diso) * 1e9, cost=float(best.cost))


def methyl_tau_R(cc_vectors, Diso, Dpar, Dper, axis):
    """Per-methyl tumbling time tau_R (ns) from C-C axis orientation.

    tau_R,i = 1 / (6 (Diso - P2(cos theta_i) (Dpar - Dper)/3)),
    theta_i = angle between the C-C(methyl) axis and the unique diffusion axis.
    """
    v = np.asarray(cc_vectors, float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    cost = v @ axis
    P2 = 0.5 * (3.0 * cost**2 - 1.0)
    Di = Diso - P2 * (Dpar - Dper) / 3.0
    return 1.0 / (6.0 * Di) * 1e9
