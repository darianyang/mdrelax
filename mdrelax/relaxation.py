#!/usr/bin/env python3
"""
nh_relaxation.py
================
Calculate per-residue NMR relaxation rates from an MD trajectory for:

  1. Backbone ¹⁵N amide (NH) — R1, R2, hetNOE
  2. Side-chain methyl groups (²H quadrupolar) — R(Dz), R(3Dz²−2), R(Dy)

Tumbling models
---------------
Isotropic (default)
    Single global τc.  Lipari-Szabo (LS) and Extended Model-Free (EMF)
    internal-motion models.

Anisotropic (--anisotropic)
    Axially symmetric diffusion tensor (D_par, D_perp).
    Spectral density decomposes into 3 Lorentzians (Woessner 1962;
    Tjandra et al. 1995).  LS and EMF internal-motion models supported.
    D_par and D_perp are estimated from the trajectory (inertia tensor +
    principal-axis ACF refinement) or supplied by the user (--Dpar, --Dperp).
    For methyls, per-methyl τR,i is computed from the C–C_methyl axis
    orientation relative to the diffusion tensor PAF.

Python API
----------
    from nh_relaxation import RelaxationCalculator

    calc = RelaxationCalculator(
        "topology.pdb", "traj.xtc",
        field_MHz=[500, 950],
        tau_c_ns=11.6,
        do_methyl=True,
    )
    df_backbone, df_methyl = calc.run()

CLI usage
---------
    # Backbone NH only, isotropic:
    python nh_relaxation.py topology.pdb traj.xtc

    # Backbone NH + methyl ²H, isotropic:
    python nh_relaxation.py topology.pdb traj.xtc --methyl

    # Methyl only, anisotropic, supply D tensor (units: 1e7 s⁻¹):
    python nh_relaxation.py topology.pdb traj.xtc --methyl --no-backbone \\
        --anisotropic --Dpar 2.0 --Dperp 1.4

    # Multiple field strengths:
    python nh_relaxation.py topology.pdb traj.xtc --methyl --field 500 700 950

    # Provide τc manually (isotropic):
    python nh_relaxation.py topology.pdb traj.xtc --tau_c 11.6

Dependencies
------------
    MDAnalysis >= 2.0
    numpy, scipy, pandas

Backbone NH output columns
--------------------------
    resid, resname, probe, field_MHz, tau_c_ns,
    LS_S2, LS_tau_e_ps, LS_rmse, LS_converged, LS_R1, LS_R2, LS_hetNOE,
    EMF_S2f, EMF_S2s, EMF_S2, EMF_tau_f_ps, EMF_tau_s_ps,
    EMF_rmse, EMF_converged, EMF_R1, EMF_R2, EMF_hetNOE
    [+ anisotropic columns when --anisotropic]

Methyl ²H output columns
------------------------
    resid, resname, probe, methyl_name, field_MHz, tau_c_ns, tau_R_i_ns,
    LS_S2_axis, LS_tau_f_ps, LS_rmse, LS_converged,
    LS_R_Dz, LS_R_3Dz2m2, LS_R_Dy,
    EMF_S2_axis, EMF_tau_f_ps, EMF_tau_s_ps, EMF_rmse, EMF_converged,
    EMF_R_Dz, EMF_R_3Dz2m2, EMF_R_Dy
    [+ anisotropic variants when --anisotropic]

References
----------
    Woessner (1962) J Chem Phys 37:647
    Lipari & Szabo (1982) JACS 104:4546, 4559
    Clore et al. (1990) JACS 112:4989  [EMF]
    Tjandra et al. (1995) J Magn Reson A 111:177
    Millet et al. (2002) JACS 124:6439  [²H relaxation rates]
    Skrynnikov et al. (2002) JACS 124:6449
    Hoffmann et al. (2018) PCCP 20:24577  [methyl MD methodology]
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
except ImportError:
    sys.exit("MDAnalysis not found. Install with: pip install MDAnalysis")


# ═══════════════════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════════════════

MU_0          = 4.0 * np.pi * 1e-7     # T·m/A
HBAR          = 1.054571817e-34         # J·s
GAMMA_H       = 2.6752219e8            # rad/(s·T)  ¹H
GAMMA_N       = -2.7126e7              # rad/(s·T)  ¹⁵N
GAMMA_D       = 4.1066e7               # rad/(s·T)  ²H
R_NH          = 1.02e-10               # m  N–H bond length
DELTA_SIGMA_N = -160e-6                # ¹⁵N CSA (dimensionless)
CHI_Q         = 2 * np.pi * 167e3      # rad/s  ²H quadrupolar coupling (167 kHz)

# ¹⁵N dipolar coupling constant squared
_D_CONST_NH = (MU_0 / (4 * np.pi)) * (GAMMA_H * abs(GAMMA_N) * HBAR) / R_NH**3
D2_NH = _D_CONST_NH**2

# ²H quadrupolar prefactor  (3/40)·χQ²
QUAD_PREFACTOR = (3.0 / 40.0) * CHI_Q**2


# ═══════════════════════════════════════════════════════════════════════════════
# Methyl group definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Maps residue name → list of (methyl_label, C_methyl_names, H_names, C_axis_names)
# C_axis_names: the carbon one bond up the chain (for C–C_methyl axis vector)
METHYL_DEFS = {
    'ALA': [('CB',  ['CB'],  ['HB1','HB2','HB3','1HB','2HB','3HB'],  ['CA'])],
    'VAL': [('CG1', ['CG1'], ['HG11','HG12','HG13','1HG1','2HG1','3HG1'], ['CB']),
            ('CG2', ['CG2'], ['HG21','HG22','HG23','1HG2','2HG2','3HG2'], ['CB'])],
    'LEU': [('CD1', ['CD1'], ['HD11','HD12','HD13','1HD1','2HD1','3HD1'], ['CG']),
            ('CD2', ['CD2'], ['HD21','HD22','HD23','1HD2','2HD2','3HD2'], ['CG'])],
    'ILE': [('CG2', ['CG2'], ['HG21','HG22','HG23','1HG2','2HG2','3HG2'], ['CB']),
            ('CD1', ['CD1'], ['HD11','HD12','HD13','1HD1','2HD1','3HD1'], ['CG1'])],
    'MET': [('CE',  ['CE'],  ['HE1','HE2','HE3','1HE','2HE','3HE'],  ['SD'])],
    'THR': [('CG2', ['CG2'], ['HG21','HG22','HG23','1HG2','2HG2','3HG2'], ['CB'])],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Pure physics functions — spectral densities
# ═══════════════════════════════════════════════════════════════════════════════
# These are stateless and kept at module level for reuse / testing.

def _tau_prime(tau_c_s, tau_e_s):
    """Effective correlation time: 1/τ' = 1/τc + 1/τe."""
    return 1.0 / (1.0 / tau_c_s + 1.0 / tau_e_s)


def J_lipari_szabo(omega, S2, tau_e_ps, tau_c_ps):
    """Lipari-Szabo spectral density (Lipari & Szabo 1982)."""
    tau_c = tau_c_ps * 1e-12
    tau_e = tau_e_ps * 1e-12
    tp    = _tau_prime(tau_c, tau_e)
    return (2.0 / 5.0) * (S2 * tau_c / (1 + (omega * tau_c)**2)
                           + (1 - S2) * tp / (1 + (omega * tp)**2))


def J_extended_mf(omega, S2f, S2s, tau_f_ps, tau_s_ps, tau_c_ps):
    """Extended Model-Free spectral density (Clore et al. 1990)."""
    tau_c = tau_c_ps * 1e-12
    tau_f = tau_f_ps * 1e-12
    tau_s = tau_s_ps * 1e-12
    S2    = S2f * S2s
    tpf   = _tau_prime(tau_c, tau_f)
    tps   = _tau_prime(tau_c, tau_s)
    return (2.0 / 5.0) * (S2 * tau_c / (1 + (omega * tau_c)**2)
                           + (1 - S2f) * tpf / (1 + (omega * tpf)**2)
                           + S2f * (1 - S2s) * tps / (1 + (omega * tps)**2))


def _aniso_taus(D_par_s, D_perp_s):
    """Three correlation times for axially symmetric diffusion tensor."""
    return (1.0 / (6.0 * D_perp_s),
            1.0 / (D_par_s + 5.0 * D_perp_s),
            1.0 / (4.0 * D_par_s + 2.0 * D_perp_s))


def _aniso_weights(theta_rad):
    """Geometric weights A0, A1, A2 for anisotropic spectral density."""
    ct = np.cos(theta_rad)
    st = np.sin(theta_rad)
    return ((3 * ct**2 - 1)**2 / 4.0,
            3 * st**2 * ct**2,
            (3.0 / 4.0) * st**4)


def J_aniso_lipari_szabo(omega, S2, tau_e_ps, D_par_s, D_perp_s, theta_rad):
    """Anisotropic Lipari-Szabo spectral density (Woessner 1962; Tjandra 1995)."""
    tau_e = tau_e_ps * 1e-12
    J = 0.0
    for Am, tau_m in zip(_aniso_weights(theta_rad), _aniso_taus(D_par_s, D_perp_s)):
        tp = _tau_prime(tau_m, tau_e)
        J += Am * (S2 * tau_m / (1 + (omega * tau_m)**2)
                   + (1 - S2) * tp / (1 + (omega * tp)**2))
    return (2.0 / 5.0) * J


def J_aniso_extended_mf(omega, S2f, S2s, tau_f_ps, tau_s_ps, D_par_s, D_perp_s, theta_rad):
    """Anisotropic Extended Model-Free spectral density."""
    tau_f = tau_f_ps * 1e-12
    tau_s = tau_s_ps * 1e-12
    S2    = S2f * S2s
    J = 0.0
    for Am, tau_m in zip(_aniso_weights(theta_rad), _aniso_taus(D_par_s, D_perp_s)):
        tpf = _tau_prime(tau_m, tau_f)
        tps = _tau_prime(tau_m, tau_s)
        J += Am * (S2 * tau_m / (1 + (omega * tau_m)**2)
                   + (1 - S2f) * tpf / (1 + (omega * tpf)**2)
                   + S2f * (1 - S2s) * tps / (1 + (omega * tps)**2))
    return (2.0 / 5.0) * J


# ═══════════════════════════════════════════════════════════════════════════════
# Pure physics functions — relaxation rates
# ═══════════════════════════════════════════════════════════════════════════════

def _nh_rates_from_J(J, omega_H, omega_N, c2):
    """R1, R2, hetNOE from a spectral density callable J(ω)."""
    J0    = J(0.0)
    JwN   = J(omega_N)
    JwH   = J(omega_H)
    JwHmN = J(omega_H - omega_N)
    JwHpN = J(omega_H + omega_N)
    R1  = (D2_NH / 4.0) * (JwHmN + 3 * JwN + 6 * JwHpN) + c2 * JwN
    R2  = (D2_NH / 8.0) * (4 * J0 + JwHmN + 3 * JwN + 6 * JwH + 6 * JwHpN) \
        + (c2 / 6.0) * (4 * J0 + 3 * JwN)
    noe = 1.0 + (GAMMA_H / GAMMA_N) * (D2_NH / 4.0) * (6 * JwHpN - JwHmN) / R1
    return R1, R2, noe


def _nh_larmor(field_MHz):
    """Return (omega_H, omega_N, c2) for a given ¹H field in MHz."""
    omega_H = 2 * np.pi * field_MHz * 1e6
    omega_N = omega_H * abs(GAMMA_N) / GAMMA_H
    c2      = ((1.0 / 3.0) * GAMMA_N * omega_H / GAMMA_H * DELTA_SIGMA_N)**2
    return omega_H, omega_N, c2


def nh_relaxation_rates_iso(J_func, field_MHz, tau_c_ps, *J_params):
    """R1, R2, hetNOE — isotropic tumbling."""
    omega_H, omega_N, c2 = _nh_larmor(field_MHz)
    return _nh_rates_from_J(lambda w: J_func(w, *J_params, tau_c_ps),
                            omega_H, omega_N, c2)


def nh_relaxation_rates_aniso(J_func, field_MHz, *J_params):
    """R1, R2, hetNOE — anisotropic tumbling (D_par, D_perp, theta in J_params)."""
    omega_H, omega_N, c2 = _nh_larmor(field_MHz)
    return _nh_rates_from_J(lambda w: J_func(w, *J_params), omega_H, omega_N, c2)


def _deuterium_rates_from_J(J, omega_D):
    """
    R(Dz), R(3Dz²−2), R(Dy) from a spectral density callable J(ω).

    Equations from Millet et al. JACS 2002 124:6439.
    Fast-motion limit gives R(Dz):R(3Dz²−2):R(Dy) = 1:2:3 (Abragam 1961 VIII.295).
    """
    J0   = J(0.0)
    JwD  = J(omega_D)
    J2wD = J(2 * omega_D)
    R_Dz     = QUAD_PREFACTOR * (JwD + 4 * J2wD)
    R_3Dz2m2 = QUAD_PREFACTOR * (3 * J0 + 5 * JwD + 2 * J2wD)
    R_Dy     = QUAD_PREFACTOR * 0.5 * (9 * J0 + 15 * JwD + 6 * J2wD)
    return R_Dz, R_3Dz2m2, R_Dy


def methyl_relaxation_rates_iso(J_func, field_MHz, tau_c_ps, S2_axis, tau_e_ps):
    """
    R(Dz), R(3Dz²−2), R(Dy) — isotropic tumbling.

    S² passed to J = S2_axis / 9  (fast 3-fold methyl rotation factor).
    """
    omega_D = 2 * np.pi * field_MHz * 1e6 * GAMMA_D / GAMMA_H
    return _deuterium_rates_from_J(
        lambda w: J_func(w, S2_axis / 9.0, tau_e_ps, tau_c_ps), omega_D)


def methyl_relaxation_rates_aniso(J_func, field_MHz, S2_axis, tau_e_ps,
                                   D_par_s, D_perp_s, theta_rad):
    """R(Dz), R(3Dz²−2), R(Dy) — anisotropic tumbling."""
    omega_D = 2 * np.pi * field_MHz * 1e6 * GAMMA_D / GAMMA_H
    return _deuterium_rates_from_J(
        lambda w: J_func(w, S2_axis / 9.0, tau_e_ps, D_par_s, D_perp_s, theta_rad),
        omega_D)


def methyl_tau_R_i(D_par_s, D_perp_s, theta_rad):
    """
    Per-methyl effective tumbling time (Brüschweiler-Wright approximation).

    τR,i = 1 / (6·(Diso − P2(cosθ)·(D_par − D_perp)/3))

    Reference: Hoffmann et al. (2018) PCCP 20:24577.
    """
    D_iso = (D_par_s + 2 * D_perp_s) / 3.0
    P2t   = (3 * np.cos(theta_rad)**2 - 1) / 2.0
    Di    = max(D_iso - P2t * (D_par_s - D_perp_s) / 3.0, 1e5)
    return 1.0 / (6.0 * Di)


# ═══════════════════════════════════════════════════════════════════════════════
# Pure physics functions — ACF
# ═══════════════════════════════════════════════════════════════════════════════

def compute_p2_acf_fft(vectors, max_lag):
    """
    FFT-based P₂ autocorrelation function.  C(t) = ⟨P₂(u(0)·u(t))⟩, C(0)=1.

    Parameters
    ----------
    vectors : ndarray (n_frames, 3)
    max_lag : int

    Returns
    -------
    acf : ndarray (max_lag,)
    """
    n     = len(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    u     = vectors / np.where(norms < 1e-10, 1.0, norms)

    acf    = np.zeros(max_lag)
    counts = np.arange(n, n - max_lag, -1, dtype=float)
    for i in range(3):
        for j in range(3):
            xi  = u[:, i] * u[:, j]
            cc  = fftconvolve(xi, xi[::-1], mode='full')
            mid = len(cc) // 2
            acf += cc[mid:mid + max_lag] / counts

    acf = 1.5 * acf - 0.5
    return acf / acf[0]


def compute_vector_theta(vectors_mean, diffusion_axis):
    """
    Angle θ between each mean bond vector and the diffusion axis.

    Parameters
    ----------
    vectors_mean   : ndarray (N, 3)
    diffusion_axis : ndarray (3,)

    Returns
    -------
    theta_rad : ndarray (N,)
    """
    axis  = diffusion_axis / np.linalg.norm(diffusion_axis)
    norms = np.linalg.norm(vectors_mean, axis=1, keepdims=True)
    u     = vectors_mean / np.where(norms < 1e-10, 1.0, norms)
    return np.arccos(np.clip(np.abs(u @ axis), 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# RelaxationCalculator
# ═══════════════════════════════════════════════════════════════════════════════

class RelaxationCalculator:
    """
    Calculate per-residue NMR relaxation rates from an MD trajectory.

    Supports backbone ¹⁵N NH (R1, R2, hetNOE) and side-chain methyl ²H
    (R(Dz), R(3Dz²−2), R(Dy)) with isotropic or axially symmetric anisotropic
    tumbling, and Lipari-Szabo / Extended Model-Free internal motion models.

    Parameters
    ----------
    topology         : str    topology file (PDB, PSF, GRO, …)
    trajectory       : str    trajectory file (XTC, DCD, TRR, …)
    field_MHz        : float or list of float   ¹H Larmor frequency (MHz)
    tau_c_ns         : float or None   isotropic τc (ns); estimated if None
    anisotropic      : bool   enable axially symmetric diffusion tensor
    D_par_1e7        : float or None   D_∥ in 10⁷ s⁻¹ (user-supplied)
    D_perp_1e7       : float or None   D_⊥ in 10⁷ s⁻¹ (user-supplied)
    diffusion_axis   : array-like (3,) or None
    do_backbone      : bool   compute backbone ¹⁵N NH rates  (default True)
    do_methyl        : bool   compute side-chain methyl ²H rates (default False)
    max_lag_fraction : float  fraction of trajectory used as max ACF lag
    acf_fit_fraction : float  fraction of ACF used for model fitting
    verbose          : bool

    Examples
    --------
    >>> calc = RelaxationCalculator("top.pdb", "traj.xtc", field_MHz=[500, 950])
    >>> df_bb, df_me = calc.run()

    >>> # Anisotropic, methyl only, user-supplied D tensor:
    >>> calc = RelaxationCalculator(
    ...     "top.pdb", "traj.xtc",
    ...     do_backbone=False, do_methyl=True,
    ...     anisotropic=True, D_par_1e7=2.0, D_perp_1e7=1.4,
    ... )
    >>> _, df_me = calc.run()
    """

    def __init__(
        self,
        topology,
        trajectory,
        field_MHz=500.0,
        tau_c_ns=None,
        anisotropic=False,
        D_par_1e7=None,
        D_perp_1e7=None,
        diffusion_axis=None,
        do_backbone=True,
        do_methyl=False,
        max_lag_fraction=0.5,
        acf_fit_fraction=0.5,
        verbose=True,
    ):
        self.topology          = topology
        self.trajectory        = trajectory
        self.fields            = [field_MHz] if np.isscalar(field_MHz) else list(field_MHz)
        self._tau_c_ns_input   = tau_c_ns       # user-supplied; None → estimate
        self.anisotropic       = anisotropic
        self._D_par_1e7_input  = D_par_1e7
        self._D_perp_1e7_input = D_perp_1e7
        self._d_axis_input     = (np.array(diffusion_axis, dtype=float)
                                  if diffusion_axis is not None else None)
        self.do_backbone       = do_backbone
        self.do_methyl         = do_methyl
        self.max_lag_fraction  = max_lag_fraction
        self.acf_fit_fraction  = acf_fit_fraction
        self.verbose           = verbose

        # Set by _setup_universe()
        self.u          = None   # MDAnalysis Universe (aligned, in-memory)
        self.n_frames   = None
        self.dt_ps      = None
        self.max_lag    = None
        self.n_fit      = None
        self.t_fit_ps   = None
        self.tau_c_ns   = None   # effective τc (ns)
        self.tau_c_ps   = None
        self.D_par_s    = None   # s⁻¹
        self.D_perp_s   = None
        self.d_axis     = None   # unit vector

        # Set by _compute_acfs_backbone() / _fit_acfs_backbone()
        self._nh_pairs   = None
        self._nh_vecs    = None   # (n_pairs, n_frames, 3)
        self._nh_theta   = None   # (n_pairs,) or None
        self._nh_acfs    = None   # (n_pairs, max_lag)
        self._nh_ls      = None   # list of dicts
        self._nh_emf     = None

        # Set by _compute_acfs_methyl() / _fit_acfs_methyl()
        self._methyl_groups      = None
        self._methyl_acfs        = None   # (n_methyls, max_lag)
        self._methyl_axis_vecs   = None   # (n_methyls, n_frames, 3)
        self._methyl_theta       = None
        self._methyl_tau_R_i_arr = None
        self._methyl_ls          = None
        self._methyl_emf         = None

    # ── Logging helper ────────────────────────────────────────────────────────

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_universe(self):
        """Load trajectory, estimate τc / D tensor, align to Cα backbone."""
        self._log(f"\n{'='*60}")
        self._log("Loading trajectory …")
        self.u        = mda.Universe(self.topology, self.trajectory)
        self.n_frames = len(self.u.trajectory)
        self.dt_ps    = self.u.trajectory.dt
        self._log(f"  {self.n_frames} frames × {self.dt_ps:.1f} ps "
                  f"= {self.n_frames * self.dt_ps / 1000:.2f} ns")
        self._log(f"  {len(self.u.residues)} residues")

        # Diffusion tensor / τc
        if self.anisotropic:
            self._setup_diffusion_tensor()
        else:
            self._setup_tau_c()

        # Align trajectory
        self._log("\nAligning trajectory to Cα backbone …")
        ref     = mda.Universe(self.topology)
        aligner = align.AlignTraj(
            self.u, ref, select="backbone and name CA", in_memory=True)
        aligner.run()
        self._log("  Done.")

        self.max_lag  = int(self.n_frames * self.max_lag_fraction)
        self.n_fit    = max(10, int(self.max_lag * self.acf_fit_fraction))
        self.t_fit_ps = np.arange(self.n_fit) * self.dt_ps

    def _setup_tau_c(self):
        """Estimate or accept isotropic τc."""
        if self._tau_c_ns_input is not None:
            self.tau_c_ns = self._tau_c_ns_input
            self._log(f"\nUsing τc = {self.tau_c_ns:.3f} ns")
        else:
            self._log("\nEstimating τc …")
            self.tau_c_ns = self._estimate_tau_c()
        self.tau_c_ps = self.tau_c_ns * 1000.0

    def _setup_diffusion_tensor(self):
        """Estimate or accept axially symmetric diffusion tensor."""
        if self._D_par_1e7_input is not None and self._D_perp_1e7_input is not None:
            self.D_par_s  = self._D_par_1e7_input  * 1e7
            self.D_perp_s = self._D_perp_1e7_input * 1e7
            if self._d_axis_input is not None:
                self.d_axis = self._d_axis_input / np.linalg.norm(self._d_axis_input)
                self._log(f"\nUsing D_par={self._D_par_1e7_input:.4f}, "
                          f"D_perp={self._D_perp_1e7_input:.4f} ×10⁷ s⁻¹")
            else:
                self._log("\nD tensor provided — estimating axis from trajectory …")
                _, _, self.d_axis = self._estimate_diffusion_tensor()
        else:
            self._log("\nEstimating axially symmetric diffusion tensor …")
            self.D_par_s, self.D_perp_s, self.d_axis = self._estimate_diffusion_tensor()

        D_iso         = (self.D_par_s + 2 * self.D_perp_s) / 3.0
        self.tau_c_ns = 1.0 / (6.0 * D_iso) * 1e9
        self.tau_c_ps = self.tau_c_ns * 1000.0
        self._log(f"  Effective isotropic τc = {self.tau_c_ns:.3f} ns")

    # ── τc / D tensor estimation ──────────────────────────────────────────────

    def _estimate_tau_c(self, n_ca_sample=50):
        """Estimate global τc from Cα bond-vector P₂ ACF."""
        ca_atoms = self.u.select_atoms("backbone and name CA")
        step     = max(1, len(ca_atoms) // n_ca_sample)
        ca_sel   = ca_atoms[::step]
        self._log(f"  Estimating τc from {len(ca_sel)} Cα atoms …")

        ca_pos = np.zeros((self.n_frames, len(ca_sel), 3))
        for i, _ in enumerate(self.u.trajectory):
            ca_pos[i] = ca_sel.positions

        v          = ca_pos[:, 1:, :] - ca_pos[:, :-1, :]
        max_lag    = self.n_frames // 2
        global_acf = np.zeros(max_lag)
        for k in range(v.shape[1]):
            global_acf += compute_p2_acf_fft(v[:, k, :], max_lag)
        global_acf /= v.shape[1]

        t_ps  = np.arange(max_lag) * self.dt_ps
        n_fit = max(10, int(max_lag * self.acf_fit_fraction))

        try:
            popt, _ = curve_fit(
                lambda t, A, tau: A * np.exp(-t / tau),
                t_ps[:n_fit], global_acf[:n_fit],
                p0=[1.0, 5000.0], bounds=([0.5, 100.0], [1.5, 1e6]),
                maxfev=10000)
            tau_c_ns = popt[1] / 1000.0
            self._log(f"  τc = {tau_c_ns:.3f} ns")
        except RuntimeError:
            tau_c_ns = 8.0
            self._log(f"  τc fit failed — using fallback {tau_c_ns} ns")
        return tau_c_ns

    def _estimate_diffusion_tensor(self, n_ca_sample=50):
        """
        Estimate axially symmetric D tensor from inertia tensor + per-axis ACF.

        Returns (D_par_s, D_perp_s, axis_unit_vector).

        NOTE: Reliable estimation requires trajectory >> τc (~5–10×).
        For short trajectories, supply D_par/D_perp manually.
        """
        ca_atoms = self.u.select_atoms("backbone and name CA")
        step     = max(1, len(ca_atoms) // n_ca_sample)
        ca_sel   = ca_atoms[::step]
        self._log(f"  Estimating diffusion tensor from {len(ca_sel)} Cα atoms …")

        ca_pos = np.zeros((self.n_frames, len(ca_sel), 3))
        for i, _ in enumerate(self.u.trajectory):
            ca_pos[i] = ca_sel.positions

        # Average inertia tensor → principal axes
        I_avg = np.zeros((3, 3))
        for fi in range(self.n_frames):
            r  = ca_pos[fi] - ca_pos[fi].mean(axis=0)
            r2 = np.sum(r**2, axis=1)
            I_avg += np.eye(3) * r2.sum() - r.T @ r
        I_avg /= self.n_frames
        eigvals, eigvecs = np.linalg.eigh(I_avg)
        self._log(f"  Inertia moments: {eigvals[0]:.2e}, {eigvals[1]:.2e}, "
                  f"{eigvals[2]:.2e} Å²·Da  (I_max/I_min={eigvals[2]/eigvals[0]:.3f})")

        # Per-axis ACF → D_axes
        v_bonds = ca_pos[:, 1:, :] - ca_pos[:, :-1, :]
        n_bonds = v_bonds.shape[1]
        max_lag = self.n_frames // 2
        t_ps    = np.arange(max_lag) * self.dt_ps
        n_fit   = max(10, int(max_lag * self.acf_fit_fraction))
        D_axes  = np.zeros(3)

        for k in range(3):
            axis_k = eigvecs[:, k]
            acf_k  = np.zeros(max_lag)
            counts = np.arange(self.n_frames, self.n_frames - max_lag, -1, dtype=float)
            for b in range(n_bonds):
                vb    = v_bonds[:, b, :]
                norms = np.linalg.norm(vb, axis=1, keepdims=True)
                ub    = vb / np.where(norms < 1e-10, 1.0, norms)
                p2    = (3 * (ub @ axis_k)**2 - 1) / 2.0
                cc    = fftconvolve(p2, p2[::-1], mode='full')
                mid   = len(cc) // 2
                acf_k += cc[mid:mid + max_lag] / counts
            acf_k /= n_bonds
            acf_k /= acf_k[0]
            try:
                popt, _ = curve_fit(
                    lambda t, A, tau: A * np.exp(-t / tau),
                    t_ps[:n_fit], acf_k[:n_fit],
                    p0=[1.0, 5000.0], bounds=([0.1, 100.0], [2.0, 1e6]),
                    maxfev=10000)
                D_axes[k] = 1.0 / (6.0 * popt[1] * 1e-12)
            except RuntimeError:
                D_axes[k] = 1.0 / (6.0 * 8e-9)
            self._log(f"  D_axis{k} = {D_axes[k]*1e-7:.3f}×10⁷ s⁻¹  "
                      f"(τ={1/(6*D_axes[k])*1e9:.3f} ns)")

        unique_idx = int(np.argmax(D_axes))
        other_idx  = [i for i in range(3) if i != unique_idx]
        D_par_s    = D_axes[unique_idx]
        D_perp_s   = float(np.mean(D_axes[other_idx]))
        axis       = eigvecs[:, unique_idx]
        self._log(f"  D_par={D_par_s*1e-7:.4f}×10⁷, D_perp={D_perp_s*1e-7:.4f}×10⁷ s⁻¹  "
                  f"ratio={D_par_s/D_perp_s:.3f}")
        self._log(f"  Unique axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
        return D_par_s, D_perp_s, axis

    # ── Probe detection ───────────────────────────────────────────────────────

    def _find_nh_pairs(self):
        """Find all backbone amide NH pairs (excluding Pro)."""
        backbone_N = self.u.select_atoms("backbone and name N")
        pairs, skipped = [], []
        self.u.trajectory[0]

        for n_atom in backbone_N:
            if n_atom.resname == "PRO":
                skipped.append(f"PRO{n_atom.resid}(PRO)")
                continue
            h_sel = self.u.select_atoms(
                f"resid {n_atom.resid} and name H HN HT1 HT2 HT3 1H 2H 3H")
            if len(h_sel) == 0:
                h_sel = self.u.select_atoms(
                    f"name H* and around 1.2 (index {n_atom.index})")
            if len(h_sel) > 0:
                dists  = np.linalg.norm(h_sel.positions - n_atom.position, axis=1)
                pairs.append((n_atom, h_sel[np.argmin(dists)]))
            else:
                skipped.append(f"{n_atom.resname}{n_atom.resid}")

        self._log(f"  Found {len(pairs)} NH pairs")
        if skipped:
            self._log(f"  Skipped: {skipped[:10]}" + (" …" if len(skipped) > 10 else ""))
        return pairs

    def _find_methyl_groups(self):
        """Find all methyl groups (ALA/VAL/LEU/ILE/MET/THR)."""
        self.u.trajectory[0]
        groups, skipped = [], []

        for res in self.u.residues:
            if res.resname not in METHYL_DEFS:
                continue
            for (label, c_names, h_names, axis_names) in METHYL_DEFS[res.resname]:
                # Methyl carbon
                c_atom = next(
                    (res.atoms.select_atoms(f"name {n}")[0]
                     for n in c_names if len(res.atoms.select_atoms(f"name {n}")) > 0),
                    None)
                if c_atom is None:
                    skipped.append(f"{res.resname}{res.resid}:{label}(no C)")
                    continue

                # Axis carbon (one bond up the chain)
                c_axis = next(
                    (res.atoms.select_atoms(f"name {n}")[0]
                     for n in axis_names if len(res.atoms.select_atoms(f"name {n}")) > 0),
                    None)
                if c_axis is None:
                    heavy = res.atoms.select_atoms("not name H*")
                    dists = np.linalg.norm(heavy.positions - c_atom.position, axis=1)
                    dists[heavy.indices == c_atom.index] = 1e9
                    c_axis = heavy[np.argmin(dists)]

                # H atoms
                h_atoms = []
                for hname in h_names:
                    sel = res.atoms.select_atoms(f"name {hname}")
                    if len(sel) > 0:
                        h_atoms.append(sel[0])
                    if len(h_atoms) == 3:
                        break
                if len(h_atoms) < 3:
                    h_near = self.u.select_atoms(
                        f"name H* and around 1.15 (index {c_atom.index})")
                    for ha in h_near:
                        if ha not in h_atoms:
                            h_atoms.append(ha)
                        if len(h_atoms) == 3:
                            break
                if len(h_atoms) < 1:
                    skipped.append(f"{res.resname}{res.resid}:{label}(no H)")
                    continue

                groups.append(dict(resid=res.resid, resname=res.resname,
                                   methyl_name=label, C_methyl=c_atom,
                                   H_atoms=h_atoms[:3], C_axis=c_axis))

        self._log(f"  Found {len(groups)} methyl groups")
        counts = {}
        for g in groups:
            counts[g['resname']] = counts.get(g['resname'], 0) + 1
        self._log("  By residue type: " +
                  ", ".join(f"{k}×{v}" for k, v in sorted(counts.items())))
        if skipped:
            self._log(f"  Skipped: {skipped[:10]}" + (" …" if len(skipped) > 10 else ""))
        return groups

    # ── ACF computation ───────────────────────────────────────────────────────

    def _compute_acfs_backbone(self):
        """Collect NH vectors and compute P₂ ACFs for all NH pairs."""
        n_pairs = len(self._nh_pairs)
        self._log(f"\nCollecting NH vectors ({self.n_frames}×{n_pairs}) …")
        self._nh_vecs = np.zeros((n_pairs, self.n_frames, 3), dtype=np.float32)
        for fi, _ in enumerate(self.u.trajectory):
            for pi, (na, ha) in enumerate(self._nh_pairs):
                self._nh_vecs[pi, fi] = ha.position - na.position

        if self.anisotropic:
            self._nh_theta = compute_vector_theta(
                self._nh_vecs.mean(axis=1), self.d_axis)
            self._log(f"  NH θ range: {np.degrees(self._nh_theta.min()):.1f}°–"
                      f"{np.degrees(self._nh_theta.max()):.1f}°")

        self._log("\nComputing NH ACFs …")
        self._nh_acfs = np.zeros((n_pairs, self.max_lag))
        for pi in range(n_pairs):
            self._nh_acfs[pi] = compute_p2_acf_fft(self._nh_vecs[pi], self.max_lag)
            if self.verbose and (pi + 1) % 30 == 0:
                self._log(f"  {pi+1}/{n_pairs} …")

    def _compute_acfs_methyl(self):
        """Collect C–H vectors, compute per-methyl averaged P₂ ACFs."""
        n_methyls = len(self._methyl_groups)
        self._log(f"\nCollecting methyl C–H vectors "
                  f"({self.n_frames}×{n_methyls}×3H) …")

        self._methyl_acfs      = np.zeros((n_methyls, self.max_lag))
        self._methyl_axis_vecs = np.zeros((n_methyls, self.n_frames, 3),
                                          dtype=np.float32)

        for mi, mg in enumerate(self._methyl_groups):
            c_atom  = mg['C_methyl']
            h_atoms = mg['H_atoms']
            c_axis  = mg['C_axis']
            n_h     = len(h_atoms)

            ch_vecs = np.zeros((n_h, self.n_frames, 3), dtype=np.float32)
            for fi, _ in enumerate(self.u.trajectory):
                for hi, ha in enumerate(h_atoms):
                    ch_vecs[hi, fi] = ha.position - c_atom.position
                self._methyl_axis_vecs[mi, fi] = c_atom.position - c_axis.position

            acf_sum = np.zeros(self.max_lag)
            for hi in range(n_h):
                acf_sum += compute_p2_acf_fft(ch_vecs[hi], self.max_lag)
            self._methyl_acfs[mi] = acf_sum / n_h

            if self.verbose and (mi + 1) % 20 == 0:
                self._log(f"  {mi+1}/{n_methyls} methyls processed …")

        if self.anisotropic:
            self._methyl_theta = compute_vector_theta(
                self._methyl_axis_vecs.mean(axis=1), self.d_axis)
            self._methyl_tau_R_i_arr = np.array([
                methyl_tau_R_i(self.D_par_s, self.D_perp_s, self._methyl_theta[mi])
                for mi in range(n_methyls)
            ])
            self._log(f"  Methyl axis θ range: "
                      f"{np.degrees(self._methyl_theta.min()):.1f}°–"
                      f"{np.degrees(self._methyl_theta.max()):.1f}°")
            self._log(f"  τR,i range: "
                      f"{self._methyl_tau_R_i_arr.min()*1e9:.3f}–"
                      f"{self._methyl_tau_R_i_arr.max()*1e9:.3f} ns")

    # ── ACF fitting ───────────────────────────────────────────────────────────

    @staticmethod
    def _fit_ls(acf, t_ps, tau_c_ps, S2_max=1.0):
        """Fit ACF to Lipari-Szabo model. S2_max caps the upper bound on S²."""
        def model(t, S2, tau_e):
            return S2 + (1.0 - S2) * np.exp(-t / tau_e)
        try:
            popt, _ = curve_fit(model, t_ps, acf,
                                p0=[S2_max * 0.85, 50.0],
                                bounds=([0.0, 0.1], [S2_max, tau_c_ps * 2]),
                                maxfev=5000)
            S2, tau_e = np.clip(popt[0], 0.0, S2_max), popt[1]
            rmse = float(np.sqrt(np.mean((model(t_ps, S2, tau_e) - acf)**2)))
            return dict(S2=S2, tau_e_ps=tau_e, rmse=rmse, converged=True)
        except (RuntimeError, ValueError):
            return dict(S2=np.nan, tau_e_ps=np.nan, rmse=np.nan, converged=False)

    @staticmethod
    def _fit_emf(acf, t_ps, tau_c_ps, S2_max=1.0):
        """Fit ACF to Extended Model-Free model. S2_max caps S²f·S²s."""
        def model(t, S2f, S2s, tau_f, tau_s):
            S2 = S2f * S2s
            return S2 + S2f * (1 - S2s) * np.exp(-t / tau_s) + (1 - S2f) * np.exp(-t / tau_f)
        try:
            popt, _ = curve_fit(model, t_ps, acf,
                                p0=[0.95, 0.90, 5.0, 500.0],
                                bounds=([0.0, 0.0, 0.1, 1.0],
                                        [1.0, S2_max, tau_c_ps * 0.1, tau_c_ps * 2]),
                                maxfev=10000)
            S2f, S2s, tau_f, tau_s = popt
            S2 = float(np.clip(S2f * S2s, 0.0, S2_max))
            rmse = float(np.sqrt(np.mean((model(t_ps, *popt) - acf)**2)))
            return dict(S2f=S2f, S2s=S2s, S2=S2,
                        tau_f_ps=tau_f, tau_s_ps=tau_s, rmse=rmse, converged=True)
        except (RuntimeError, ValueError):
            return dict(S2f=np.nan, S2s=np.nan, S2=np.nan,
                        tau_f_ps=np.nan, tau_s_ps=np.nan, rmse=np.nan, converged=False)

    def _fit_acfs_backbone(self):
        """Fit all NH ACFs with LS and EMF models."""
        self._log("Fitting NH ACFs …")
        self._nh_ls  = [self._fit_ls(self._nh_acfs[pi, :self.n_fit],
                                     self.t_fit_ps, self.tau_c_ps)
                        for pi in range(len(self._nh_pairs))]
        self._nh_emf = [self._fit_emf(self._nh_acfs[pi, :self.n_fit],
                                      self.t_fit_ps, self.tau_c_ps)
                        for pi in range(len(self._nh_pairs))]

    def _fit_acfs_methyl(self):
        """
        Fit all methyl ACFs with LS and EMF models.

        The C–H ACF decays to S²_axis/9 (fast 3-fold rotation factor).
        We fit S² ∈ [0, 1/9] and report S²_axis = 9·S².
        """
        self._log("Fitting methyl ACFs …")
        S2_max = 1.0 / 9.0
        self._methyl_ls  = []
        self._methyl_emf = []

        for mi in range(len(self._methyl_groups)):
            acf = self._methyl_acfs[mi, :self.n_fit]

            # LS
            r = self._fit_ls(acf, self.t_fit_ps, self.tau_c_ps, S2_max=S2_max)
            if r['converged']:
                r['S2_axis'] = min(9.0 * r['S2'], 1.0)
                r['tau_f_ps'] = r.pop('tau_e_ps')   # rename for clarity
            else:
                r['S2_axis'] = np.nan
                r['tau_f_ps'] = np.nan
            self._methyl_ls.append(r)

            # EMF
            e = self._fit_emf(acf, self.t_fit_ps, self.tau_c_ps, S2_max=S2_max)
            if e['converged']:
                e['S2_axis'] = min(9.0 * e['S2'], 1.0)
            else:
                e['S2_axis'] = np.nan
            self._methyl_emf.append(e)

    # ── DataFrame assembly ────────────────────────────────────────────────────

    def _aniso_common_cols(self, theta_rad):
        """Dict of anisotropic metadata columns for one probe."""
        return dict(
            D_par_1e7=self.D_par_s * 1e-7,
            D_perp_1e7=self.D_perp_s * 1e-7,
            aniso_ratio=self.D_par_s / self.D_perp_s,
            diffusion_axis_x=self.d_axis[0],
            diffusion_axis_y=self.d_axis[1],
            diffusion_axis_z=self.d_axis[2],
            theta_deg=np.degrees(theta_rad),
        )

    def _build_backbone_df(self):
        """Assemble backbone NH results into a DataFrame."""
        rows = []
        for pi, (na, _) in enumerate(self._nh_pairs):
            ls  = self._nh_ls[pi]
            emf = self._nh_emf[pi]
            for fmhz in self.fields:
                row = dict(
                    probe='NH', resid=na.resid, resname=na.resname,
                    field_MHz=fmhz, tau_c_ns=self.tau_c_ns,
                    LS_S2=ls['S2'], LS_tau_e_ps=ls['tau_e_ps'],
                    LS_rmse=ls['rmse'], LS_converged=ls['converged'],
                    EMF_S2f=emf.get('S2f'), EMF_S2s=emf.get('S2s'), EMF_S2=emf.get('S2'),
                    EMF_tau_f_ps=emf.get('tau_f_ps'), EMF_tau_s_ps=emf.get('tau_s_ps'),
                    EMF_rmse=emf['rmse'], EMF_converged=emf['converged'],
                )

                # Isotropic rates
                if ls['converged']:
                    R1, R2, noe = nh_relaxation_rates_iso(
                        J_lipari_szabo, fmhz, self.tau_c_ps, ls['S2'], ls['tau_e_ps'])
                    row.update(LS_R1=R1, LS_R2=R2, LS_hetNOE=noe)
                else:
                    row.update(LS_R1=np.nan, LS_R2=np.nan, LS_hetNOE=np.nan)

                if emf['converged']:
                    R1, R2, noe = nh_relaxation_rates_iso(
                        J_extended_mf, fmhz, self.tau_c_ps,
                        emf['S2f'], emf['S2s'], emf['tau_f_ps'], emf['tau_s_ps'])
                    row.update(EMF_R1=R1, EMF_R2=R2, EMF_hetNOE=noe)
                else:
                    row.update(EMF_R1=np.nan, EMF_R2=np.nan, EMF_hetNOE=np.nan)

                # Anisotropic rates
                if self.anisotropic:
                    theta = self._nh_theta[pi]
                    row.update(self._aniso_common_cols(theta))
                    if ls['converged']:
                        R1, R2, noe = nh_relaxation_rates_aniso(
                            J_aniso_lipari_szabo, fmhz,
                            ls['S2'], ls['tau_e_ps'], self.D_par_s, self.D_perp_s, theta)
                        row.update(LS_aniso_R1=R1, LS_aniso_R2=R2, LS_aniso_hetNOE=noe)
                    else:
                        row.update(LS_aniso_R1=np.nan, LS_aniso_R2=np.nan,
                                   LS_aniso_hetNOE=np.nan)
                    if emf['converged']:
                        R1, R2, noe = nh_relaxation_rates_aniso(
                            J_aniso_extended_mf, fmhz,
                            emf['S2f'], emf['S2s'],
                            emf['tau_f_ps'], emf['tau_s_ps'],
                            self.D_par_s, self.D_perp_s, theta)
                        row.update(EMF_aniso_R1=R1, EMF_aniso_R2=R2,
                                   EMF_aniso_hetNOE=noe)
                    else:
                        row.update(EMF_aniso_R1=np.nan, EMF_aniso_R2=np.nan,
                                   EMF_aniso_hetNOE=np.nan)
                rows.append(row)

        df = pd.DataFrame(rows)
        fc = [c for c in df.columns if df[c].dtype == float]
        df[fc] = df[fc].round(6)

        n_pairs = len(self._nh_pairs)
        ls_ok   = int(df['LS_converged'].sum()) // len(self.fields)
        self._log(f"\n  NH: {n_pairs} residues, LS converged {ls_ok}/{n_pairs}")
        if len(self.fields) == 1:
            sub = df[df['field_MHz'] == self.fields[0]]
            self._log(f"  Mean at {self.fields[0]} MHz (LS iso): "
                      f"R1={sub['LS_R1'].mean():.3f}  "
                      f"R2={sub['LS_R2'].mean():.3f}  "
                      f"hetNOE={sub['LS_hetNOE'].mean():.3f}")
        return df

    def _build_methyl_df(self):
        """Assemble methyl ²H results into a DataFrame."""
        rows = []
        for mi, mg in enumerate(self._methyl_groups):
            ls  = self._methyl_ls[mi]
            emf = self._methyl_emf[mi]
            tau_Ri_ns = (self._methyl_tau_R_i_arr[mi] * 1e9
                         if self._methyl_tau_R_i_arr is not None else self.tau_c_ns)
            tau_eff_ps = tau_Ri_ns * 1000.0
            theta_mi   = (self._methyl_theta[mi]
                          if self._methyl_theta is not None else None)

            for fmhz in self.fields:
                row = dict(
                    probe='methyl', resid=mg['resid'], resname=mg['resname'],
                    methyl_name=mg['methyl_name'],
                    field_MHz=fmhz, tau_c_ns=self.tau_c_ns, tau_R_i_ns=tau_Ri_ns,
                    LS_S2_axis=ls['S2_axis'], LS_tau_f_ps=ls['tau_f_ps'],
                    LS_rmse=ls['rmse'], LS_converged=ls['converged'],
                    EMF_S2_axis=emf['S2_axis'],
                    EMF_tau_f_ps=emf.get('tau_f_ps'), EMF_tau_s_ps=emf.get('tau_s_ps'),
                    EMF_rmse=emf['rmse'], EMF_converged=emf['converged'],
                )

                # Isotropic ²H rates
                if ls['converged']:
                    Rdz, R3, Rdy = methyl_relaxation_rates_iso(
                        J_lipari_szabo, fmhz, tau_eff_ps,
                        ls['S2_axis'], ls['tau_f_ps'])
                    row.update(LS_R_Dz=Rdz, LS_R_3Dz2m2=R3, LS_R_Dy=Rdy)
                else:
                    row.update(LS_R_Dz=np.nan, LS_R_3Dz2m2=np.nan, LS_R_Dy=np.nan)

                if emf['converged']:
                    Rdz, R3, Rdy = methyl_relaxation_rates_iso(
                        J_lipari_szabo, fmhz, tau_eff_ps,
                        emf['S2_axis'], emf['tau_f_ps'])
                    row.update(EMF_R_Dz=Rdz, EMF_R_3Dz2m2=R3, EMF_R_Dy=Rdy)
                else:
                    row.update(EMF_R_Dz=np.nan, EMF_R_3Dz2m2=np.nan, EMF_R_Dy=np.nan)

                # Anisotropic ²H rates
                if self.anisotropic:
                    aniso_cols = self._aniso_common_cols(theta_mi)
                    aniso_cols['theta_deg'] = np.degrees(theta_mi)  # methyl axis θ
                    aniso_cols.pop('theta_deg')
                    row.update(
                        D_par_1e7=self.D_par_s * 1e-7,
                        D_perp_1e7=self.D_perp_s * 1e-7,
                        aniso_ratio=self.D_par_s / self.D_perp_s,
                        diffusion_axis_x=self.d_axis[0],
                        diffusion_axis_y=self.d_axis[1],
                        diffusion_axis_z=self.d_axis[2],
                        methyl_axis_theta_deg=np.degrees(theta_mi),
                    )
                    if ls['converged']:
                        Rdz, R3, Rdy = methyl_relaxation_rates_aniso(
                            J_aniso_lipari_szabo, fmhz,
                            ls['S2_axis'], ls['tau_f_ps'],
                            self.D_par_s, self.D_perp_s, theta_mi)
                        row.update(LS_aniso_R_Dz=Rdz, LS_aniso_R_3Dz2m2=R3,
                                   LS_aniso_R_Dy=Rdy)
                    else:
                        row.update(LS_aniso_R_Dz=np.nan, LS_aniso_R_3Dz2m2=np.nan,
                                   LS_aniso_R_Dy=np.nan)
                    if emf['converged']:
                        Rdz, R3, Rdy = methyl_relaxation_rates_aniso(
                            J_aniso_lipari_szabo, fmhz,
                            emf['S2_axis'], emf['tau_f_ps'],
                            self.D_par_s, self.D_perp_s, theta_mi)
                        row.update(EMF_aniso_R_Dz=Rdz, EMF_aniso_R_3Dz2m2=R3,
                                   EMF_aniso_R_Dy=Rdy)
                    else:
                        row.update(EMF_aniso_R_Dz=np.nan, EMF_aniso_R_3Dz2m2=np.nan,
                                   EMF_aniso_R_Dy=np.nan)
                rows.append(row)

        df = pd.DataFrame(rows)
        fc = [c for c in df.columns if df[c].dtype == float]
        df[fc] = df[fc].round(6)

        n_methyls = len(self._methyl_groups)
        ls_ok     = int(df['LS_converged'].sum()) // len(self.fields)
        self._log(f"\n  Methyls: {n_methyls} groups, LS converged {ls_ok}/{n_methyls}")
        if len(self.fields) == 1:
            sub = df[df['field_MHz'] == self.fields[0]]
            self._log(f"  Mean at {self.fields[0]} MHz (LS iso): "
                      f"S²_axis={sub['LS_S2_axis'].mean():.3f}  "
                      f"R(Dz)={sub['LS_R_Dz'].mean():.2f}  "
                      f"R(3Dz²−2)={sub['LS_R_3Dz2m2'].mean():.2f}  "
                      f"R(Dy)={sub['LS_R_Dy'].mean():.2f} s⁻¹")
        return df

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self):
        """
        Execute the full pipeline and return results.

        Returns
        -------
        df_backbone : pd.DataFrame or None
        df_methyl   : pd.DataFrame or None
        """
        self._setup_universe()

        df_backbone = None
        if self.do_backbone:
            self._log("\nFinding backbone NH pairs …")
            self._nh_pairs = self._find_nh_pairs()
            self._compute_acfs_backbone()
            self._fit_acfs_backbone()
            df_backbone = self._build_backbone_df()

        df_methyl = None
        if self.do_methyl:
            self._log("\nFinding methyl groups …")
            self._methyl_groups = self._find_methyl_groups()
            self._compute_acfs_methyl()
            self._fit_acfs_methyl()
            df_methyl = self._build_methyl_df()

        self._log(f"\n{'='*60}")
        self._log("Done.")
        return df_backbone, df_methyl


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_relaxation(
    topology,
    trajectory,
    field_MHz=500.0,
    tau_c_ns=None,
    anisotropic=False,
    D_par_1e7=None,
    D_perp_1e7=None,
    diffusion_axis=None,
    do_backbone=True,
    do_methyl=False,
    max_lag_fraction=0.5,
    acf_fit_fraction=0.5,
    verbose=True,
):
    """
    Thin wrapper around RelaxationCalculator for backward compatibility.

    Returns (df_backbone, df_methyl).
    """
    return RelaxationCalculator(
        topology=topology,
        trajectory=trajectory,
        field_MHz=field_MHz,
        tau_c_ns=tau_c_ns,
        anisotropic=anisotropic,
        D_par_1e7=D_par_1e7,
        D_perp_1e7=D_perp_1e7,
        diffusion_axis=diffusion_axis,
        do_backbone=do_backbone,
        do_methyl=do_methyl,
        max_lag_fraction=max_lag_fraction,
        acf_fit_fraction=acf_fit_fraction,
        verbose=verbose,
    ).run()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Calculate per-residue NMR relaxation rates from MD trajectory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("topology",   help="Topology file (PDB, PSF, GRO, …)")
    p.add_argument("trajectory", help="Trajectory file (XTC, DCD, TRR, …)")
    p.add_argument("--field", "-f", type=float, nargs="+", default=[500.0],
                   metavar="MHz",
                   help="¹H Larmor frequency in MHz (default: 500). Multiple allowed.")
    p.add_argument("--tau_c", type=float, default=None, metavar="ns",
                   help="Isotropic τc in ns. Estimated from trajectory if not given.")
    p.add_argument("--out", "-o", type=str, default=None,
                   help="Output CSV prefix (default: relaxation).")
    p.add_argument("--max_lag", type=float, default=0.5, metavar="FRACTION",
                   help="Max ACF lag as fraction of trajectory (default: 0.5).")
    p.add_argument("--fit_fraction", type=float, default=0.5, metavar="FRACTION",
                   help="Fraction of ACF used for fitting (default: 0.5).")
    p.add_argument("--methyl", action="store_true",
                   help="Also compute side-chain methyl ²H relaxation rates.")
    p.add_argument("--no-backbone", dest="backbone", action="store_false",
                   help="Skip backbone NH relaxation (methyl only).")
    p.set_defaults(backbone=True)
    p.add_argument("--anisotropic", action="store_true",
                   help="Enable axially symmetric diffusion tensor model.")
    p.add_argument("--Dpar", type=float, default=None, metavar="1e7_s-1",
                   help="D_∥ in 10⁷ s⁻¹. Estimated from trajectory if not given.")
    p.add_argument("--Dperp", type=float, default=None, metavar="1e7_s-1",
                   help="D_⊥ in 10⁷ s⁻¹. Estimated from trajectory if not given.")
    p.add_argument("--diffusion_axis", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Unique diffusion axis (lab frame). Estimated if not given.")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress output.")
    return p.parse_args()


def main():
    args   = _parse_args()
    prefix = args.out or "relaxation"

    calc = RelaxationCalculator(
        topology=args.topology,
        trajectory=args.trajectory,
        field_MHz=args.field,
        tau_c_ns=args.tau_c,
        anisotropic=args.anisotropic,
        D_par_1e7=args.Dpar,
        D_perp_1e7=args.Dperp,
        diffusion_axis=args.diffusion_axis,
        do_backbone=args.backbone,
        do_methyl=args.methyl,
        max_lag_fraction=args.max_lag,
        acf_fit_fraction=args.fit_fraction,
        verbose=not args.quiet,
    )
    df_bb, df_me = calc.run()

    if df_bb is not None:
        out_bb = f"{prefix}_backbone.csv" if args.methyl else f"{prefix}.csv"
        df_bb.to_csv(out_bb, index=False)
        print(f"Backbone NH results → {out_bb}")

    if df_me is not None:
        out_me = f"{prefix}_methyl.csv"
        df_me.to_csv(out_me, index=False)
        print(f"Methyl ²H results   → {out_me}")


if __name__ == "__main__":
    main()

