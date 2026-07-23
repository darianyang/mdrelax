"""Side-chain methyl 2H relaxation from an MD trajectory (ABSURDer method).

Pipeline (pure Python, no GROMACS / pdbinertia / quadric):

1. Internal methyl motion: align the trajectory (remove overall tumbling), build
   the P2 correlation function of each C-H bond, average the three per methyl.
2. Overall tumbling: from the un-aligned trajectory, fit backbone tau_M, then an
   axially symmetric diffusion tensor, giving a methyl-specific tau_R
   (:mod:`mdrelax.tumbling`).  A precomputed tau_R can be supplied instead.
3. Spectral density: fit the internal C(t) to a multi-exponential, reintroduce
   tau_R, evaluate J at 0, wD, 2wD and form the 2H quadrupolar rates.
"""

import warnings

import numpy as np
import pandas as pd

import MDAnalysis as mda
from MDAnalysis.analysis import align

from . import acf, fitting, geometry, timestep, tumbling
from .spectral_density import J_multiexp
from .rates import deuterium_rates, methyl_omega_D
from .constants import CHI_Q

warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")


class MethylRelaxation:
    """Methyl 2H relaxation-rate calculator (R_Dz, R_Dy, R_3Dz2).

    Parameters
    ----------
    topology : str
    trajectory : str
        Trajectory used for backbone tau_M (should retain overall tumbling,
        e.g. a pbc-corrected but not rotationally fitted trajectory).
    trajectory_fitted : str, optional
        Trajectory with overall tumbling already removed (rot+trans fit); used
        for the methyl internal correlation functions.  If omitted, ``trajectory``
        is aligned to CA internally to remove tumbling.
    field_MHz : float                1H Larmor frequency (MHz)
    tau_R_ns : dict or None          {methyl_label: tau_R_ns} to override the
                                     diffusion-tensor estimate
    diffusion_model : {None, 'iso', 'axial', 'aniso'}
        Rotational-diffusion model fit to the backbone tau_M when tau_R is not
        supplied.  Which one suits a protein depends on its shape: 'iso' for a
        globular one, 'axial' (what ABSURDer assumes) for a prolate/oblate one,
        'aniso' when no two principal values are alike.  The default None fits
        all three and F-tests between them
        (:func:`mdrelax.tumbling.select_diffusion_model`), which costs well under
        a second; pass an explicit model to force one.  Unused when ``tau_R_ns``
        is given.
    dt_ps : float or None
        Save interval of the trajectory in ps, before ``traj_step``. Default
        None reads it from the file header, which DCDs routinely get wrong --
        see :mod:`mdrelax.timestep`. Supply it whenever you know it.
    ct_lim : float                   tail fraction for S2 (2 = last half)
    n_free : int                     free amplitudes in the internal fit (5)
    accuracy : int                   exponential-sampling density (80)
    max_lag_fraction : float         ACF length as a fraction of the trajectory

    Attributes set by :meth:`run`
    -----------------------------
    results : pandas.DataFrame       the per-methyl rates
    diffusion : dict or None         the diffusion-tensor fit that was used (see
                                     :func:`mdrelax.tumbling.fit_diffusion`);
                                     None when ``tau_R_ns`` was supplied and no
                                     fit was done
    diffusion_trials : dict or None  {model: fit} for all three models, when they
                                     were fit to select one; None otherwise
    """

    def __init__(self, topology, trajectory, trajectory_fitted=None,
                 field_MHz=950.0, tau_R_ns=None, diffusion_model=None,
                 dt_ps=None, ct_lim=2.0, n_free=5, accuracy=80,
                 max_lag_fraction=0.5, traj_step=1, verbose=True):
        if diffusion_model is not None and diffusion_model not in tumbling.MODELS:
            raise ValueError("diffusion_model must be None (auto-select) or one "
                             f"of {tumbling.MODELS}, got {diffusion_model!r}")
        self.topology = topology
        self.trajectory = trajectory
        self.trajectory_fitted = trajectory_fitted
        self.field_MHz = field_MHz
        self.tau_R_ns = tau_R_ns
        self.diffusion_model = diffusion_model
        self.dt_ps = dt_ps
        self.ct_lim = ct_lim
        self.n_free = n_free
        self.accuracy = accuracy
        self.max_lag_fraction = max_lag_fraction
        self.traj_step = traj_step
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ── tumbling ────────────────────────────────────────────────────────────
    def _compute_tau_R(self):
        """Per-methyl tau_R via backbone tau_M + an axial diffusion tensor.

        Locates the NH pairs and methyl groups on its own universe, built from
        the tumbling-retaining trajectory, so every vector is read from the same
        universe and the same frame.

        Returns
        -------
        ({methyl_label: tau_R_ns}, fit) : the per-methyl tumbling times and the
            diffusion-tensor fit (:func:`mdrelax.tumbling.fit_diffusion`).
        """
        u = mda.Universe(self.topology, self.trajectory, in_memory=True,
                         in_memory_step=self.traj_step)
        dt = timestep.resolve_dt_ps(u, self.dt_ps, self.traj_step)
        n_frames = len(u.trajectory)
        pairs = geometry.find_nh_pairs(u)
        nh_vecs_traj = geometry.collect_vectors(u, pairs)     # tumbling present
        max_lag = max(2, int(n_frames * self.max_lag_fraction))
        nh_acfs = acf.p2_acf_many(nh_vecs_traj, max_lag=max_lag, normalize=False)

        # per-residue tau_M from NH TCF (single block over the whole trajectory)
        tcf = np.column_stack([np.arange(max_lag) * dt, nh_acfs])
        l_block_ps = n_frames * dt
        tauM = tumbling.fit_backbone_tauM(
            tcf, len(pairs), l_block_ps=l_block_ps, dt_ps=dt,
            start_tau_ps=max(l_block_ps / 100.0, 5000.0),
            accuracy=100)

        # reference geometry, all from frame 0 (collect_vectors left us at the end)
        u.trajectory[0]
        groups = geometry.find_methyl_groups(u)
        nh_ref = np.array([p['H'].position - p['N'].position for p in pairs])
        cc = np.array([g['C'].position - g['C_axis'].position for g in groups])

        if self.diffusion_model is None:
            fit, self.diffusion_trials = tumbling.select_diffusion_model(
                tauM, nh_ref)
            for m in tumbling.MODELS:
                self._log(f"    {tumbling.describe(self.diffusion_trials[m])}")
            for simple, cplx, F, p, ok in fit["selection"]:
                self._log(f"    F({simple}->{cplx})={F:.3f} p={p:.3g} "
                          f"-> {'accept' if ok else 'reject'} {cplx}")
        else:
            fit = tumbling.fit_diffusion(tauM, nh_ref, model=self.diffusion_model)
            self.diffusion_trials = None
        self._log(f"  using {tumbling.describe(fit)}")

        tau_R = tumbling.tau_R_from_fit(cc, fit)
        return dict(zip([g['label'] for g in groups], tau_R)), fit

    # ── internal ACFs ───────────────────────────────────────────────────────
    def _internal_acfs(self):
        """Methyl C-H P2 ACFs on the tumbling-free trajectory.

        Finds the methyl groups itself: MDAnalysis atoms belong to the universe
        they came from, so groups located elsewhere cannot be used here.
        """
        if self.trajectory_fitted is not None:
            u = mda.Universe(self.topology, self.trajectory_fitted,
                             in_memory=True, in_memory_step=self.traj_step)
        else:
            u = mda.Universe(self.topology, self.trajectory, in_memory=True,
                             in_memory_step=self.traj_step)
            ref = mda.Universe(self.topology)
            align.AlignTraj(u, ref, select="protein and name CA",
                            in_memory=True).run()
        dt = timestep.resolve_dt_ps(u, self.dt_ps, self.traj_step)
        n_frames = len(u.trajectory)
        # re-find groups in this universe (atom objects are universe-specific)
        groups_u = geometry.find_methyl_groups(u)
        ch = geometry.collect_methyl_ch_vectors(u, groups_u)   # (F, M, 3, 3)
        max_lag = max(2, int(n_frames * self.max_lag_fraction))
        n_meth = ch.shape[1]
        acfs = np.zeros((max_lag, n_meth))
        for mi in range(n_meth):
            s = np.zeros(max_lag)
            for hj in range(3):
                s += acf.p2_acf_fft(ch[:, mi, hj, :], max_lag=max_lag)
            acfs[:, mi] = s / 3.0
        labels_u = [g['label'] for g in groups_u]
        return acfs, dt, n_frames, labels_u

    # ── main ────────────────────────────────────────────────────────────────
    def run(self):
        """Run the pipeline; return a per-methyl DataFrame of 2H rates."""
        acfs, dt, n_frames, labels = self._internal_acfs()
        max_lag = acfs.shape[0]
        self._log(f"Found {len(labels)} methyl groups")

        if self.tau_R_ns is not None:
            tau_R = np.array([self.tau_R_ns[l] for l in labels])
            self.diffusion = self.diffusion_trials = None
        else:
            model = self.diffusion_model or "auto-selected"
            self._log(f"Estimating tumbling (backbone tau_M -> {model} D tensor)...")
            tau_R_map, self.diffusion = self._compute_tau_R()
            tau_R = np.array([tau_R_map[l] for l in labels])

        # exponentially spaced sampling (frames -> ps)
        exp_t = fitting.exp_time_points(n_frames, max_lag, self.accuracy)
        times_ps = exp_t.astype(float) * dt
        omega_D = methyl_omega_D(self.field_MHz)
        exp_freq = np.array([0.0, omega_D, 2 * omega_D])

        rows = []
        for mi, label in enumerate(labels):
            ct = np.round(acfs[exp_t, mi], 5)
            S2 = fitting.estimate_S2_tail(exp_t, ct, max_lag, self.ct_lim)
            fit = fitting.fit_internal_multiexp(times_ps, ct, S2, n_free=self.n_free)
            tau_R_s = tau_R[mi] * 1e-9
            t_red, w = fitting.reduced_times_and_weights(fit, tau_R_s)
            Jv = J_multiexp(exp_freq, S2, tau_R_s, t_red, w)
            R_Dz, R_Dy, R_3Dz2 = deuterium_rates(Jv[0], Jv[1], Jv[2], CHI_Q)
            rows.append(dict(label=label,
                             resid=int(label.split('-')[0][3:]) if '-' in label else -1,
                             field_MHz=self.field_MHz, S2_axis=S2,
                             tau_R_ns=tau_R[mi], R_Dz=R_Dz, R_Dy=R_Dy,
                             R_3Dz2=R_3Dz2))
        df = pd.DataFrame(rows)
        self.results = df
        self._log(f"Mean @ {self.field_MHz} MHz: S2={df['S2_axis'].mean():.3f} "
                  f"R_Dz={df['R_Dz'].mean():.2f} R_Dy={df['R_Dy'].mean():.2f} s^-1")
        return df
