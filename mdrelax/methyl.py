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

from . import acf, fitting, geometry, tumbling
from .spectral_density import J_multiexp
from .rates import deuterium_rates, methyl_omega_D
from .constants import CHI_Q

warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")


class MethylRelaxation:
    """Methyl 2H relaxation-rate calculator (R_Dz, R_Dy, R3, R4).

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
    ct_lim : float                   tail fraction for S2 (2 = last half)
    n_free : int                     free amplitudes in the internal fit (5)
    accuracy : int                   exponential-sampling density (80)
    max_lag_fraction : float         ACF length as a fraction of the trajectory
    """

    def __init__(self, topology, trajectory, trajectory_fitted=None,
                 field_MHz=950.0, tau_R_ns=None, ct_lim=2.0, n_free=5,
                 accuracy=80, max_lag_fraction=0.5, traj_step=1, verbose=True):
        self.topology = topology
        self.trajectory = trajectory
        self.trajectory_fitted = trajectory_fitted
        self.field_MHz = field_MHz
        self.tau_R_ns = tau_R_ns
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
    def _compute_tau_R(self, groups):
        """Per-methyl tau_R (ns) via backbone tau_M + axial diffusion tensor."""
        u = mda.Universe(self.topology, self.trajectory, in_memory=True,
                         in_memory_step=self.traj_step)
        dt = float(u.trajectory.dt)
        n_frames = len(u.trajectory)
        pairs = geometry.find_nh_pairs(u)
        nh_vecs_traj = geometry.collect_vectors(u, pairs)     # tumbling present
        max_lag = max(2, int(n_frames * self.max_lag_fraction))
        nh_acfs = acf.p2_acf_many(nh_vecs_traj, max_lag=max_lag, normalize=False)

        # per-residue tau_M from NH TCF (single block over the whole trajectory)
        tcf = np.column_stack([np.arange(max_lag) * dt, nh_acfs])
        l_block_ps = n_frames * dt
        tauM = tumbling.fit_backbone_tauM(
            tcf, len(pairs), l_block_ps=l_block_ps,
            start_tau_ps=max(l_block_ps / 100.0, 5000.0),
            accuracy=100)

        u.trajectory[0]
        nh_ref = np.array([p['H'].position - p['N'].position for p in pairs])
        fit = tumbling.fit_axial_diffusion(tauM, nh_ref)
        self._log(f"  Diso={fit['Diso']*1e-7:.3f}e7 s^-1  Dpar/Dper={fit['Dratio']:.3f}"
                  f"  tau_c={fit['tau_c_ns']:.2f} ns")

        u.trajectory[0]
        cc = np.array([g['C'].position - g['C_axis'].position for g in groups])
        tau_R = tumbling.methyl_tau_R(cc, fit['Diso'], fit['Dpar'],
                                      fit['Dper'], fit['axis'])
        return tau_R, fit

    # ── internal ACFs ───────────────────────────────────────────────────────
    def _internal_acfs(self, groups):
        if self.trajectory_fitted is not None:
            u = mda.Universe(self.topology, self.trajectory_fitted,
                             in_memory=True, in_memory_step=self.traj_step)
        else:
            u = mda.Universe(self.topology, self.trajectory, in_memory=True,
                             in_memory_step=self.traj_step)
            ref = mda.Universe(self.topology)
            align.AlignTraj(u, ref, select="protein and name CA",
                            in_memory=True).run()
        dt = float(u.trajectory.dt)
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
        u0 = mda.Universe(self.topology, self.trajectory)
        groups = geometry.find_methyl_groups(u0)
        self._log(f"Found {len(groups)} methyl groups")

        acfs, dt, n_frames, labels = self._internal_acfs(groups)
        max_lag = acfs.shape[0]

        if self.tau_R_ns is not None:
            tau_R = np.array([self.tau_R_ns[l] for l in labels])
        else:
            self._log("Estimating tumbling (backbone tau_M -> axial D tensor)...")
            groups_t = geometry.find_methyl_groups(
                mda.Universe(self.topology, self.trajectory))
            tau_R_map = dict(zip([g['label'] for g in groups_t],
                                 self._compute_tau_R(groups_t)[0]))
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
            tred, w = fitting.reduced_times_and_weights(fit, tau_R_s)
            Jv = J_multiexp(exp_freq, S2, tau_R_s, tred, w)
            R_Dz, R_Dy, R3, R4 = deuterium_rates(Jv[0], Jv[1], Jv[2], CHI_Q)
            rows.append(dict(label=label,
                             resid=int(label.split('-')[0][3:]) if '-' in label else -1,
                             field_MHz=self.field_MHz, S2_axis=S2,
                             tau_R_ns=tau_R[mi], R_Dz=R_Dz, R_Dy=R_Dy,
                             R3=R3, R4=R4))
        df = pd.DataFrame(rows)
        self.results = df
        self._log(f"Mean @ {self.field_MHz} MHz: S2={df['S2_axis'].mean():.3f} "
                  f"R_Dz={df['R_Dz'].mean():.2f} R_Dy={df['R_Dy'].mean():.2f} s^-1")
        return df
