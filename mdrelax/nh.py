"""Backbone amide 15N-1H relaxation from an MD trajectory.

Pipeline: align the trajectory to remove overall tumbling, build per-residue N-H
internal correlation functions, fit them with Lipari-Szabo / Extended Model-Free,
then combine with an overall tau_c to get R1, R2 and the heteronuclear NOE.
"""

import warnings

import numpy as np
import pandas as pd

import MDAnalysis as mda
from MDAnalysis.analysis import align

from . import acf, fitting, geometry, tumbling
from .spectral_density import J_lipari_szabo, J_extended_mf
from .rates import nh_rates_from_J

warnings.filterwarnings("ignore", category=UserWarning,
                        module="MDAnalysis")


class NHRelaxation:
    """Backbone 15N NH relaxation-rate calculator.

    Parameters
    ----------
    topology, trajectory : str    input structure and trajectory
    fields_MHz : float or list    1H Larmor frequency(ies) in MHz
    tau_c_ns : float or None       isotropic tau_c (ns); estimated if None
    traj_step : int                stride when loading the trajectory
    align_traj : bool              align to CA to remove tumbling (default True)
    max_lag_fraction : float       ACF length as a fraction of the trajectory
    fit_fraction : float           fraction of the ACF used for model fitting
    """

    def __init__(self, topology, trajectory, fields_MHz=600.0, tau_c_ns=None,
                 traj_step=1, align_traj=True, max_lag_fraction=0.5,
                 fit_fraction=0.5, verbose=True):
        self.topology = topology
        self.trajectory = trajectory
        self.fields = [fields_MHz] if np.isscalar(fields_MHz) else list(fields_MHz)
        self.tau_c_ns = tau_c_ns
        self.traj_step = traj_step
        self.align_traj = align_traj
        self.max_lag_fraction = max_lag_fraction
        self.fit_fraction = fit_fraction
        self.verbose = verbose
        self.u = None

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _load(self):
        self.u = mda.Universe(self.topology, self.trajectory, in_memory=True,
                              in_memory_step=self.traj_step)
        self.dt_ps = float(self.u.trajectory.dt)
        self.n_frames = len(self.u.trajectory)
        self._log(f"Loaded {self.n_frames} frames, dt={self.dt_ps} ps "
                  f"({self.n_frames * self.dt_ps / 1000:.2f} ns)")
        if self.align_traj:
            ref = mda.Universe(self.topology)
            align.AlignTraj(self.u, ref, select="protein and name CA",
                            in_memory=True).run()

    def run(self):
        """Run the pipeline and return a pandas DataFrame of per-residue rates."""
        self._load()
        pairs = geometry.find_nh_pairs(self.u)
        self._log(f"Found {len(pairs)} NH pairs")

        vecs = geometry.collect_vectors(self.u, pairs, step=None)
        max_lag = max(2, int(self.n_frames * self.max_lag_fraction))
        acfs = acf.p2_acf_many(vecs, max_lag=max_lag)   # (max_lag, n_pairs)

        if self.tau_c_ns is None:
            self.tau_c_ns = tumbling.estimate_tau_c(acfs, self.dt_ps,
                                                    self.fit_fraction)
            self._log(f"Estimated tau_c = {self.tau_c_ns:.3f} ns")
        tau_c_ps = self.tau_c_ns * 1000.0

        n_fit = max(10, int(max_lag * self.fit_fraction))
        t_fit = np.arange(n_fit) * self.dt_ps

        rows = []
        for pi, p in enumerate(pairs):
            c = acfs[:n_fit, pi]
            ls = fitting.fit_lipari_szabo(c, t_fit, tau_c_ps)
            emf = fitting.fit_extended_mf(c, t_fit, tau_c_ps)
            for f in self.fields:
                row = dict(resid=p['resid'], resname=p['resname'], field_MHz=f,
                           tau_c_ns=self.tau_c_ns, S2=ls['S2'],
                           tau_e_ps=ls['tau_e_ps'], LS_rmse=ls['rmse'],
                           EMF_S2=emf['S2'], EMF_rmse=emf['rmse'])
                if ls['converged']:
                    R1, R2, noe = nh_rates_from_J(
                        lambda w: J_lipari_szabo(w, ls['S2'], ls['tau_e_ps'],
                                                 tau_c_ps), f)
                    row.update(R1=R1, R2=R2, hetNOE=noe)
                else:
                    row.update(R1=np.nan, R2=np.nan, hetNOE=np.nan)
                if emf['converged']:
                    R1e, R2e, noee = nh_rates_from_J(
                        lambda w: J_extended_mf(w, emf['S2f'], emf['S2s'],
                                                emf['tau_f_ps'], emf['tau_s_ps'],
                                                tau_c_ps), f)
                    row.update(EMF_R1=R1e, EMF_R2=R2e, EMF_hetNOE=noee)
                else:
                    row.update(EMF_R1=np.nan, EMF_R2=np.nan, EMF_hetNOE=np.nan)
                rows.append(row)

        df = pd.DataFrame(rows)
        self.results = df
        if self.verbose and len(self.fields) >= 1:
            sub = df[df['field_MHz'] == self.fields[0]]
            self._log(f"Mean @ {self.fields[0]} MHz (LS): R1={sub['R1'].mean():.3f} "
                      f"R2={sub['R2'].mean():.3f} NOE={sub['hetNOE'].mean():.3f}")
        return df
