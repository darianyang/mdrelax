"""Backbone NH pipeline: deterministic snapshot + closeness to experiment."""

import warnings

import numpy as np
import pytest

from conftest import needs_t4l, T4L_PDB, T4L_XTC, DATA_NH

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def nh_df():
    from mdrelax import NHRelaxation
    return NHRelaxation(str(T4L_PDB), str(T4L_XTC), fields_MHz=600.0,
                        tau_c_ns=10.5, traj_step=1, max_lag_fraction=0.5,
                        fit_fraction=0.4, verbose=False).run()


@needs_t4l
def test_nh_output_shape_and_finiteness(nh_df):
    assert len(nh_df) == 158            # NH pairs (excl. Pro + N-terminus)
    assert nh_df["R1"].notna().all()
    assert (nh_df["R1"] > 0).all()
    assert (nh_df["R2"] > nh_df["R1"]).all()


@needs_t4l
def test_nh_deterministic_snapshot(nh_df):
    # Values computed with this pipeline on data/md-t4l at 600 MHz, tau_c=10.5 ns.
    # Tolerances catch regressions while allowing tiny platform fit differences.
    assert nh_df["R1"].mean() == pytest.approx(1.126, abs=0.03)
    assert nh_df["R2"].mean() == pytest.approx(13.08, abs=0.3)
    assert nh_df["hetNOE"].mean() == pytest.approx(0.859, abs=0.02)


@needs_t4l
def test_nh_estimated_tau_c_is_physical():
    # tau_c must be estimated from the un-aligned ACF (overall tumbling present).
    # Aligning first strips tumbling and estimate_tau_c just fits the S^2
    # plateau, giving a wildly inflated value (~90 ns on this fixture). Guard
    # that the estimate stays in a physically sensible band near T4L's true
    # ~10.5 ns and never regresses to that buggy aligned-ACF behavior.
    from mdrelax import NHRelaxation
    r = NHRelaxation(str(T4L_PDB), str(T4L_XTC), fields_MHz=600.0,
                     tau_c_ns=None, traj_step=1, max_lag_fraction=0.5,
                     fit_fraction=0.4, verbose=False)
    r.run()
    assert 6.0 < r.tau_c_ns < 15.0     # physical; excludes the ~90 ns bug


@needs_t4l
def test_nh_close_to_experiment(nh_df):
    exp = np.loadtxt(DATA_NH / "600MHz-R1R2NOE.dat")  # resid R1 e R2 e NOE e
    md = nh_df.set_index("resid")
    er = {int(r[0]): r for r in exp}
    common = [r for r in er if r in md.index]
    r1_md = np.array([md.loc[r, "R1"] for r in common])
    r1_ex = np.array([er[r][1] for r in common])
    r2_md = np.array([md.loc[r, "R2"] for r in common])
    r2_ex = np.array([er[r][3] for r in common])
    # aggregate agreement (per-residue detail is limited by the 10 ns trajectory)
    assert abs(np.mean(r1_md) - np.mean(r1_ex)) < 0.15
    assert abs(np.mean(r2_md) - np.mean(r2_ex)) < 1.5
