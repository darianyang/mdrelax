"""Methyl 2H relaxation: numeric port vs ABSURDer rates.pkl + pipeline smoke test."""

import warnings

import numpy as np
import pytest

from conftest import needs_ff15ipq, FF15IPQ

warnings.filterwarnings("ignore")

# ABSURDer run settings (algorithm/experiment parameters, not data properties);
# the time axis is derived from the TCF array instead of being hardcoded.
ACCURACY, CT_LIM = 80, 2
WD_RAD = 145.858415 * 2 * np.pi * 1e6      # 2H Larmor at 950 MHz 1H (ABSURDer)


def _rates_for_block(tcf, block, exp_t, exp_freq, tauR_ns, n_methyl, fitlen):
    from mdrelax import fitting as F, spectral_density as SD, rates as R
    from mdrelax.constants import CHI_Q
    out = np.zeros((3, n_methyl))
    for k in range(n_methyl):
        ct = np.round(np.mean(tcf[block, exp_t, 1 + 3 * k:1 + 3 * k + 3], axis=1), 5)
        S2 = F.estimate_S2_tail(exp_t, ct, fitlen, CT_LIM)
        fit = F.fit_internal_multiexp(exp_t.astype(float), ct, S2)
        tau_R_s = round(tauR_ns[k] * 1e-12 * 1000, 15)
        tred, w = F.reduced_times_and_weights(fit, tau_R_s)
        Jv = SD.J_multiexp(exp_freq, S2, tau_R_s, tred, w)
        out[:, k] = R.deuterium_rates(Jv[0], Jv[1], Jv[2], CHI_Q)
    return out


@needs_ff15ipq
def test_ch3_rates_reproduce_absurder(ff15ipq_methyl_tcf, ff15ipq_tauR,
                                      ff15ipq_rates):
    """Fit + J + rate port reproduces ABSURDer rates.pkl on shared TCFs."""
    from mdrelax import fitting as F
    tcf = ff15ipq_methyl_tcf
    labels_ctor, tauR_ns = ff15ipq_tauR
    rates_ref, methyls_sorted = ff15ipq_rates
    n_methyl = (tcf.shape[2] - 1) // 3
    # TCFs are stored over t = 0..fit_length inclusive, with fit_length =
    # block_length/2, so the block length is twice the stored span.
    fitlen = tcf.shape[1] - 1                    # 5000
    simlen = 2 * fitlen                          # 10000 (block length)
    exp_t = F.exp_time_points(simlen, fitlen, ACCURACY).astype(int)
    exp_freq = np.array([0.0, WD_RAD, 2 * WD_RAD])
    order = [labels_ctor.index(l) for l in methyls_sorted]

    for block in (0, 1, 2):
        mine = _rates_for_block(tcf, block, exp_t, exp_freq, tauR_ns, n_methyl,
                                fitlen)
        mine = mine[:, order]                    # ctor -> sorted (rates.pkl order)
        ref = rates_ref[:3, :, block]            # R(Dz), R(Dy), R(3Dz^2-2)
        for ri in range(3):
            rel = np.abs(mine[ri] - ref[ri]) / np.abs(ref[ri])
            assert np.median(rel) < 0.02         # typical methyl within 2 %
            assert np.corrcoef(mine[ri], ref[ri])[0, 1] > 0.999


@needs_ff15ipq
def test_ch3_rate_definitions_ordered(ff15ipq_rates):
    """R_Dy (transverse) is the largest of the 2H rates."""
    rates_ref, _ = ff15ipq_rates
    mean_rates = rates_ref[:3].mean(axis=(1, 2))  # R_Dz, R_Dy, R(3Dz^2-2)
    assert mean_rates[1] == max(mean_rates)       # R_Dy largest


@needs_ff15ipq
def test_methyl_pipeline_smoke():
    """Full MethylRelaxation on the 10 ns trajectory with reference tau_R."""
    from mdrelax import MethylRelaxation
    base = FF15IPQ.parent
    tauR = {}
    with open(FF15IPQ / "tau" / "tauR_methyl_specific") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split()
            tauR[p[0]] = float(p[1])
    mr = MethylRelaxation(
        str(base / "initial.pdb"), str(base / "sim1-10ns_nopbc.xtc"),
        trajectory_fitted=str(base / "sim1-10ns_nopbc_rot_trans.xtc"),
        field_MHz=950.0, tau_R_ns=tauR, verbose=False)
    df = mr.run()
    assert len(df) == 101
    for col in ("R_Dz", "R_Dy", "R_3Dz2"):
        assert np.isfinite(df[col]).all()
        assert (df[col] > 0).all()
    assert (df["R_Dy"] > df["R_Dz"]).all()
