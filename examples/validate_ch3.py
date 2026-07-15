#!/usr/bin/env python
"""Methyl 2H validation: pure-Python mdrelax vs the original ABSURDer results.

Reproduces the ABSURDer methyl relaxation rates from the shared 1 us ff15ipq
time-correlation functions and per-methyl tau_R, using only the mdrelax
fit/spectral-density/rate code (no GROMACS / pdbinertia / quadric), and compares
to ``results/rates.pkl``.  Writes ``ch3_validation.png`` (parity plots).
"""

import pickle
import bz2
import _pickle as cPickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdrelax import fitting as F, spectral_density as SD, rates as R
from mdrelax.constants import CHI_Q

ROOT = Path(__file__).resolve().parent.parent
FF = ROOT / "data-ch3-ff15ipq" / "tcf-1us"
SIMLEN, FITLEN, ACCURACY, CT_LIM = 10000, 5000, 80, 2
WD_RAD = 145.858415 * 2 * np.pi * 1e6
RATE_NAMES = ["R(Dz)", "R(Dy)", "R3", "R4"]


def load_bz2(p):
    with bz2.BZ2File(p, "rb") as f:
        return cPickle.load(f)


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def main(n_blocks=100):
    tcf = np.array(load_bz2(FF / "tcf_methyl" / "sim1_rot_trans.pbz2"))
    labels_ctor, tauR_ns = [], []
    for line in open(FF / "tau" / "tauR_methyl_specific"):
        if line.startswith("#"):
            continue
        p = line.split()
        labels_ctor.append(p[0])
        tauR_ns.append(float(p[1]))
    tauR_ns = np.array(tauR_ns)
    rates_ref = np.array(load_pkl(FF / "results" / "rates.pkl"))
    methyls_sorted = load_pkl(FF / "results" / "methyls.pkl")
    order = [labels_ctor.index(l) for l in methyls_sorted]

    n_methyl = (tcf.shape[2] - 1) // 3
    n_blocks = min(n_blocks, tcf.shape[0])
    exp_t = F.exp_time_points(SIMLEN, FITLEN, ACCURACY).astype(int)
    exp_freq = np.array([0.0, WD_RAD, 2 * WD_RAD])

    mine = np.zeros((4, n_methyl, n_blocks))
    for b in range(n_blocks):
        for k in range(n_methyl):
            ct = np.round(np.mean(tcf[b, exp_t, 1 + 3 * k:1 + 3 * k + 3], axis=1), 5)
            S2 = F.estimate_S2_tail(exp_t, ct, FITLEN, CT_LIM)
            fit = F.fit_internal_multiexp(exp_t.astype(float), ct, S2)
            tau_R_s = round(tauR_ns[k] * 1e-12 * 1000, 15)
            tred, w = F.reduced_times_and_weights(fit, tau_R_s)
            Jv = SD.J_multiexp(exp_freq, S2, tau_R_s, tred, w)
            mine[:, k, b] = R.deuterium_rates(Jv[0], Jv[1], Jv[2], CHI_Q)
    mine = mine[:, order, :]

    my_avg = mine.mean(axis=2)
    ref_avg = rates_ref[:, :, :n_blocks].mean(axis=2)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for ri in range(4):
        m, r = my_avg[ri], ref_avg[ri]
        lim = [0, max(m.max(), r.max()) * 1.05]
        ax[ri].plot(lim, lim, "k--", lw=0.8)
        ax[ri].scatter(r, m, s=12, alpha=0.6)
        cc = np.corrcoef(m, r)[0, 1]
        mad = np.mean(np.abs(m - r))
        ax[ri].set_title(f"{RATE_NAMES[ri]}\nr={cc:.4f}  MAD={mad:.2f} s$^{{-1}}$")
        ax[ri].set_xlabel("ABSURDer (s$^{-1}$)")
        ax[ri].set_ylabel("mdrelax (s$^{-1}$)")
        print(f"{RATE_NAMES[ri]:7s}: r={cc:.5f}  MAD={mad:.3f}  "
              f"mean_ref={r.mean():.2f}")
    fig.suptitle(f"Methyl 2H rates: mdrelax vs ABSURDer "
                 f"({n_blocks} blocks, 1 us ff15ipq T4L)")
    fig.tight_layout()
    out = ROOT / "examples" / "ch3_validation.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
