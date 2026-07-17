#!/usr/bin/env python
"""Maximum-entropy reweighting of methyl 2H rates against experiment.

The 1 us ff15ipq T4L trajectory is split into blocks; each block gives a set of
R(Dz), R(Dy), R(3Dz^2-2) rates for the 73 experimentally measured methyls
(``data/ch3/experimental/md.npy``, shape ``(3, 73, n_blocks)``).  With the force
field as the prior (uniform block weights), :class:`mdrelax.Reweighter` finds new
block weights that improve agreement with the measured NMR rates
(``nmr_rates.npy`` +/- ``nmr_errors.npy``) while staying as close to the prior as
possible -- the Bayesian / Maximum-Entropy (ABSURDer / BME) scheme.

``theta`` (the entropy weight) is chosen automatically from the L-curve knee of a
scan; pass ``--theta`` to override.  Writes ``ch3_reweight_validation.png``: the
three rate correlations (prior vs reweighted vs experiment), the L-curve, and the
resulting block-weight distribution.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdrelax import Reweighter, select_theta

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "data" / "ch3" / "experimental"
RATE_NAMES = ["R(Dz)", "R(Dy)", "R(3Dz$^2$-2)"]


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", type=float, default=950.0,
                    help="1H Larmor frequency in MHz (experiment: 950), label only")
    ap.add_argument("--theta", type=float, default=None,
                    help="entropy weight; default = L-curve knee of a theta scan")
    ap.add_argument("--out",
                    default=str(ROOT / "examples" / "ch3_reweight_validation.png"))
    return ap.parse_args()


def weighted_std(calc, w, avg):
    """Per-observable weighted standard deviation across blocks."""
    return np.sqrt(np.maximum(calc ** 2 @ w - avg ** 2, 0.0))


def main():
    args = parse_args()

    calc = np.load(EXP / "md.npy")               # (3, 73, n_blocks)
    exp = np.load(EXP / "nmr_rates.npy")         # (3, 73)
    err = np.load(EXP / "nmr_errors.npy")        # (3, 73)
    n_rates, n_methyls, n_blocks = calc.shape
    flat = calc.reshape(n_rates * n_methyls, n_blocks)

    rw = Reweighter(flat, exp.reshape(-1), err.reshape(-1))

    # --- choose theta from the L-curve, then reweight ----------------------
    scan = rw.scan()
    if args.theta is None:
        knee = select_theta(scan)
        theta = knee.theta
        print(f"L-curve knee: theta={theta:.3g}")
    else:
        theta = args.theta
    res, _ = rw.optimize(theta)

    prior_avg = (flat @ rw.w0).reshape(n_rates, n_methyls)
    rew_avg = res.avg.reshape(n_rates, n_methyls)
    rew_std = weighted_std(flat, res.weights, res.avg).reshape(n_rates, n_methyls)

    print(f"blocks={n_blocks}  theta={theta:.3g}  phi_eff={res.phi_eff:.3f}  "
          f"neff={res.neff:.0f}")
    print(f"chi2_red: prior={res.chi2_red_prior:.2f} -> reweighted={res.chi2_red:.2f}")

    # --- plot --------------------------------------------------------------
    fig, axd = plt.subplot_mosaic(
        [["R0", "R1", "R2"], ["lcurve", "lcurve", "weights"]],
        figsize=(12, 8), constrained_layout=True)

    for ri in range(n_rates):
        ax = axd[f"R{ri}"]
        x, xe = exp[ri], err[ri]
        lim = [0, max(x.max(), prior_avg[ri].max(), rew_avg[ri].max()) * 1.08]
        ax.plot(lim, lim, "k--", lw=0.8, zorder=0)
        ax.errorbar(x, prior_avg[ri], xerr=xe, fmt="o", ms=4, alpha=0.4,
                    color="0.5", elinewidth=0.5, capsize=0, label="prior (MD)")
        ax.errorbar(x, rew_avg[ri], yerr=rew_std[ri], fmt="o", ms=4, alpha=0.75,
                    color="tab:red", elinewidth=0.5, capsize=0, label="reweighted")
        chi2p = np.mean(((prior_avg[ri] - x) / xe) ** 2)
        chi2r = np.mean(((rew_avg[ri] - x) / xe) ** 2)
        ax.set_title(fr"{RATE_NAMES[ri]}" "\n"
                     fr"$\chi^2_{{\rm red}}$: {chi2p:.2f} -> {chi2r:.2f}")
        ax.set_xlabel("experiment (s$^{-1}$)")
        ax.set_ylabel("MD (s$^{-1}$)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        if ri == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")

    # L-curve: chi2_red vs effective fraction, knee marked
    ax = axd["lcurve"]
    phi = np.array([r.phi_eff for r in scan])
    chi = np.array([r.chi2_red for r in scan])
    ax.plot(phi, chi, "-o", ms=3, color="tab:blue")
    ax.plot(res.phi_eff, res.chi2_red, "*", ms=15, color="tab:red",
            label=fr"$\theta$={theta:.2g}")
    ax.axhline(res.chi2_red_prior, color="0.6", ls=":", lw=0.8, label="prior")
    ax.set_xlabel(r"effective fraction $\phi_{\rm eff}=e^{S_{\rm rel}}$")
    ax.set_ylabel(r"$\chi^2_{\rm red}$")
    ax.set_title("L-curve")
    ax.invert_xaxis()          # more reweighting (lower phi_eff) to the right
    ax.legend(frameon=False, fontsize=8)

    # block-weight distribution relative to the uniform prior
    ax = axd["weights"]
    ax.plot(res.weights * n_blocks, lw=0.6, color="tab:red")
    ax.axhline(1.0, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("block")
    ax.set_ylabel(r"$w_b\,/\,w^0_b$")
    ax.set_title("reweighted block weights")

    fig.suptitle(f"Methyl 2H maximum-entropy reweighting vs experiment - "
                 f"T4L, {n_methyls} methyls @ {args.field:.0f} MHz")
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
