#!/usr/bin/env python
"""Methyl 2H validation: mdrelax rates *from an MD trajectory* vs experimental NMR.

This is the full user-facing pipeline: give it a trajectory and it computes the
methyl C-H correlation functions, fits them, builds J(omega) and returns the three
experimentally measured deuterium rates: R(Dz), R(Dy), R(3Dz^2-2) - which are
then compared to the measured NMR values in ``data/ch3/experimental/``.

Nothing is precomputed here: the TCFs are calculated inside
:class:`mdrelax.MethylRelaxation`.  Compare with ``validate_ch3.py``, which
instead reuses ABSURDer's *precomputed* TCFs in order to isolate the numeric
fit/J/rate port from any difference in how the TCFs themselves are built.

tau_R note
----------
The overall tumbling time (tau_c ~ 13 ns for T4L) can only be measured from a
trajectory much longer than tau_c itself.  The bundled 10 ns trajectory is the
ABSURDer *methyl* block length - right for the fast internal motion, far too
short for tumbling (it yields tau_c ~ 8.7 ns instead of ~12.9 ns).  So by default
tau_R is read from the 1 us backbone fit, exactly as ABSURDer does
(``--lblocks_m 10000`` for methyls, ``--lblocks_bb 1000000`` for the backbone).
Pass ``--estimate-tau-R`` to instead derive it from the given trajectory, which is
the right choice when your trajectory is long compared with tau_c.

Everything else (block/fit lengths, sampling) is derived from the trajectory
itself by MethylRelaxation; nothing about the time axis is hardcoded.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdrelax import MethylRelaxation

ROOT = Path(__file__).resolve().parent.parent
FF = ROOT / "data" / "ch3-ff15ipq"
EXP = ROOT / "data" / "ch3" / "experimental"
RATE_COLS = ["R_Dz", "R_Dy", "R_3Dz2"]
RATE_NAMES = ["R(Dz)", "R(Dy)", "R(3Dz$^2$-2)"]


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def read_tauR(path):
    """ABSURDer per-methyl tau_R file -> {label: tau_R_ns}."""
    tau = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split()
            tau[p[0]] = float(p[1])
    return tau


def norm(label):
    """ILE delta methyl is 'CDHD' in the NMR list but 'CD1HD1' in the topology."""
    return label.replace("-CDHD", "-CD1HD1")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology", default=str(FF / "initial.pdb"))
    ap.add_argument("--traj", default=str(FF / "sim1-10ns_nopbc.xtc"),
                    help="trajectory retaining overall tumbling (for tau_M)")
    ap.add_argument("--fitted", default=str(FF / "sim1-10ns_nopbc_rot_trans.xtc"),
                    help="rot+trans-fitted trajectory (methyl internal motion)")
    ap.add_argument("--field", type=float, default=950.0,
                    help="1H Larmor frequency in MHz (experiment: 950)")
    ap.add_argument("--tau-R-file", default=str(FF / "tcf-1us" / "tau" /
                                                "tauR_methyl_specific"),
                    help="per-methyl tau_R from a long-trajectory backbone fit")
    ap.add_argument("--estimate-tau-R", action="store_true",
                    help="estimate tau_R from --traj instead of --tau-R-file "
                         "(only valid if the trajectory is much longer than tau_c)")
    ap.add_argument("--out", default=str(ROOT / "examples" / "ch3_exp_validation.png"))
    return ap.parse_args()


def main():
    args = parse_args()

    tau_R_ns = None if args.estimate_tau_R else read_tauR(args.tau_R_file)
    if tau_R_ns is None:
        print("Estimating tau_R from the trajectory (needs traj >> tau_c)")
    else:
        print(f"Using tau_R from {Path(args.tau_R_file).name} "
              f"(1 us backbone fit; the 10 ns traj is too short for tumbling)")

    # --- the actual pipeline: trajectory in, rates out ---------------------
    mr = MethylRelaxation(args.topology, args.traj,
                          trajectory_fitted=args.fitted,
                          field_MHz=args.field, tau_R_ns=tau_R_ns, verbose=True)
    df = mr.run()
    md = {row.label: row for row in df.itertuples()}

    # --- experimental rates ------------------------------------------------
    rex = np.load(EXP / "nmr_rates.npy")            # (3, 73)
    eex = np.load(EXP / "nmr_errors.npy")           # (3, 73)
    methyls_nmr = load_pkl(ROOT / "data" / "ch3" / "methyls_nmr.pkl")

    missing = [l for l in methyls_nmr if norm(l) not in md]
    if missing:
        raise SystemExit(f"{len(missing)} NMR methyls absent from the MD set: "
                         f"{missing[:5]}")
    md_sub = np.array([[getattr(md[norm(l)], c) for l in methyls_nmr]
                       for c in RATE_COLS])          # (3, 73)

    # --- plot --------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for ri in range(3):
        x, xe, y = rex[ri], eex[ri], md_sub[ri]
        lim = [0, max(x.max(), y.max()) * 1.08]
        ax[ri].plot(lim, lim, "k--", lw=0.8, zorder=0)
        ax[ri].errorbar(x, y, xerr=xe, fmt="o", ms=4, alpha=0.6,
                        elinewidth=0.6, capsize=0)
        cc = np.corrcoef(x, y)[0, 1]
        rmsd = np.sqrt(np.mean((x - y) ** 2))
        ax[ri].set_title(f"{RATE_NAMES[ri]}\nr={cc:.3f}  RMSD={rmsd:.2f} s$^{{-1}}$")
        ax[ri].set_xlabel("experiment (s$^{-1}$)")
        ax[ri].set_ylabel("mdrelax MD (s$^{-1}$)")
        ax[ri].set_xlim(lim)
        ax[ri].set_ylim(lim)
        print(f"{RATE_NAMES[ri]:12s}: r={cc:.4f}  RMSD={rmsd:.3f}  "
              f"mean_MD={y.mean():.2f}  mean_exp={x.mean():.2f}")
    fig.suptitle(f"Methyl 2H rates from MD trajectory (mdrelax) vs experiment "
                 f"- T4L, {len(methyls_nmr)} methyls @ {args.field:.0f} MHz")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
