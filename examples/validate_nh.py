#!/usr/bin/env python
"""Backbone NH validation: MD (data/md-t4l) vs experiment (data/nh).

Runs the NH pipeline at 600 MHz with tau_c = 10.5 ns (from the experimental
Diso of T4L) and overlays R1, R2 and the heteronuclear NOE against the
experimental 600 MHz data.  Writes ``nh_validation.png``.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdrelax import NHRelaxation

ROOT = Path(__file__).resolve().parent.parent
PDB = ROOT / "data" / "md-t4l" / "sim1_dry.pdb"
XTC = ROOT / "data" / "md-t4l" / "segment_001.xtc"
EXP = ROOT / "data" / "nh" / "600MHz-R1R2NOE.dat"


def main():
    df = NHRelaxation(str(PDB), str(XTC), fields_MHz=600.0, tau_c_ns=10.5,
                      max_lag_fraction=0.5, fit_fraction=0.4).run()
    df = df[df.field_MHz == 600.0]
    exp = np.loadtxt(EXP)   # resid, R1, R1e, R2, R2e, NOE, NOEe

    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for a, (mcol, ecol, eecol, label) in zip(ax, [
            ("R1", 1, 2, r"$R_1$ (s$^{-1}$)"),
            ("R2", 3, 4, r"$R_2$ (s$^{-1}$)"),
            ("hetNOE", 5, 6, r"$^{15}$N-{$^1$H} NOE")]):
        a.errorbar(exp[:, 0], exp[:, ecol], yerr=exp[:, eecol], fmt="o",
                   ms=3, color="k", zorder=0, label="NMR (600 MHz)")
        a.plot(df["resid"], df[mcol], color="tab:red", lw=1, label="MD")
        a.set_ylabel(label)
    ax[0].legend(frameon=False)
    ax[-1].set_xlabel("Residue")
    fig.suptitle("T4L backbone NH: MD vs experiment (600 MHz)")
    fig.tight_layout()
    out = ROOT / "examples" / "nh_validation.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
