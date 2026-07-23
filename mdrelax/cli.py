"""Command-line interface for mdrelax.

    mdrelax-nh      TOPOLOGY TRAJECTORY [options]
    mdrelax-methyl  TOPOLOGY TRAJECTORY [options]
"""

import argparse

from .nh import NHRelaxation
from .methyl import MethylRelaxation
from .tumbling import MODELS


def nh_main(argv=None):
    p = argparse.ArgumentParser(
        prog="mdrelax-nh",
        description="Backbone 15N NH R1/R2/hetNOE from an MD trajectory.")
    p.add_argument("topology")
    p.add_argument("trajectory")
    p.add_argument("--field", "-f", type=float, nargs="+", default=[600.0],
                   metavar="MHz", help="1H Larmor frequency(ies) in MHz.")
    p.add_argument("--tau_c", type=float, default=None, metavar="ns",
                   help="Isotropic tau_c (ns). Estimated if omitted.")
    p.add_argument("--dt", type=float, default=None, metavar="ps",
                   help="Trajectory save interval (ps). Read from the file "
                        "header if omitted, which DCDs often get wrong.")
    p.add_argument("--step", type=int, default=1, help="Trajectory stride.")
    p.add_argument("--no-align", dest="align", action="store_false",
                   help="Do not align to CA (trajectory already fitted).")
    p.add_argument("--out", "-o", default="nh_rates.csv")
    p.add_argument("--quiet", "-q", action="store_true")
    a = p.parse_args(argv)
    df = NHRelaxation(a.topology, a.trajectory, fields_MHz=a.field,
                      tau_c_ns=a.tau_c, dt_ps=a.dt, traj_step=a.step,
                      align_traj=a.align, verbose=not a.quiet).run()
    df.to_csv(a.out, index=False)
    print(f"Wrote {a.out}")


def methyl_main(argv=None):
    p = argparse.ArgumentParser(
        prog="mdrelax-methyl",
        description="Side-chain methyl 2H relaxation from an MD trajectory.")
    p.add_argument("topology")
    p.add_argument("trajectory", help="Trajectory retaining overall tumbling.")
    p.add_argument("--fitted", default=None,
                   help="Trajectory with tumbling removed (methyl internal motion).")
    p.add_argument("--field", "-f", type=float, default=950.0, metavar="MHz")
    p.add_argument("--diffusion-model", choices=("auto",) + MODELS,
                   default="auto",
                   help="Rotational-diffusion model fit to the backbone tau_M. "
                        "'auto' (default) fits all three and picks between them "
                        "with an F-test.")
    p.add_argument("--ct_lim", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=None, metavar="ps",
                   help="Trajectory save interval (ps). Read from the file "
                        "header if omitted, which DCDs often get wrong.")
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--out", "-o", default="methyl_rates.csv")
    p.add_argument("--quiet", "-q", action="store_true")
    a = p.parse_args(argv)
    model = None if a.diffusion_model == "auto" else a.diffusion_model
    df = MethylRelaxation(a.topology, a.trajectory, trajectory_fitted=a.fitted,
                          field_MHz=a.field, diffusion_model=model,
                          ct_lim=a.ct_lim, dt_ps=a.dt, traj_step=a.step,
                          verbose=not a.quiet).run()
    df.to_csv(a.out, index=False)
    print(f"Wrote {a.out}")
