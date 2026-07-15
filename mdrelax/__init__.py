"""mdrelax — NMR spin-relaxation rates from MD simulation trajectories.

Backbone amide 15N-1H (R1, R2, hetNOE) via :class:`~mdrelax.nh.NHRelaxation`
and side-chain methyl 2H (R_Dz, R_Dy, R_3Dz2) via
:class:`~mdrelax.methyl.MethylRelaxation`.
"""

from .nh import NHRelaxation
from .methyl import MethylRelaxation

__version__ = "0.1.0"
__all__ = ["NHRelaxation", "MethylRelaxation"]
