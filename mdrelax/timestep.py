"""Working out the real time between trajectory frames.

Every correlation time this package fits is read off a time axis built from the
spacing between frames, so getting that spacing wrong rescales every result by
the same factor -- silently, because nothing about the output looks wrong.

The spacing cannot be trusted to the trajectory file. DCDs in particular store
their timestep in AKMA units, and several writers (OpenMM via mdtraj's
``DCDReporter`` among them) never set it, leaving the header at 1 AKMA unit =
0.04888821 ps regardless of how the run was actually configured. MDAnalysis
reports that value back faithfully; a caller who trusts it gets correlation
times short by a factor of 20.455.

Hence :func:`resolve_dt_ps`: callers who know their save interval say so, and
the file header is only a fallback -- one that warns when it returns the value
that means "nobody set this".
"""

import warnings

# one AKMA time unit in ps: what an unset DCD header reports
AKMA_PS = 0.04888821
_AKMA_RTOL = 1e-4


def resolve_dt_ps(universe, dt_ps, traj_step=1):
    """Time between the frames of ``universe``, in ps.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Already loaded, including any ``in_memory_step`` striding.
    dt_ps : float or None
        Save interval of the trajectory *on disk*, before striding. None falls
        back to the file header.
    traj_step : int
        Stride the universe was loaded with. ``dt_ps`` is scaled by it, since
        it describes the file rather than the strided universe.

    Returns
    -------
    float : the spacing between successive frames of ``universe``, in ps.
    """
    if dt_ps is not None:
        return float(dt_ps) * traj_step

    dt = float(universe.trajectory.dt)
    if abs(dt - AKMA_PS * traj_step) <= _AKMA_RTOL * AKMA_PS * traj_step:
        warnings.warn(
            f"trajectory reports dt={dt:.8f} ps, which is {traj_step} x one AKMA "
            f"time unit -- the value a DCD carries when its writer never set a "
            f"timestep (OpenMM via mdtraj's DCDReporter does this). Every "
            f"correlation time fitted from it will be wrong by the same factor. "
            f"Pass dt_ps=<save interval in ps> to use the real value.",
            RuntimeWarning, stacklevel=3)
    return dt
