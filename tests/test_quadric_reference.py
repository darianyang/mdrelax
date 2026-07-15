"""mdrelax.tumbling vs the reference pdbinertia / quadric_diffusion programs.

Both reference programs ship an ubiquitin test case, and both are exercised here
from the *same* single PDB (`1ubq.prot.pdb`):

* pdbinertia  -- centre of mass, principal moments and axes of the inertia
  tensor, checked against its `ubq.inertia.out`.
* quadric_diffusion -- the iso / axial / aniso diffusion tensors fit to the
  per-residue tau_M in `ubq.tm.input`, checked against its `ubq.output`.  The
  N-H bond vectors are taken from the PDB, reproducing the `x y z` columns that
  quadric prints in its input table.

Those files are vendored under `tests/data/ubq/` (see the README there for their
provenance), so these tests always run.
"""

import numpy as np
import pytest

from conftest import UBQ_DIR

from mdrelax import tumbling

UBQ_PDB = UBQ_DIR / "1ubq.prot.pdb"

# pdbinertia only recognises these elements and guesses them from the atom name.
_MASSES = {"H": 1.00794, "C": 12.011, "N": 14.00674, "O": 15.9994,
           "P": 30.973762, "S": 32.066}


def _element(name):
    n = name.lstrip("0123456789")
    return n[0] if n else None


def _read_pdb(path):
    """(name, resid, xyz) for every ATOM record, as pdbinertia reads them."""
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                out.append((line[12:16].strip(), int(line[22:26]),
                            [float(line[30:38]), float(line[38:46]),
                             float(line[46:54])]))
    return out


@pytest.fixture(scope="module")
def ubq():
    """Coordinates, masses and a {resid: {name: xyz}} lookup for 1ubq.prot.pdb."""
    atoms = _read_pdb(UBQ_PDB)
    coords = np.array([a[2] for a in atoms])
    masses = np.array([_MASSES[_element(a[0])] for a in atoms])
    byres = {}
    for name, resid, xyz in atoms:
        byres.setdefault(resid, {})[name] = np.array(xyz)
    return coords, masses, byres


@pytest.fixture(scope="module")
def ubq_local_D(ubq):
    """(nh_vectors, tauM_ps, sigma_tauM_ps) for quadric's 55 ubiquitin residues."""
    _, _, byres = ubq
    tm = np.loadtxt(UBQ_DIR / "ubq.tm.input")      # resid, tau_M (ns), dtau_M
    resids = tm[:, 0].astype(int)
    nh = np.array([byres[r]["H"] - byres[r]["N"] for r in resids])
    return nh, tm[:, 1] * 1000.0, tm[:, 2] * 1000.0


# ── pdbinertia ─────────────────────────────────────────────────────────────

def test_inertia_matches_pdbinertia(ubq):
    """Centre of mass, principal moments and axes vs pdbinertia's ubq.inertia.out."""
    coords, masses, _ = ubq
    assert len(coords) == 1230                          # "# atoms read  1230"

    axes, moments, com = tumbling.inertia_principal_axes(coords, masses)

    # our atomic-mass table differs from pdbinertia's in the 5th digit, so the
    # mass-weighted quantities are compared at that level rather than exactly
    assert masses.sum() == pytest.approx(8563.9004, rel=1e-5)
    assert com == pytest.approx([30.3128, 28.8001, 15.3504], abs=5e-4)
    assert moments == pytest.approx([934104.0, 844540.75, 594138.9375], rel=1e-4)
    assert moments[1] / moments[0] == pytest.approx(0.9041, abs=1e-4)
    assert moments[2] / moments[0] == pytest.approx(0.6361, abs=1e-4)

    # pdbinertia prints the principal axes as the *rows* of its rotation matrix;
    # each axis is only defined up to a sign, hence the abs().
    rotation = np.array([[0.7981, -0.4332, -0.4187],
                         [-0.0741, -0.7603, 0.6453],
                         [-0.5979, -0.4840, -0.6389]])
    assert np.abs(axes.T) == pytest.approx(np.abs(rotation), abs=1e-3)


# ── quadric: bond vectors ──────────────────────────────────────────────────

def test_nh_vectors_match_quadric_input(ubq, ubq_local_D):
    """The N-H vectors we build from the PDB are the ones quadric analysed."""
    nh, _, _ = ubq_local_D
    # "Input Data" rows are: resid, atom, D/10^7, dD/10^7, x, y, z
    ref = np.loadtxt(UBQ_DIR / "ubq.output", skiprows=6, max_rows=55,
                     usecols=(4, 5, 6), dtype=float, comments=None)
    ours = nh / np.linalg.norm(nh, axis=1, keepdims=True)
    # quadric stores H->N where we store N->H; P2 is even, so only the axis matters
    assert np.abs(np.sum(ours * ref, axis=1)) == pytest.approx(1.0, abs=1e-3)


def test_local_D_reduces_to_axial_formula():
    """The quadric equation and the axial P2 expression agree (they must)."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(20, 3))
    Diso, Dratio = 4.0e7, 1.3
    Dpar, Dper = tumbling._axial_D(Diso, Dratio)
    axis = np.array([0.0, 0.0, 1.0])
    quadric = 1.0 / (6.0 * tumbling.local_D((Dper, Dper, Dpar), np.eye(3), v)) * 1e9
    axial = tumbling.methyl_tau_R(v, Diso, Dpar, Dper, axis)
    assert quadric == pytest.approx(axial, rel=1e-12)


# ── quadric: the three diffusion models ────────────────────────────────────

def test_isotropic_matches_quadric(ubq_local_D):
    """Isotropic Diso and chi-square vs quadric's 'Isotropic Results'."""
    nh, tauM, dtauM = ubq_local_D
    fit = tumbling.fit_diffusion(tauM, nh, model="iso", sigma_tauM_ps=dtauM)

    assert fit["Diso"] * 1e-7 == pytest.approx(4.05073, abs=1e-5)
    assert fit["chi2"] == pytest.approx(697.8904, rel=1e-5)
    assert fit["chi2_red"] == pytest.approx(12.92390, rel=1e-5)
    assert fit["Dxx"] == fit["Dyy"] == fit["Dzz"]


def test_axial_matches_quadric(ubq_local_D):
    """Axial tensor, orientation and chi-square vs quadric's 'Axial Results'."""
    nh, tauM, dtauM = ubq_local_D
    fit = tumbling.fit_diffusion(tauM, nh, model="axial", sigma_tauM_ps=dtauM)

    assert fit["Diso"] * 1e-7 == pytest.approx(4.01315, abs=1e-4)
    assert fit["Dratio"] == pytest.approx(1.15315, abs=1e-4)
    assert fit["theta"] == pytest.approx(0.72561, abs=2e-3)
    assert fit["phi"] == pytest.approx(0.84029, abs=2e-3)
    assert fit["chi2"] == pytest.approx(407.6873, rel=1e-4)
    assert fit["chi2_red"] == pytest.approx(7.993868, rel=1e-4)
    # prolate: Dpar/Dper > 1 puts the unique axis on z, and Dxx == Dyy
    assert fit["Dxx"] == pytest.approx(fit["Dyy"], rel=1e-9)
    assert fit["Dzz"] > fit["Dyy"]


def test_anisotropic_matches_quadric(ubq_local_D):
    """Fully anisotropic tensor vs quadric's 'Anisotropic Results'."""
    nh, tauM, dtauM = ubq_local_D
    fit = tumbling.fit_diffusion(tauM, nh, model="aniso", sigma_tauM_ps=dtauM)

    assert fit["Diso"] * 1e-7 == pytest.approx(4.01763, abs=1e-4)
    assert fit["zz_ratio"] == pytest.approx(1.15870, abs=1e-3)
    assert fit["xy_ratio"] == pytest.approx(0.97565, abs=1e-3)
    assert fit["chi2"] == pytest.approx(399.9402, rel=1e-4)
    assert fit["chi2_red"] == pytest.approx(8.162045, rel=1e-4)

    # the principal values themselves, back-solved from quadric's ratios
    assert (fit["Dxx"] * 1e-7, fit["Dyy"] * 1e-7, fit["Dzz"] * 1e-7) == \
        pytest.approx((3.76875, 3.86281, 4.42134), abs=1e-3)


def test_f_statistics_match_quadric(ubq_local_D):
    """quadric's model-comparison F values."""
    nh, tauM, dtauM = ubq_local_D
    fits = {m: tumbling.fit_diffusion(tauM, nh, model=m, sigma_tauM_ps=dtauM)
            for m in ("iso", "axial", "aniso")}

    assert tumbling.f_statistic(fits["iso"], fits["axial"]) == \
        pytest.approx(12.10107, rel=1e-3)
    assert tumbling.f_statistic(fits["axial"], fits["aniso"]) == \
        pytest.approx(0.4745771, rel=1e-2)


def test_model_selection_picks_axial_for_ubiquitin(ubq_local_D):
    """The F-test ladder must land on the model quadric's own F values imply.

    F(iso->axial) = 12.10 is significant (p ~ 1e-6), F(axial->aniso) = 0.47 is
    not -- so ubiquitin is axially symmetric, its accepted description.
    """
    nh, tauM, dtauM = ubq_local_D
    fit, trials = tumbling.select_diffusion_model(tauM, nh, sigma_tauM_ps=dtauM)

    assert fit["model"] == "axial"
    assert set(trials) == set(tumbling.MODELS)
    assert fit["Dratio"] == pytest.approx(1.15315, abs=1e-4)

    (s0, c0, F0, p0, ok0), (s1, c1, F1, p1, ok1) = fit["selection"]
    assert (s0, c0, ok0) == ("iso", "axial", True)
    assert F0 == pytest.approx(12.10107, rel=1e-3) and p0 < 1e-4
    assert (s1, c1, ok1) == ("axial", "aniso", False)
    assert F1 == pytest.approx(0.4745771, rel=1e-2) and p1 > 0.5


@pytest.mark.parametrize("model", ["iso", "axial", "aniso"])
def test_model_selection_recovers_synthetic_tensors(model):
    """Given tau_M generated by a known tensor, the ladder recovers its model."""
    rng = np.random.default_rng(7)
    v = rng.normal(size=(120, 3))
    Diso = 2.0e7
    Dpar, Dper = tumbling._axial_D(Diso, 1.6)
    D = {"iso": (Diso, Diso, Diso),
         "axial": (Dper, Dper, Dpar),
         "aniso": tumbling._aniso_D(Diso, 1.5, 0.7)}[model]
    axes = tumbling._zyz(0.4, 0.9, -0.7)
    tauM = 1.0 / (6.0 * tumbling.local_D(D, axes, v)) * 1e12
    # a little noise, else iso fits perfectly and F is degenerate
    tauM = tauM * (1.0 + rng.normal(scale=0.002, size=len(v)))

    fit, _ = tumbling.select_diffusion_model(tauM, v)
    assert fit["model"] == model


def test_explicit_model_overrides_selection(ubq_local_D):
    """Naming a model must bypass the ladder, even a badly-fitting one."""
    nh, tauM, dtauM = ubq_local_D
    fit = tumbling.fit_diffusion(tauM, nh, model="iso", sigma_tauM_ps=dtauM)
    assert fit["model"] == "iso"
    assert "selection" not in fit


def test_axial_fit_is_orientation_independent(ubq_local_D):
    """quadric needs pdbinertia to pre-align the structure; we fit the axis, so
    the same tensor must come back from an arbitrarily rotated input frame."""
    nh, tauM, dtauM = ubq_local_D
    ref = tumbling.fit_diffusion(tauM, nh, model="axial", sigma_tauM_ps=dtauM)

    rot = tumbling._zyz(0.7, 1.1, -0.4)
    fit = tumbling.fit_diffusion(tauM, nh @ rot.T, model="axial",
                                 sigma_tauM_ps=dtauM)

    assert fit["Diso"] == pytest.approx(ref["Diso"], rel=1e-4)
    assert fit["Dratio"] == pytest.approx(ref["Dratio"], rel=1e-3)
    assert fit["chi2"] == pytest.approx(ref["chi2"], rel=1e-4)
    # the unique axis must co-rotate with the input frame
    assert abs(float((rot @ ref["axis"]) @ fit["axis"])) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("Dratio", [1.5, 0.6])
def test_axial_round_trip_prolate_and_oblate(Dratio):
    """A synthetic axial tensor is recovered from the tau_M it generates.

    Covers the oblate branch (Dratio < 1), where sorting puts the unique axis on
    x rather than z -- 'axis' must track that, and tau_R_from_fit must agree with
    the P2 formula either way.
    """
    rng = np.random.default_rng(3)
    v = rng.normal(size=(80, 3))
    Diso = 2.0e7
    Dpar, Dper = tumbling._axial_D(Diso, Dratio)
    true_axis = np.array([0.3, -0.5, 0.8])
    true_axis /= np.linalg.norm(true_axis)

    tau_true = tumbling.methyl_tau_R(v, Diso, Dpar, Dper, true_axis)
    fit = tumbling.fit_diffusion(tau_true * 1000.0, v, model="axial")

    assert fit["Diso"] == pytest.approx(Diso, rel=1e-4)
    assert fit["Dratio"] == pytest.approx(Dratio, rel=1e-3)
    assert abs(float(fit["axis"] @ true_axis)) == pytest.approx(1.0, abs=1e-4)
    assert tumbling.tau_R_from_fit(v, fit) == pytest.approx(tau_true, rel=1e-4)
    # 'axis' is the symmetry axis, so methyl_tau_R must reproduce the fit too
    assert tumbling.methyl_tau_R(v, fit["Diso"], fit["Dpar"], fit["Dper"],
                                 fit["axis"]) == pytest.approx(tau_true, rel=1e-4)


def test_tau_R_from_fit_matches_axial_helper(ubq_local_D):
    """tau_R_from_fit is model-agnostic; on an axial fit it must reproduce
    methyl_tau_R, which methyl.py's axial path uses."""
    nh, tauM, dtauM = ubq_local_D
    fit = tumbling.fit_diffusion(tauM, nh, model="axial", sigma_tauM_ps=dtauM)
    rng = np.random.default_rng(1)
    cc = rng.normal(size=(30, 3))

    general = tumbling.tau_R_from_fit(cc, fit)
    axial = tumbling.methyl_tau_R(cc, fit["Diso"], fit["Dpar"], fit["Dper"],
                                  fit["axis"])
    assert general == pytest.approx(axial, rel=1e-9)


def test_iso_fit_gives_uniform_tau_R(ubq_local_D):
    """An isotropic tensor must give every vector the same tau_R = tau_c."""
    nh, tauM, dtauM = ubq_local_D
    fit = tumbling.fit_diffusion(tauM, nh, model="iso", sigma_tauM_ps=dtauM)
    rng = np.random.default_rng(2)
    tau_R = tumbling.tau_R_from_fit(rng.normal(size=(10, 3)), fit)
    assert tau_R == pytest.approx(fit["tau_c_ns"], rel=1e-9)
