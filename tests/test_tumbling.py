"""Pure-Python tumbling (pdbinertia+quadric replacement) vs ABSURDer reference."""

import warnings

import numpy as np
import pytest

from conftest import needs_ff15ipq, FF15IPQ, load_pkl

warnings.filterwarnings("ignore")


@needs_ff15ipq
def test_backbone_tauM_matches_reference(ff15ipq_bb_tcf):
    from mdrelax import tumbling
    tm_ref = np.loadtxt(FF15IPQ / "tau" / "tm.avg.input")   # resid, tauM_ns, std
    n_nh = ff15ipq_bb_tcf.shape[1] - 1
    tauM = tumbling.fit_backbone_tauM(ff15ipq_bb_tcf, n_nh,
                                      l_block_ps=1_000_000,
                                      start_tau_ps=10000, accuracy=100)
    assert np.mean(np.abs(tauM / 1000 - tm_ref[:, 1])) < 0.1  # < 0.1 ns


@needs_ff15ipq
def test_axial_diffusion_close_to_quadric(ff15ipq_bb_tcf):
    import MDAnalysis as mda
    from mdrelax import tumbling, geometry
    tm_ref = np.loadtxt(FF15IPQ / "tau" / "tm.avg.input")
    n_nh = ff15ipq_bb_tcf.shape[1] - 1
    tauM = tumbling.fit_backbone_tauM(ff15ipq_bb_tcf, n_nh,
                                      l_block_ps=1_000_000,
                                      start_tau_ps=10000, accuracy=100)
    u = mda.Universe(str(FF15IPQ / "initial.pdb"))
    pairs = geometry.find_nh_pairs(u)
    nh_res = load_pkl(FF15IPQ / "nh_residues.pkl")
    resids = [p["resid"] for p in pairs]
    order = [resids.index(r) for r in nh_res]
    nhv = np.array([pairs[i]["H"].position - pairs[i]["N"].position for i in order])
    fit = tumbling.fit_axial_diffusion(tauM, nhv)
    # quadric reference: Diso = 1.28761e7 s^-1, Dpar/Dper = 1.17569
    assert fit["Diso"] == pytest.approx(1.28761e7, rel=0.05)
    assert fit["Dratio"] == pytest.approx(1.17569, rel=0.10)


@needs_ff15ipq
def test_methyl_tauR_geometry_matches_reference(ff15ipq_tauR):
    import MDAnalysis as mda
    from mdrelax import tumbling, geometry
    labels_ref, tauR_ref = ff15ipq_tauR
    Diso, Dratio = 1.28761e7, 1.17569
    Dpar = 3 * Diso / (1 + 2 / Dratio)
    Dper = 3 * Diso / (2 + Dratio)
    ax = mda.Universe(str(FF15IPQ / "tau" / "avg.axial.pdb"))
    groups = geometry.find_methyl_groups(ax)
    lbl2i = {g["label"]: i for i, g in enumerate(groups)}
    cc = np.array([g["C"].position - g["C_axis"].position for g in groups])
    tauR = tumbling.methyl_tau_R(cc, Diso, Dpar, Dper, np.array([0, 0, 1.0]))
    common = [l for l in labels_ref if l in lbl2i]
    assert len(common) == len(labels_ref)   # every reference methyl found
    mine = np.array([tauR[lbl2i[l]] for l in common])
    ref = np.array([tauR_ref[labels_ref.index(l)] for l in common])
    assert np.mean(np.abs(mine - ref)) < 0.05    # < 0.05 ns
