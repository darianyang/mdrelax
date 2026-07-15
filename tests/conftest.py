"""Shared fixtures and data paths for the mdrelax test suite."""

import bz2
import pickle
import _pickle as cPickle
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

# ── data locations ─────────────────────────────────────────────────────────
DATA_MD_T4L = ROOT / "data-md-t4l"
DATA_NH = ROOT / "data-nh"
FF15IPQ = ROOT / "data-ch3-ff15ipq" / "tcf-1us"

T4L_PDB = DATA_MD_T4L / "sim1_dry.pdb"
T4L_XTC = DATA_MD_T4L / "segment_001.xtc"

# The ubiquitin test case shipped with pdbinertia / quadric_diffusion, vendored
# into the repo so the reference tests always run.  See tests/data/ubq/README.md.
UBQ_DIR = ROOT / "tests" / "data" / "ubq"


def _have(*paths):
    return all(Path(p).exists() for p in paths)


needs_t4l = pytest.mark.skipif(
    not _have(T4L_PDB, T4L_XTC), reason="data-md-t4l trajectory not present")
needs_ff15ipq = pytest.mark.skipif(
    not _have(FF15IPQ / "results" / "rates.pkl"),
    reason="ff15ipq ABSURDer reference data not present")


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_bz2(path):
    with bz2.BZ2File(path, "rb") as f:
        return cPickle.load(f)


@pytest.fixture(scope="session")
def ff15ipq_methyl_tcf():
    """ABSURDer methyl TCF array (n_blocks, n_time, 3*n_methyl + 1)."""
    return np.array(load_bz2(FF15IPQ / "tcf_methyl" / "sim1_rot_trans.pbz2"))


@pytest.fixture(scope="session")
def ff15ipq_bb_tcf():
    """ABSURDer backbone NH TCF, first block (n_time, n_nh + 1)."""
    return np.array(load_bz2(FF15IPQ / "tcf_bb" / "sim1_prot_nopbc.pbz2"))[0]


@pytest.fixture(scope="session")
def ff15ipq_tauR():
    """(labels, tauR_ns) in ABSURDer construction order."""
    labels, tau = [], []
    with open(FF15IPQ / "tau" / "tauR_methyl_specific") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split()
            labels.append(p[0])
            tau.append(float(p[1]))
    return labels, np.array(tau)


@pytest.fixture(scope="session")
def ff15ipq_rates():
    """Reference rates (4, n_methyl, n_blocks) and sorted methyl labels."""
    rates = np.array(load_pkl(FF15IPQ / "results" / "rates.pkl"))
    methyls = load_pkl(FF15IPQ / "results" / "methyls.pkl")
    return rates, methyls
