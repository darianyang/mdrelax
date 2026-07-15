"""Autocorrelation-function correctness tests."""

import numpy as np

from mdrelax import acf


def _random_bond_traj(n_frames=2000, seed=0):
    """A slowly reorienting unit vector via a random walk on the sphere."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(scale=0.05, size=(n_frames, 3))
    v = np.cumsum(steps, axis=0) + np.array([0.0, 0.0, 5.0])
    return v


def test_fft_matches_direct():
    v = _random_bond_traj()
    max_lag = 400
    a_fft = acf.p2_acf_fft(v, max_lag=max_lag)
    a_dir = acf.p2_acf_direct(v, max_lag=max_lag)
    assert np.allclose(a_fft, a_dir, atol=1e-10)


def test_acf_starts_at_one():
    v = _random_bond_traj(seed=3)
    a = acf.p2_acf_fft(v, max_lag=100)
    assert a[0] == np.float64(1.0) or abs(a[0] - 1.0) < 1e-12


def test_static_vector_has_flat_acf():
    v = np.tile(np.array([0.0, 0.0, 1.0]), (500, 1))
    a = acf.p2_acf_fft(v, max_lag=100, normalize=False)
    assert np.allclose(a, 1.0, atol=1e-10)


def test_p2_acf_many_shapes_and_consistency():
    v = np.stack([_random_bond_traj(seed=s) for s in range(4)], axis=1)
    a = acf.p2_acf_many(v, max_lag=200)
    assert a.shape == (200, 4)
    assert np.allclose(a[:, 0], acf.p2_acf_fft(v[:, 0, :], max_lag=200))
