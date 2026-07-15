"""Second-Legendre-polynomial (P2) autocorrelation functions of bond vectors.

C(t) = < P2( u(0) . u(t) ) >,   P2(x) = (3 x^2 - 1) / 2

The FFT implementation (:func:`p2_acf_fft`) is the workhorse; the direct
double loop (:func:`p2_acf_direct`) is kept for validation/testing.
"""

import numpy as np


def _unit(vectors):
    """Normalize an (..., 3) array of vectors to unit length."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms < 1e-12, 1.0, norms)


def p2_acf_fft(vectors, max_lag=None, normalize=True):
    """P2 autocorrelation of a single time series of vectors via FFT.

    Uses the Cartesian expansion of ``(u(0).u(t))^2`` into the six quadratic
    products (xx, yy, zz, xy, xz, yz), each autocorrelated with the
    Wiener-Khinchin theorem, giving an O(N log N) exact result.

    Parameters
    ----------
    vectors : ndarray, shape (n_frames, 3)
    max_lag : int, optional
        Number of lag points to return (default: n_frames).
    normalize : bool
        If True, divide by C(0) so that C(0) = 1.

    Returns
    -------
    acf : ndarray, shape (max_lag,)
    """
    u = _unit(np.asarray(vectors, dtype=np.float64))
    n = u.shape[0]
    if max_lag is None:
        max_lag = n
    max_lag = min(max_lag, n)
    n_fft = 2 * n  # zero-pad to avoid circular correlation

    # quadratic components: xx, yy, zz, xy, xz, yz
    q = np.empty((n, 6), dtype=np.float64)
    q[:, 0] = u[:, 0] * u[:, 0]
    q[:, 1] = u[:, 1] * u[:, 1]
    q[:, 2] = u[:, 2] * u[:, 2]
    q[:, 3] = u[:, 0] * u[:, 1]
    q[:, 4] = u[:, 0] * u[:, 2]
    q[:, 5] = u[:, 1] * u[:, 2]

    f = np.fft.rfft(q, n=n_fft, axis=0)
    acf_q = np.fft.irfft(f * np.conjugate(f), n=n_fft, axis=0)[:n]

    counts = np.arange(n, 0, -1, dtype=np.float64)[:, None]
    acf_q /= counts

    # <(u0.ut)^2> = <xx> + <yy> + <zz> + 2(<xy> + <xz> + <yz>)
    dot2 = acf_q[:, 0] + acf_q[:, 1] + acf_q[:, 2] + 2.0 * (
        acf_q[:, 3] + acf_q[:, 4] + acf_q[:, 5])
    acf = 1.5 * dot2 - 0.5
    acf = acf[:max_lag]
    if normalize:
        acf = acf / acf[0]
    return acf


def p2_acf_direct(vectors, max_lag=None, normalize=True):
    """Reference O(N^2) P2 autocorrelation (for testing/validation)."""
    u = _unit(np.asarray(vectors, dtype=np.float64))
    n = u.shape[0]
    if max_lag is None:
        max_lag = n
    max_lag = min(max_lag, n)
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        dots = np.einsum('ij,ij->i', u[: n - lag if lag else None], u[lag:])
        acf[lag] = np.mean(0.5 * (3.0 * dots**2 - 1.0))
    if normalize:
        acf = acf / acf[0]
    return acf


def p2_acf_many(vectors, max_lag=None, normalize=True):
    """P2 ACF for many bond vectors at once.

    Parameters
    ----------
    vectors : ndarray, shape (n_frames, n_bonds, 3)
    max_lag : int, optional
    normalize : bool

    Returns
    -------
    acfs : ndarray, shape (max_lag, n_bonds)
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    n_bonds = vectors.shape[1]
    if max_lag is None:
        max_lag = vectors.shape[0]
    max_lag = min(max_lag, vectors.shape[0])
    out = np.empty((max_lag, n_bonds))
    for b in range(n_bonds):
        out[:, b] = p2_acf_fft(vectors[:, b, :], max_lag, normalize)
    return out
