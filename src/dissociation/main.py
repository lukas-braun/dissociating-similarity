# ruff: noqa: E741

import jax
import jax.numpy as np

"""
import threading


class RNGKey:
    _key = None
    _lock = threading.Lock()

    def __new__(cls, keys_n=1):
        if cls._key is None:
            with cls._lock:
                if cls._key is None:
                    cls._key = jax.random.key(0)
        cls._key, key = jax.random.split(cls._key, 2)
        return key if keys_n == 1 else jax.random.split(key, keys_n)


def sample_weights(in_dim, hidden_dim, out_dim, std1=1.0, std2=1.0):
    w1 = jax.random.normal(RNGKey(), (hidden_dim, in_dim)) * std1
    w2 = jax.random.normal(RNGKey(), (out_dim, hidden_dim)) * std2
    return w1, w2


def random_regression_task(in_dim, out_dim, n):
    xs = jax.random.normal(RNGKey(), (in_dim, n)) * 1 / np.sqrt(in_dim)
    ys = jax.random.normal(RNGKey(), (out_dim, n)) * 1 / np.sqrt(out_dim)
    return xs, ys
"""


def noise_sensitivity(w1, w2, xs, ys, sigma_x, sigma_1, sigma_2, samples_n=100000):
    _, n = xs.shape
    xis_x = jax.random.normal(RNGKey(), xs.shape + (samples_n,)) * sigma_x
    xis_1 = jax.random.normal(RNGKey(), w1.shape + (samples_n,)) * sigma_1
    xis_2 = jax.random.normal(RNGKey(), w2.shape + (samples_n,)) * sigma_2

    def loss(xi_x, xi_1, xi_2):
        return (
            0.5 / n * np.linalg.norm((w2 + xi_2) @ (w1 + xi_1) @ (xs + xi_x) - ys) ** 2
        )

    loss = jax.vmap(loss, in_axes=(-1, -1, -1))

    return np.mean(loss(xis_x, xis_1, xis_2))


def input_noise_sensitivity(w1, w2, xs, ys, sigma_x, samples_n=100000):
    _, n = xs.shape
    xis = jax.random.normal(RNGKey(), xs.shape + (samples_n,)) * sigma_x

    def loss(xi):
        return 0.5 / n * np.linalg.norm(w2 @ w1 @ (xs + xi) - ys) ** 2

    loss = jax.vmap(loss, in_axes=(-1,))

    return np.mean(loss(xis))


def input_noise_sensitivity_theory(w1, w2, xs, ys, sigma):
    p = xs.shape[-1]

    sig_xx = 1.0 / p * xs @ xs.T
    sig_yx = 1.0 / p * ys @ xs.T
    sig_yy = 1.0 / p * ys @ ys.T

    c = np.trace(sig_yy) - np.trace(sig_yx @ np.linalg.pinv(sig_xx) @ sig_yx.T)
    return 0.5 * (sigma**2 * np.linalg.norm(w2 @ w1, ord="fro") ** 2 + c)


def parameter_noise_sensitivity(w1, w2, xs, ys, sigma_1, sigma_2, samples_n=100000):
    n = xs.shape[-1]
    xis_1 = jax.random.normal(RNGKey(), w1.shape + (samples_n,)) * sigma_1
    xis_2 = jax.random.normal(RNGKey(), w2.shape + (samples_n,)) * sigma_2

    def loss(xi_1, xi_2):
        return 0.5 / n * np.linalg.norm((w2 + xi_2) @ (w1 + xi_1) @ xs - ys) ** 2

    loss = jax.vmap(loss, in_axes=(-1, -1))

    return np.mean(loss(xis_1, xis_2))


def parameter_noise_sensitivity_theory(w1, w2, xs, ys, sigma_1, sigma_2):
    p = xs.shape[-1]
    out_dim, hidden_dim = w2.shape

    sig_xx = 1.0 / p * xs @ xs.T
    sig_yx = 1.0 / p * ys @ xs.T
    sig_yy = 1.0 / p * ys @ ys.T

    t1 = sigma_1**2 * np.linalg.norm(w2, ord="fro") ** 2 * np.linalg.trace(sig_xx)
    t2 = out_dim * sigma_2**2 * np.linalg.trace(w1.T @ w1 @ sig_xx)
    t3 = sigma_1**2 * hidden_dim * sigma_2**2 * out_dim * np.linalg.trace(sig_xx)
    c = np.trace(sig_yy) - np.trace(sig_yx @ np.linalg.pinv(sig_xx) @ sig_yx.T)

    return 0.5 * (t1 + t2 + t3 + c)


def compact_svd(a, threshold=1e-6):
    u, s, vt = np.linalg.svd(a, False)
    mask = s > threshold
    return u[:, mask], np.diag(s[mask]), vt.T[:, mask]

