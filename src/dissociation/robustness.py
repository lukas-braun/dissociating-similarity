import jax
import jax.numpy as np


def input_noise_sensitivity(key, w1, w2, xs, ys, sigma_x, p):
    _, n = xs.shape
    xis = jax.random.normal(key, xs.shape + (p,)) * sigma_x

    def loss(xi):
        return 0.5 / n * np.linalg.norm(w2 @ w1 @ (xs + xi) - ys) ** 2

    loss = jax.vmap(loss, in_axes=(-1,))

    return np.mean(loss(xis))


def input_noise_sensitivity_theory(w1, w2, xs, ys, sigma_x):
    p = xs.shape[-1]

    sig_xx = 1.0 / p * xs @ xs.T
    sig_yx = 1.0 / p * ys @ xs.T
    sig_yy = 1.0 / p * ys @ ys.T

    c = np.trace(sig_yy) - np.trace(sig_yx @ np.linalg.pinv(sig_xx) @ sig_yx.T)
    return 0.5 * (sigma_x**2 * np.linalg.norm(w2 @ w1, ord="fro")**2 + c)


def parameter_noise_sensitivity(key, w1, w2, xs, ys, sigma_1, sigma_2, p):
    n = xs.shape[-1]

    key1, key2 = jax.random.split(key)
    xis_1 = jax.random.normal(key1, w1.shape + (p,)) * sigma_1
    xis_2 = jax.random.normal(key2, w2.shape + (p,)) * sigma_2

    def loss(xi_1, xi_2):
        return 0.5 / n * np.linalg.norm((w2 + xi_2) @ (w1 + xi_1) @ xs - ys)**2

    loss = jax.vmap(loss, in_axes=(-1, -1))

    return np.mean(loss(xis_1, xis_2))


def parameter_noise_sensitivity_theory(w1, w2, xs, ys, sigma_1, sigma_2):
    p = xs.shape[-1]
    out_dim, hidden_dim = w2.shape

    sig_xx = 1.0 / p * xs @ xs.T
    sig_yx = 1.0 / p * ys @ xs.T
    sig_yy = 1.0 / p * ys @ ys.T

    tr_sigxx = np.linalg.trace(sig_xx)

    t1 = sigma_1**2 * np.linalg.norm(w2, ord="fro")**2 * tr_sigxx
    t2 = out_dim * sigma_2**2 * np.linalg.trace(w1.T @ w1 @ sig_xx)
    t3 = sigma_1**2 * hidden_dim * sigma_2**2 * out_dim * tr_sigxx
    c = np.trace(sig_yy) - np.trace(sig_yx @ np.linalg.pinv(sig_xx) @ sig_yx.T)

    return 0.5 * (t1 + t2 + t3 + c)
