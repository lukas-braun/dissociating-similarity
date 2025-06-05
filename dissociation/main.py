# ruff: noqa: E741

import jax
import jax.numpy as np

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


"""
def compact_svd(a, threshold=1e-6):
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    mask = s > threshold
    s_masked = s * mask
    u_masked = u * mask[np.newaxis, :]
    vt_masked = vt * mask[:, np.newaxis]
    return u_masked, np.diag(s_masked), vt_masked.T
"""

"""
def walk_gls(xs, ys, hidden_dim, steps, alpha=0.0125 / 2.):
    in_dim, n = xs.shape
    out_dim = ys.shape[0]
    std1 = 1. / np.sqrt(in_dim)
    std2 = 1. / np.sqrt(hidden_dim)
    w1, w2 = sample_weights(in_dim, hidden_dim, out_dim, std1, std2)

    w1s = np.zeros((steps, hidden_dim, in_dim))
    w2s = np.zeros((steps, out_dim, hidden_dim))

    for i in range(steps*2 - 1):
        w1 = (1. - alpha) * w1 + alpha * jax.random.normal(RNGKey(), w1.shape) * 5.
        w2 = (1. - alpha) * w2 + alpha * jax.random.normal(RNGKey(), w2.shape) * 5.
        loss = np.linalg.norm(w2 @ w1 @ xs - ys, ord="fro")**2
        while loss > 1e-6:
            dw1 = 1. / n * w2.T @ (w2 @ w1 @ xs - ys) @ xs.T
            dw2 = 1. / n * (w2 @ w1 @ xs - ys) @ xs.T @ w1.T
            w1 -= 0.5 * dw1
            w2 -= 0.5 * dw2
            loss = np.linalg.norm(w2 @ w1 @ xs @ xs.T - ys @ xs.T, ord="fro")**2
            assert np.max(w2) < 25

        if i >= steps - 1:
            w1s = w1s.at[i-steps+1].set(w1)
            w2s = w2s.at[i-steps+1].set(w2)

    return w1s, w2s
"""

_warmup = 200


def walk_gls(xs, ys, hidden_dim, steps, alpha=0.00625):
    in_dim, n = xs.shape
    out_dim = ys.shape[0]
    std1 = 1.0 / np.sqrt(in_dim)
    std2 = 1.0 / np.sqrt(hidden_dim)
    w1, w2 = sample_weights(in_dim, hidden_dim, out_dim, std1, std2)

    key = RNGKey()

    def step(carry, _):
        w1, w2, key = carry
        key, k1, k2 = jax.random.split(key, 3)

        w1 = (1.0 - alpha) * w1 + alpha * jax.random.normal(k1, w1.shape) * 5.0
        w2 = (1.0 - alpha) * w2 + alpha * jax.random.normal(k2, w2.shape) * 5.0

        def condition(state):
            w1, w2 = state
            return np.linalg.norm(w2 @ w1 @ xs - ys, ord="fro") ** 2 > 1e-6

        def step(state):
            w1, w2 = state
            l = 1.0 / n * (w2 @ w1 @ xs - ys) @ xs.T
            dw1 = w2.T @ l
            dw2 = l @ w1.T
            return (w1 - 0.25 * dw1, w2 - 0.25 * dw2)

        w1, w2 = jax.lax.while_loop(condition, step, (w1, w2))
        return (w1, w2, key), (w1, w2)

    step = jax.jit(step)

    (w1, w2, key), _ = jax.lax.scan(step, (w1, w2, key), None, length=_warmup)
    _, (w1s, w2s) = jax.lax.scan(step, (w1, w2, key), None, length=steps)

    return w1s, w2s


"""
def walk_lss(xs, ys, hidden_dim, steps, alpha=0.0125 / 2.):
    in_dim, n = xs.shape
    out_dim = ys.shape[0]
    std1 = 1. / np.sqrt(in_dim)
    std2 = 1. / np.sqrt(hidden_dim)
    w1, w2 = sample_weights(in_dim, hidden_dim, out_dim, std1, std2)

    g, _, _ = compact_svd(xs @ xs.T)

    w1s = np.zeros((steps, hidden_dim, in_dim))
    w2s = np.zeros((steps, out_dim, hidden_dim))

    for i in range(steps*2 - 1):
        w1 = (1. - alpha) * w1 + alpha * jax.random.normal(RNGKey(), w1.shape) * 5.
        w2 = (1. - alpha) * w2 + alpha * jax.random.normal(RNGKey(), w2.shape) * 5.
        w1 = w1 @ (g @ g.T)

        loss = np.linalg.norm(w2 @ w1 @ xs @ xs.T - ys @ xs.T, ord="fro")**2
        while loss > 1e-6:
            dw1 = 1. / n * w2.T @ (w2 @ w1 @ xs - ys) @ xs.T
            dw2 = 1. / n * (w2 @ w1 @ xs - ys) @ xs.T @ w1.T
            w1 -= 0.5 * dw1
            w2 -= 0.5 * dw2
            loss = np.linalg.norm(w2 @ w1 @ xs @ xs.T - ys @ xs.T, ord="fro")**2
            assert np.max(w2) < 25

        if i >= steps - 1:
            w1s = w1s.at[i-steps+1].set(w1)
            w2s = w2s.at[i-steps+1].set(w2)

    return w1s, w2s
"""


def walk_lss(xs, ys, hidden_dim, steps, alpha=0.00625):
    in_dim, n = xs.shape
    out_dim = ys.shape[0]
    std1 = 1.0 / np.sqrt(in_dim)
    std2 = 1.0 / np.sqrt(hidden_dim)
    w1, w2 = sample_weights(in_dim, hidden_dim, out_dim, std1, std2)

    key = RNGKey()

    u, s, v = compact_svd(ys @ xs.T @ np.linalg.pinv(xs @ xs.T))
    a, _, _ = compact_svd(xs)
    pr = v @ v.T
    pi = a @ a.T - v @ v.T
    pu = np.identity(in_dim) - a @ a.T

    def step(carry, _):
        w1, w2, key = carry
        key, k1, k2 = jax.random.split(key, 3)

        w1 = (1.0 - alpha) * w1 + alpha * jax.random.normal(k1, w1.shape) * 5.0
        w2 = (1.0 - alpha) * w2 + alpha * jax.random.normal(k2, w2.shape) * 5.0

        def condition(state):
            w1, w2 = state
            return np.linalg.norm(w2 @ w1 @ xs - ys, ord="fro") ** 2 > 1e-6

        def step(state):
            w1, w2 = state
            l = 1.0 / n * (w2 @ w1 @ xs - ys) @ xs.T
            dw1 = w2.T @ l
            dw2 = l @ w1.T
            return (w1 - 0.25 * dw1, w2 - 0.25 * dw2)

        w1, w2 = jax.lax.while_loop(condition, step, (w1, w2))

        # First rank constraint
        q = w1 @ v @ np.diag(1.0 / np.sqrt(np.diag(s)))
        q_ = np.linalg.pinv(q)
        qq = q @ q_
        iqq = np.identity(hidden_dim) - qq
        a = iqq @ w1 @ pi
        a_ = qq @ w1 @ pi
        _, s1, _ = np.linalg.svd(a, full_matrices=False)
        d, e, ft = np.linalg.svd(a_, full_matrices=False)
        g1 = a + d @ np.diag(e * (s1 > 1e-6)) @ ft

        psi = -u @ np.sqrt(s) @ q_ @ g1 @ pi @ np.linalg.pinv(iqq @ g1 @ pi)

        # Second rank constraint
        h = w1 @ xs
        h_ = np.linalg.pinv(h)
        hh = h @ h_
        ihh = np.identity(hidden_dim) - hh
        a = ihh @ w1 @ pu
        a_ = hh @ w1 @ pu
        _, s1, _ = np.linalg.svd(a, full_matrices=False)
        d, e, ft = np.linalg.svd(a_, full_matrices=False)
        g2 = a + d @ np.diag(e * (s1 > 1e-6)) @ ft

        w1 = w1 @ pr + g1 @ pi + g2 @ pu

        phi = -(u @ np.sqrt(s) @ q_ + psi) @ w1 @ pu @ np.linalg.pinv(ihh @ w1 @ pu)
        g3 = w2 @ (np.identity(hidden_dim) - w1 @ np.linalg.pinv(w1))
        w2 = u @ np.sqrt(s) @ q_ + psi + phi + g3

        return (w1, w2, key), (w1, w2)

    step = jax.jit(step)

    (w1, w2, key), _ = jax.lax.scan(step, (w1, w2, key), None, length=_warmup)
    _, (w1s, w2s) = jax.lax.scan(step, (w1, w2, key), None, length=steps)

    return w1s, w2s


"""
def walk_mrns(xs, ys, hidden_dim, steps, alpha=0.0125):
    in_dim = xs.shape[0]
    out_dim = ys.shape[0]
    m, n, o = compact_svd(ys @ xs.T @ np.linalg.pinv(xs).T)
    r_ = jax.random.normal(RNGKey(), (hidden_dim, n.shape[0]))
    gamma = jax.random.normal(RNGKey(), (hidden_dim, in_dim)) * 1. / np.sqrt(in_dim)

    g, _, _ = compact_svd(xs)

    w1s = np.zeros((steps, hidden_dim, in_dim))
    w2s = np.zeros((steps, out_dim, hidden_dim))

    for i in range(steps*2 - 1):
        r_ = (1. - alpha) * r_ + alpha * jax.random.normal(RNGKey(), r_.shape) * 5.
        gamma = (1. - alpha) * gamma + alpha * jax.random.normal(RNGKey(), gamma.shape) * 5.

        gamma = gamma @ (np.identity(g.shape[0]) - g @ g.T)

        r1, _, r2 = np.linalg.svd(r_, False)
        r = r1 @ r2
        w1 = r @ np.sqrt(n) @ o.T @ np.linalg.pinv(xs) + gamma
        w2 = m @ np.sqrt(n) @ r.T

        if i >= steps - 1:
            w1s = w1s.at[i-steps+1].set(w1)
            w2s = w2s.at[i-steps+1].set(w2)

    return w1s, w2s
"""


def walk_mrns(xs, ys, hidden_dim, steps, alpha=0.00625):
    in_dim, n = xs.shape

    m, n, o = compact_svd(ys @ xs.T @ np.linalg.pinv(xs).T)
    r_ = jax.random.normal(RNGKey(), (hidden_dim, n.shape[0]))
    g2 = jax.random.normal(RNGKey(), (hidden_dim, in_dim)) * 1.0 / np.sqrt(in_dim)

    a, _, _ = compact_svd(xs)
    pu = np.identity(in_dim) - a @ a.T

    key = RNGKey()

    def step(carry, _):
        r_, g2, key = carry
        key, k1, k2 = jax.random.split(key, 3)

        r_ = (1.0 - alpha) * r_ + alpha * jax.random.normal(k1, r_.shape) * 5.0
        g2 = (1.0 - alpha) * g2 + alpha * jax.random.normal(k2, g2.shape) * 5.0

        r1, _, r2 = np.linalg.svd(r_, False)
        r = r1 @ r2
        w1 = r @ np.sqrt(n) @ o.T @ np.linalg.pinv(xs) + g2 @ pu
        w2 = m @ np.sqrt(n) @ r.T

        return (r_, g2, key), (w1, w2)

    step = jax.jit(step)

    (r_, g2, key), _ = jax.lax.scan(step, (r_, g2, key), None, length=_warmup)
    _, (w1s, w2s) = jax.lax.scan(step, (r_, g2, key), None, length=steps)

    return w1s, w2s


"""
def walk_mwns(xs, ys, hidden_dim, steps, alpha=0.0125):
    in_dim = xs.shape[0]
    out_dim = ys.shape[0]
    u, s, v = compact_svd(ys @ xs.T @ np.linalg.pinv(xs @ xs.T))
    r_ = jax.random.normal(RNGKey(), (hidden_dim, s.shape[0]))

    w1s = np.zeros((steps, hidden_dim, in_dim))
    w2s = np.zeros((steps, out_dim, hidden_dim))

    for i in range(steps*2 - 1):
        r_ = (1. - alpha) * r_ + alpha * jax.random.normal(RNGKey(), r_.shape) * 5.
        r1, _, r2 = np.linalg.svd(r_, False)
        r = r1 @ r2
        w1 = r @ np.sqrt(s) @ v.T
        w2 = u @ np.sqrt(s) @ r.T

        if i >= steps - 1:
            w1s = w1s.at[i-steps+1].set(w1)
            w2s = w2s.at[i-steps+1].set(w2)

    return w1s, w2s
"""


def walk_mwns(xs, ys, hidden_dim, steps, alpha=0.00625):
    u, s, v = compact_svd(ys @ xs.T @ np.linalg.pinv(xs @ xs.T))
    r_ = jax.random.normal(RNGKey(), (hidden_dim, s.shape[0]))

    key = RNGKey()

    def step(carry, _):
        r_, key = carry
        key, k1 = jax.random.split(key, 2)

        r_ = (1.0 - alpha) * r_ + alpha * jax.random.normal(k1, r_.shape) * 5.0

        r1, _, r2 = np.linalg.svd(r_, False)
        r = r1 @ r2
        w1 = r @ np.sqrt(s) @ v.T
        w2 = u @ np.sqrt(s) @ r.T

        return (r_, key), (w1, w2)

    step = jax.jit(step)

    (r_, key), _ = jax.lax.scan(step, (r_, key), None, length=_warmup)
    _, (w1s, w2s) = jax.lax.scan(step, (r_, key), None, length=steps)

    return w1s, w2s
