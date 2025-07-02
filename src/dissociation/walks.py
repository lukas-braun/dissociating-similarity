from .utils import sample_weights, jitable_compact_svd

import jax
import jax.numpy as np


_warmup = 200


def walk_gls(rng, xs, ys, hidden_dim, steps, alpha=5e-3):
    in_dim, n = xs.shape
    out_dim = ys.shape[0]

    std1, std2 = 1. / np.sqrt(in_dim), 1. / np.sqrt(hidden_dim)
    rng, key = jax.random.split(rng)
    w1, w2 = sample_weights(key, in_dim, hidden_dim, out_dim, std1, std2)

    def step(carry, _):
        w1, w2, rng = carry
        rng, key1, key2 = jax.random.split(rng, 3)

        w1 = (1. - alpha) * w1 + alpha * jax.random.normal(key1, w1.shape) * 5.
        w2 = (1. - alpha) * w2 + alpha * jax.random.normal(key2, w2.shape) * 5.

        def condition(state, threshold=1e-6):
            w1, w2 = state
            return np.linalg.norm(w2 @ w1 @ xs - ys, ord="fro")**2 > threshold

        def gd(state, lr=0.2):
            w1, w2 = state
            loss = 1. / n * (w2 @ w1 @ xs - ys) @ xs.T
            dw1 = w2.T @ loss
            dw2 = loss @ w1.T
            return w1 - lr * dw1, w2 - lr * dw2

        w1, w2 = jax.lax.while_loop(condition, gd, (w1, w2))
        return (w1, w2, rng), (w1, w2)

    (w1, w2, rng), _ = jax.lax.scan(step, (w1, w2, rng), None, length=_warmup)
    _, (w1s, w2s) = jax.lax.scan(step, (w1, w2, rng), None, length=steps)

    return w1s, w2s


walk_gls = jax.jit(walk_gls, static_argnums=range(3, 6))


def walk_lss(rng, xs, ys, hidden_dim, steps, alpha=5e-3):
    in_dim, n = xs.shape
    out_dim = ys.shape[0]

    std1, std2 = 1. / np.sqrt(in_dim), 1. / np.sqrt(hidden_dim)
    rng, key = jax.random.split(rng)
    w1, w2 = sample_weights(key, in_dim, hidden_dim, out_dim, std1, std2)

    u, s, v = jitable_compact_svd(ys @ xs.T @ np.linalg.pinv(xs @ xs.T))
    a, _, _ = jitable_compact_svd(xs)
    pr = v @ v.T
    pi = a @ a.T - v @ v.T
    pu = np.identity(in_dim) - a @ a.T

    def step(carry, _):
        w1, w2, rng = carry
        rng, key1, key2 = jax.random.split(rng, 3)

        w1 = (1. - alpha) * w1 + alpha * jax.random.normal(key1, w1.shape) * 5.
        w2 = (1. - alpha) * w2 + alpha * jax.random.normal(key2, w2.shape) * 5.

        def condition(state, threshold=1e-6):
            w1, w2 = state
            return np.linalg.norm(w2 @ w1 @ xs - ys, ord="fro")**2 > threshold

        def gd(state, lr=0.2):
            w1, w2 = state
            loss = 1. / n * (w2 @ w1 @ xs - ys) @ xs.T
            dw1 = w2.T @ loss
            dw2 = loss @ w1.T
            return w1 - lr * dw1, w2 - lr * dw2

        w1, w2 = jax.lax.while_loop(condition, gd, (w1, w2))

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

        return (w1, w2, rng), (w1, w2)

    (w1, w2, rng), _ = jax.lax.scan(step, (w1, w2, rng), None, length=_warmup)
    _, (w1s, w2s) = jax.lax.scan(step, (w1, w2, rng), None, length=steps)

    return w1s, w2s


walk_lss = jax.jit(walk_lss, static_argnums=range(3, 6))


def walk_mrns(rng, xs, ys, hidden_dim, steps, alpha=5e-3):
    in_dim, n = xs.shape

    m, n, o = jitable_compact_svd(ys @ xs.T @ np.linalg.pinv(xs).T)
    rng, key1, key2 = jax.random.split(rng, 3)
    r_ = jax.random.normal(key1, (hidden_dim, n.shape[0]))
    g2 = jax.random.normal(key2, (hidden_dim, in_dim)) * 1.0 / np.sqrt(in_dim)

    a, _, _ = jitable_compact_svd(xs)
    pu = np.identity(in_dim) - a @ a.T

    def step(carry, _):
        r_, g2, rng = carry
        rng, key1, key2 = jax.random.split(rng, 3)

        r_ = (1. - alpha) * r_ + alpha * jax.random.normal(key1, r_.shape) * 5.
        g2 = (1. - alpha) * g2 + alpha * jax.random.normal(key2, g2.shape) * 5.

        r1, _, r2 = np.linalg.svd(r_, False)
        r = r1 @ r2
        w1 = r @ np.sqrt(n) @ o.T @ np.linalg.pinv(xs) + g2 @ pu
        w2 = m @ np.sqrt(n) @ r.T

        return (r_, g2, rng), (w1, w2)

    (r_, g2, rng), _ = jax.lax.scan(step, (r_, g2, rng), None, length=_warmup)
    _, (w1s, w2s) = jax.lax.scan(step, (r_, g2, rng), None, length=steps)

    return w1s, w2s


walk_mrns = jax.jit(walk_mrns, static_argnums=range(3, 6))


def walk_mwns(rng, xs, ys, hidden_dim, steps, alpha=5e-3):
    u, s, v = jitable_compact_svd(ys @ xs.T @ np.linalg.pinv(xs @ xs.T))
    rng, key = jax.random.split(rng)
    r_ = jax.random.normal(key, (hidden_dim, s.shape[0]))

    def step(carry, _):
        r_, rng = carry
        rng, key1 = jax.random.split(rng, 2)

        r_ = (1. - alpha) * r_ + alpha * jax.random.normal(key1, r_.shape) * 5.

        r1, _, r2 = np.linalg.svd(r_, False)
        r = r1 @ r2
        w1 = r @ np.sqrt(s) @ v.T
        w2 = u @ np.sqrt(s) @ r.T

        return (r_, rng), (w1, w2)

    (r_, rng), _ = jax.lax.scan(step, (r_, rng), None, length=_warmup)
    _, (w1s, w2s) = jax.lax.scan(step, (r_, rng), None, length=steps)

    return w1s, w2s


walk_mwns = jax.jit(walk_mwns, static_argnums=range(3, 6))
