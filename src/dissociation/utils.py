import jax
import jax.numpy as np


def sample_weights(rng, in_dim, hidden_dim, out_dim, std1=1.0, std2=1.0):
    key1, key2 = jax.random.split(rng)
    w1 = jax.random.normal(key1, (hidden_dim, in_dim)) * std1
    w2 = jax.random.normal(key2, (out_dim, hidden_dim)) * std2
    return w1, w2


def random_regression_task(rng, in_dim, out_dim, n):
    key1, key2 = jax.random.split(rng)
    xs = jax.random.normal(key1, (in_dim, n)) / np.sqrt(in_dim)
    ys = jax.random.normal(key2, (out_dim, n)) / np.sqrt(out_dim)
    return xs, ys


def jitable_compact_svd(a, threshold=1e-6):
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    mask = s > threshold
    return u * mask, np.diag(s * mask), vt.T * mask
