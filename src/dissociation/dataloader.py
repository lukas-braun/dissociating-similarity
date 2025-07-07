from .fig5.mnist import mnist_raw_data

import jax.numpy as np


def load_mnist(flat=True, normalised=True, one_hot=True, batch_last=True):
    mnist = list(mnist_raw_data())

    mnist[0] = mnist[0].astype(float)
    mnist[2] = mnist[2].astype(float)

    if flat:
        mnist[0] = mnist[0].reshape(-1, 28 * 28)
        mnist[2] = mnist[2].reshape(-1, 28 * 28)

    if normalised:
        for i in [0, 2]:
            mnist[i] += 1.0
            mnist[i] /= 256.0

    if one_hot:
        for i in [1, 3]:
            mnist[i] = np.eye(10)[mnist[i]]

    if batch_last:
        for i in range(4):
            mnist[i] = np.transpose(mnist[i], tuple(range(1, mnist[i].ndim)) + (0,))

    return mnist
