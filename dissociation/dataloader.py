import jax.numpy as np
import numpy as onp

import os
import struct


def load_mnist(flat=True, normalised=True, one_hot=True, batch_last=True):
    def load_binaries(file):
        with open(file, "rb") as fd:
            check, items_n = struct.unpack(">ii", fd.read(8))

            if check == 2051 and "images" in file:
                h, w = struct.unpack(">II", fd.read(8))
                return np.asarray(
                    onp.fromfile(fd, dtype="uint8").reshape(items_n, h, w)
                )
            elif check == 2049 and "labels" in file:
                return np.asarray(onp.fromfile(fd, dtype="uint8"))

    file_names = [
        "train-images.idx3-ubyte",
        "train-labels.idx1-ubyte",
        "t10k-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte",
    ]

    mnist = []
    for file_name in file_names:
        file = os.path.join("../data/mnist/", file_name)
        mnist.append(load_binaries(file))

    if flat:
        mnist[0] = mnist[0].reshape(-1, 28 * 28)
        mnist[2] = mnist[2].reshape(-1, 28 * 28)

    if normalised:
        # mean = np.mean(mnist[0])
        # std = np.std(mnist[0])

        # for i in [0, 2]:
        #    mnist[i] -= mean
        #    mnist[i] /= (std + 1e-8)

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
