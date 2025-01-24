import array
import gzip
import os
import struct
import urllib.request
from os import path
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import numpy as np


_DATA = "/tmp/jax_example_data/"


def _download(url, filename):
    """Download a url to a file in the JAX data temp directory."""
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        print(f"downloaded {url} to {_DATA}")


def _partial_flatten(x):
    """Flatten all but the first dimension of an ndarray."""
    return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw():
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        _download(base_url + filename, filename)

    train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
    """Download, parse and process MNIST data to unit scale and one-hot labels."""
    train_images, train_labels, test_images, test_labels = mnist_raw()

    train_images = _partial_flatten(train_images) / np.float32(255.0)
    test_images = _partial_flatten(test_images) / np.float32(255.0)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels


def cross_entropy(pred_y, y):
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(pred_y), axis=1))


def batch_objective(params, args):
    static, X, y = args
    model = eqx.combine(params, static)
    pred_y = eqx.filter_vmap(model)(X)
    objective_value = cross_entropy(pred_y, y)
    return objective_value, None


def train_mnist(
    max_iterations: int = 100,
    # TODO: Use the full dataset.
    subset=128,
):
    train_images, train_labels, test_images, test_labels = mnist()

    X = jnp.array(train_images[:subset])
    y = jnp.array(train_labels[:subset])

    model = eqx.nn.MLP(
        in_size=784,  # 28*28
        width_size=512,
        out_size=10,
        depth=2,
        activation=jax.nn.relu,
        # TODO: Use biases
        use_bias=False,
        use_final_bias=False,
        key=jax.random.PRNGKey(0),
    )

    optimizer = optx.GradientDescent(learning_rate=1e-1, rtol=1e-4, atol=1e-4)
    options = None
    f_struct = jax.ShapeDtypeStruct((), jnp.float32)
    tags = frozenset()

    init = eqx.Partial(
        optimizer.init,
        fn=batch_objective,
        options=options,
        f_struct=f_struct,
        aux_struct=None,
        tags=tags,
    )
    step = eqx.Partial(optimizer.step, fn=batch_objective, options=options, tags=tags)
    terminate = eqx.Partial(
        optimizer.terminate, fn=batch_objective, options=options, tags=tags
    )
    postprocess = eqx.Partial(
        optimizer.postprocess, fn=batch_objective, options=options, tags=tags
    )

    params, static = eqx.partition(model, eqx.is_array)
    state = init(y=params, args=(static, X, y))
    done, result = terminate(y=params, args=(static, X, y), state=state)

    iteration = 0
    while not done and iteration < max_iterations:
        params, state, _ = step(y=params, args=(static, X, y), state=state)
        done, result = terminate(y=params, args=(static, X, y), state=state)
        loss, _ = batch_objective(params, (static, X, y))
        print(f"Iteration {iteration}, loss: {loss}")
        iteration += 1

    if result != optx.RESULTS.successful:
        print("Optimization failed!")

    params, _, _ = postprocess(
        y=params, aux=None, args=(static, X, y), state=state, result=result
    )
    model = eqx.combine(params, static)

    return model


if __name__ == "__main__":
    model = train_mnist()
