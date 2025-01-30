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
import optax
import numpy as np
from functools import partial


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


def mnist_raw_data():
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


def mnist_data(permute_train=False):
    """Download, parse and process MNIST data to unit scale and one-hot labels."""
    train_images, train_labels, test_images, test_labels = mnist_raw_data()

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


def accuracy(pred_y, y):
    target_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(pred_y, axis=1)
    return jnp.mean(predicted_class == target_class)


def batch_objective(params, args):
    static, X, y = args
    model = eqx.combine(params, static)
    pred_y = eqx.filter_vmap(model)(X)
    objective_value = cross_entropy(pred_y, y)
    return objective_value, None


@partial(eqx.filter_vmap, in_axes=(eqx.if_array(0), None))
@partial(eqx.filter_vmap, in_axes=(None, 0))
def predict(model, X):
    return model(X)


@eqx.filter_vmap
def create_model(key):
    # TODO use param_scale = 0.1
    model = eqx.nn.MLP(
        in_size=784,  # 28 * 28
        width_size=1024,
        out_size=10,
        depth=2,
        activation=jax.nn.relu,
        use_bias=False,
        use_final_bias=False,
        key=key,
    )
    return model


def create_optimizer(num_steps):
    # encapsulate optimistix boilerplate
    optimizer = optx.OptaxMinimiser(
        optax.sgd(
            learning_rate=optax.linear_schedule(
                init_value=1e-1,
                end_value=1e-6,
                transition_steps=num_steps,
            )
        ),
        rtol=1e-4,
        atol=1e-4,
    )
    options = None
    f_struct = jax.ShapeDtypeStruct((), jnp.float32)
    tags = frozenset()

    @partial(eqx.filter_vmap, in_axes=(0, None, None, None))
    def init(params, static, X, y):
        return optimizer.init(
            fn=batch_objective,
            y=params,
            args=(static, X, y),
            options=options,
            f_struct=f_struct,
            aux_struct=None,
            tags=tags,
        )

    @partial(eqx.filter_vmap, in_axes=(0, None, None, None, 0))
    def step(params, static, X, y, state):
        return optimizer.step(
            fn=batch_objective,
            y=params,
            args=(static, X, y),
            options=options,
            state=state,
            tags=tags,
        )

    @partial(eqx.filter_vmap, in_axes=(0, None, None, None, 0))
    def terminate(params, static, X, y, state):
        return optimizer.terminate(
            fn=batch_objective,
            y=params,
            args=(static, X, y),
            options=options,
            state=state,
            tags=tags,
        )

    @partial(eqx.filter_vmap, in_axes=(0, None, None, None, 0, 0))
    def postprocess(params, static, X, y, state, result):
        return optimizer.postprocess(
            fn=batch_objective,
            aux=None,
            y=params,
            args=(static, X, y),
            options=options,
            state=state,
            result=result,
            tags=tags,
        )

    return init, step, terminate, postprocess


def train_mnist(
    *,
    num_epochs: int = 30,
    num_seeds: int = 10,
    key=jax.random.PRNGKey(37),
):
    train_images, train_labels, test_images, test_labels = mnist_data()
    batch_size = 128
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = np.random.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]

    batches = data_stream()

    model = create_model(jax.random.split(key, num_seeds))
    init, step, terminate, postprocess = create_optimizer(
        num_steps=num_epochs * num_batches
    )

    X, y = train_images[:batch_size], train_labels[:batch_size]
    params, static = eqx.partition(model, eqx.is_array)
    state = init(params, static, X, y)
    done, result = terminate(params, static, X, y, state)

    loss, _ = eqx.filter_vmap(batch_objective, in_axes=(0, None))(
        params, (static, X, y)
    )
    model = eqx.combine(params, static)
    pred_y = predict(model, test_images)
    test_error = 1 - jax.vmap(accuracy, in_axes=(0, None))(pred_y, test_labels)
    loss_mean, loss_std = loss.mean(), loss.std()
    error_mean, error_std = test_error.mean(), test_error.std()
    print(
        f"Epoch 0\t\ttrain loss: {loss_mean:.3g} ± {loss_std:.3g}\ttest error: {error_mean*100:.4g}%"
    )

    epoch = 0
    for epoch in range(1, num_epochs + 1):
        for _ in range(num_batches):
            X, y = next(batches)
            params, state, _ = step(params, static, X, y, state)
            done, result = terminate(params, static, X, y, state)

        loss, _ = eqx.filter_vmap(batch_objective, in_axes=(0, None))(
            params, (static, X, y)
        )
        print(f"Epoch {epoch}\t\ttrain loss: {loss.mean():.3g} ± {loss.std():.3g}")

    model = eqx.combine(params, static)
    pred_y = predict(model, test_images)
    test_error = 1 - jax.vmap(accuracy, in_axes=(0, None))(pred_y, test_labels)
    loss_mean, loss_std = loss.mean(), loss.std()
    error_mean, error_std = test_error.mean(), test_error.std()
    print(
        f"Epoch {epoch}\t\ttrain loss: {loss_mean:.3g} ± {loss_std:.3g}\ttest error: {error_mean*100:.4g}%"
    )

    # TODO: Adapt to multi-dim case
    # if result != optx.RESULTS.successful:
    #    print("Optimization failed!")

    params, _, _ = postprocess(params, static, X, y, state, result)
    model = eqx.combine(params, static)
    return model


if __name__ == "__main__":
    model = train_mnist()
