from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import PRNGKeyArray

from crossval import kfold_op
from regression import RegressionSolver


@partial(jax.jit, static_argnums=(2, 3))
def solve_softmax(
    x: Array, y: Array, max_iter: int = 200, lr: float = 0.2
) -> tuple[Array, Array]:
    """Solve softmax regression using gradient descent for both coefficients and intercept."""

    beta = jnp.zeros((x.shape[1], y.shape[1]))
    intercept = jnp.zeros((1, y.shape[1]))

    def softmax(z: Array) -> Array:
        z_exp = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
        return z_exp / jnp.sum(z_exp, axis=1, keepdims=True)

    def loss(params: tuple[Array, Array]) -> Array:
        beta, intercept = params
        logits = jnp.dot(x, beta) + intercept
        probs = softmax(logits)
        return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-10), axis=1))

    for _ in range(max_iter):
        grads = jax.grad(loss)((beta, intercept))
        grad_beta, grad_intercept = grads
        beta = beta - lr * grad_beta
        intercept = intercept - lr * grad_intercept

    return beta, intercept


def predict_softmax(x_train: Array, y_train: Array, x_test: Array) -> Array:
    """Make predictions using softmax regression."""
    beta, intercept = solve_softmax(x_train, y_train)
    logits = jnp.dot(x_test, beta) + intercept
    return jax.nn.softmax(logits)


def score_softmax(
    x_train: Array, y_train: Array, x_test: Array, y_test: Array
) -> Array:
    """Score softmax regression on the test set using accuracy."""
    predictions = predict_softmax(x_train, y_train, x_test)
    predicted_classes = jnp.argmax(predictions, axis=1)
    actual_classes = jnp.argmax(y_test, axis=1)
    return jnp.mean(predicted_classes == actual_classes)


def crossval_softmax_score(
    x: Array,
    y: Array,
    num_splits: int = 10,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Compute the accuracy score for cross-validation of softmax regression."""
    return kfold_op(x=x, y=y, num_splits=num_splits, fold_op=score_softmax)


def averaged_crossval_softmax_score(
    x: Array,
    y: Array,
    num_splits: int = 10,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Compute average accuracy score for cross-validated folds of softmax regression."""
    return jnp.mean(crossval_softmax_score(x, y, num_splits=num_splits, key=key))


def _predict_without_test_labels(
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
    solver: RegressionSolver,
) -> Array:
    """Make predictions without using test labels."""
    del y_test
    beta, intercept = solver(x_train, y_train)
    logits = jnp.dot(x_test, beta) + intercept
    return jax.nn.softmax(logits)


def _kfold_predict(
    x: Array,
    y: Array,
    num_splits: int,
    solver: RegressionSolver,
) -> Array:
    """Make predictions using k-fold cross-validation."""
    fold_predictions = kfold_op(
        x=x,
        y=y,
        num_splits=num_splits,
        fold_op=partial(
            _predict_without_test_labels,
            solver=solver,
        ),
    )
    return fold_predictions.reshape((x.shape[0], -1))


def _kfold_score(
    x: Array,
    y: Array,
    num_splits: int,
    solver: RegressionSolver,
) -> Array:
    """Compute accuracy score for cross-validation."""
    return kfold_op(
        x=x,
        y=y,
        num_splits=num_splits,
        fold_op=partial(
            score_softmax,
            solver=solver,
        ),
    )


def crossval_softmax_predict(
    x: Array,
    y: Array,
    num_splits: int = 10,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Make cross-validated predictions with softmax regression."""
    del key
    return _kfold_predict(x=x, y=y, num_splits=num_splits, solver=solve_softmax)
