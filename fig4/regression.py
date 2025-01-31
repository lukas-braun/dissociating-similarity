"""Coefficient of determination (R^2) of linear regression."""

from collections.abc import Callable
from functools import partial
from typing import Protocol
from typing import cast

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import PRNGKeyArray

from crossval import kfold_op


class RegressionSolver(Protocol):
    """Solver for regressions."""

    def __call__(self, x: Array, y: Array) -> tuple[Array, ...]:
        """Regress `y` on to `x`."""
        ...


def _solve_normal_equations(x: Array, y: Array, alpha: float = 0.0) -> Array:
    """Solve the normal equations for (regularized) linear regression."""
    # beta = inv(X^t X + alpha * Id) * X.T y
    xx = jnp.dot(x.T, x)
    xy = jnp.dot(x.T, y)
    coefficient = jnp.linalg.solve(xx + alpha * jnp.identity(xx.shape[0]), xy)
    return cast(Array, coefficient).T


def _solve_lstsq(x: Array, y: Array) -> Array:
    """Solve the least squares linear regression problem."""
    coefficient, _, _, _ = jnp.linalg.lstsq(x, y, rcond=None)
    return coefficient.T


def _solve_affine(
    x: Array, y: Array, coefficient_solver: Callable[[Array, Array], Array]
) -> tuple[Array, Array]:
    # Center the data.
    x_offset = jnp.mean(x, axis=0)
    x -= x_offset

    y_offset = jnp.mean(y, axis=0)
    y -= y_offset

    # Fit intercept.
    coefficient = coefficient_solver(x, y)
    intercept = y_offset - jnp.dot(x_offset, coefficient.T)

    return coefficient, intercept


def solve_ols(x: Array, y: Array) -> tuple[Array, Array]:
    """Solve ordinary least squares using least squares or the normal equations."""
    if x.shape[0] < x.shape[1]:
        return _solve_affine(x, y, coefficient_solver=_solve_lstsq)
    else:
        return _solve_affine(
            x,
            y,
            coefficient_solver=partial(_solve_normal_equations, alpha=0.0),
        )


def solve_ridge(x: Array, y: Array, alpha: float = 1.0) -> tuple[Array, Array]:
    """Solve ridge regression using the normal equations."""
    return _solve_affine(
        x, y, coefficient_solver=partial(_solve_normal_equations, alpha=alpha)
    )


def _predict(
    x_train: Array,
    y_train: Array,
    x_test: Array,
    solver: RegressionSolver,
) -> Array:
    """Make predictions using the solver."""
    coefficient, intercept = solver(x_train, y_train)
    return jnp.dot(x_test, coefficient.T) + intercept


def predict_ols(x_train: Array, y_train: Array, x_test: Array) -> Array:
    """Make predictions using ordinary least squares."""
    solver = solve_ols
    return _predict(x_train=x_train, y_train=y_train, x_test=x_test, solver=solver)


def predict_ridge(
    x_train: Array, y_train: Array, x_test: Array, alpha: float = 1.0
) -> Array:
    """Make predictions using ridge regression."""
    solver = partial(solve_ridge, alpha=alpha)
    return _predict(x_train=x_train, y_train=y_train, x_test=x_test, solver=solver)


def difference_norm(x: Array, y: Array) -> Array:
    """Compute the vector norm of the difference between `x` and `y`."""
    return jnp.linalg.norm(x - y, axis=-1)


def _r2_score(y_true: Array, y_pred: Array) -> Array:
    """Coefficient of determination (R^2) for linear regression."""
    rss = difference_norm(y_true, y_pred) ** 2
    tss = difference_norm(y_true, jnp.mean(y_true, axis=0)) ** 2
    r2 = 1 - rss / tss
    return r2


def _score(
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
    solver: RegressionSolver,
) -> Array:
    """Compute the R^2 score of `solver` on the given train-test split."""
    y_pred = _predict(x_train=x_train, y_train=y_train, x_test=x_test, solver=solver)
    return _r2_score(y_test, y_pred)


def score_ols(x_train: Array, y_train: Array, x_test: Array, y_test: Array) -> Array:
    """Score ordinary least squares on the test set."""
    solver = solve_ols
    return _score(x_train, y_train, x_test, y_test, solver)


def score_ridge(
    x_train: Array, y_train: Array, x_test: Array, y_test: Array, alpha: float = 1.0
) -> Array:
    """Score ridge regression on the test set."""
    solver = partial(solve_ridge, alpha=alpha)
    return _score(x_train, y_train, x_test, y_test, solver)


def _kfold_predict(
    x: Array,
    y: Array,
    num_splits: int,
    solver: RegressionSolver,
) -> Array:
    """Compute the R^2 score for cross-validation of `solver` with `num_splits` folds."""

    def _predict_without_test_labels(
        x_train: Array,
        y_train: Array,
        x_test: Array,
        y_test: Array,
        solver: RegressionSolver,
    ) -> Array:
        del y_test
        return _predict(x_train=x_train, y_train=y_train, x_test=x_test, solver=solver)

    fold_predictions = kfold_op(
        x=x,
        y=y,
        num_splits=num_splits,
        fold_op=partial(
            _predict_without_test_labels,
            solver=solver,
        ),
    )
    return fold_predictions.reshape((x.shape[0], y.shape[1]))


def _kfold_score(
    x: Array,
    y: Array,
    num_splits: int,
    solver: RegressionSolver,
) -> Array:
    """Compute the R^2 score for cross-validation of `solver` with `num_splits` folds."""
    return kfold_op(
        x=x,
        y=y,
        num_splits=num_splits,
        fold_op=partial(
            _score,
            solver=solver,
        ),
    )


def crossval_ols_predict(
    x: Array,
    y: Array,
    num_splits: int = 10,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Make cross-validated predictions with ordinary least squares."""
    del key
    solver = solve_ols
    return _kfold_predict(x=x, y=y, num_splits=num_splits, solver=solver)


def crossval_ridge_predict(
    x: Array,
    y: Array,
    num_splits: int = 10,
    alpha: float = 1.0,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Make cross-validated predictions with ridge regression."""
    del key
    solver = partial(solve_ridge, alpha=alpha)
    return _kfold_predict(x=x, y=y, num_splits=num_splits, solver=solver)


def crossval_ols_score(
    x: Array,
    y: Array,
    num_splits: int = 10,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Compute the R^2 score for cross-validation of ordinary least squares."""
    del key
    solver = solve_ols
    return _kfold_score(x=x, y=y, num_splits=num_splits, solver=solver)


def averaged_crossval_ols_score(
    x: Array,
    y: Array,
    num_splits: int = 10,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Compute average R^2 score for cross-validated folds of ordinary least squares."""
    return jnp.mean(crossval_ols_score(x, y, num_splits=num_splits, key=key))


def crossval_ridge_score(
    x: Array,
    y: Array,
    num_splits: int = 10,
    alpha: float = 1.0,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Compute the R^2 score for cross-validation of ridge regression."""
    del key
    solver = partial(solve_ridge, alpha=alpha)
    return _kfold_score(x=x, y=y, num_splits=num_splits, solver=solver)


def averaged_crossval_ridge_score(
    x: Array,
    y: Array,
    num_splits: int = 10,
    alpha: float = 1.0,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Compute average R^2 score for cross-validated folds of ridge regression."""
    return jnp.mean(
        crossval_ridge_score(x, y, num_splits=num_splits, alpha=alpha, key=key)
    )
