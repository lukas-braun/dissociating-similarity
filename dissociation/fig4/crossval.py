"""Utilities for cross-validation."""

from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array


class CrossFoldOp(Protocol):
    """Cross-fold operation protocol."""

    def __call__(
        self,
        x_train: Array,
        y_train: Array,
        x_test: Array,
        y_test: Array,
    ) -> Array:
        """Perform a cross-fold operation."""
        ...


def _fold(
    x: Array, y: Array, start: int, size: int
) -> tuple[Array, Array, Array, Array]:
    """Construct a hold-out fold of size `size` at index `start`."""
    x_shifted = jnp.roll(x, -start, axis=0)
    x_test = jax.lax.dynamic_slice_in_dim(x_shifted, 0, size, axis=0)
    x_train = jax.lax.dynamic_slice_in_dim(x_shifted, size, x.shape[0] - size, axis=0)

    y_shifted = jnp.roll(y, -start, axis=0)
    y_test = jax.lax.dynamic_slice_in_dim(y_shifted, 0, size, axis=0)
    y_train = jax.lax.dynamic_slice_in_dim(y_shifted, size, y.shape[0] - size, axis=0)

    return x_train, y_train, x_test, y_test


def _fold_op(
    x: Array, y: Array, start: int, size: int, cross_fold_op: CrossFoldOp
) -> Array:
    """Perform operation `fold_op` on a single fold defined by `start` and `size`."""
    x_train, y_train, x_test, y_test = _fold(x, y, start, size)
    return cross_fold_op(x_train, y_train, x_test, y_test)


class FoldOp(Protocol):
    """Fold operation protocol."""

    def __call__(self, x: Array, y: Array, start: int, size: int) -> Array:
        """Perform an operation on a single fold defined by `start` and `size`."""
        ...


def kfold_op(x: Array, y: Array, num_splits: int, fold_op: CrossFoldOp) -> Array:
    """Make cross-validated predictions with `solver` over `num_splits` folds."""
    size = x.shape[0] // num_splits
    starts = jnp.arange(num_splits) * size

    vmapped_fold_op = jax.jit(
        jax.vmap(
            partial(
                _fold_op,
                cross_fold_op=fold_op,
            ),
            in_axes=(None, None, 0, None),
        ),
        static_argnums=(3,),
    )

    return vmapped_fold_op(x, y, starts, size)
