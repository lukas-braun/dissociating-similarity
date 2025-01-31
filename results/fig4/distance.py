"""Distance measures."""

import jax.numpy as jnp
from jaxtyping import Array


def _constant_like(x: Array, const: float | Array) -> Array:
    """Return a constant array with the same shape and dtype as `x`."""
    return jnp.array(const, dtype=x.dtype)


def vector_norm(x: Array, p: int | float) -> Array:
    """Compute the vector norm of a tensor `x` treated as a vector."""
    if x.size == 0:
        msg = "Cannot compute norm of an empty array."
        raise ValueError(msg)

    axes = tuple(range(x.ndim))

    if p == jnp.inf:
        return jnp.amax(jnp.abs(x), axis=axes, keepdims=False)
    if p == -jnp.inf:
        return jnp.amin(jnp.abs(x), axis=axes, keepdims=False)

    match p:
        case 2:
            return jnp.sqrt(
                jnp.sum(jnp.real(x * jnp.conj(x)), axis=axes, keepdims=False),
            )

        case 1:
            return jnp.sum(jnp.abs(x), axis=axes, keepdims=False)

        case 0:
            return jnp.sum(x != 0, dtype=x.dtype, axis=axes, keepdims=False)

        case _:
            if x.dtype in (jnp.int32, jnp.int64):
                msg = "Cannot compute general p-norm for integer tensors."
                raise ValueError(msg)
            abs_x = jnp.abs(x)
            p_arr = _constant_like(abs_x, p)
            p_inv = _constant_like(abs_x, 1.0 / p_arr)
            out = jnp.sum(abs_x**p_arr, axis=axes, keepdims=False)
            return jnp.power(out, p_inv)


def difference_norm(x: Array, y: Array, p: int) -> Array:
    """Compute the vector norm of the difference between `x` and `y`."""
    return vector_norm(x - y, p=p)


def dot_product(x: Array, y: Array) -> Array:
    """Compute the dot product of `x` and `y` matched as vectors."""
    axes = (tuple(range(x.ndim)), tuple(range(y.ndim)))
    return jnp.tensordot(x, y, axes=axes)


def cosine_similarity(x: Array, y: Array) -> Array:
    """Compute the safe cosine similarity of `x` and `y` matched as vectors."""
    norm_x = vector_norm(x, p=2)
    norm_y = vector_norm(y, p=2)

    nonzero_norm_x = jnp.greater(norm_x, 0.0)
    nonzero_norm_y = jnp.greater(norm_y, 0.0)
    valid_norm = jnp.logical_and(nonzero_norm_x, nonzero_norm_y)

    cosine_similarity = dot_product(x, y) / (norm_x * norm_y)
    return jnp.where(valid_norm, cosine_similarity, 0.0)


def cosine_distance(x: Array, y: Array) -> Array:
    """Compute the safe cosine distance of `x` and `y` matched as vectors."""
    return 1 - cosine_similarity(x, y)


def angular_distance(x: Array, y: Array) -> Array:
    """Compute the angular distance between `x` and `y` matched as vectors."""
    return jnp.arccos(jnp.clip(cosine_similarity(x, y), -1.0, 1.0))


def pearson_correlation(x: Array, y: Array) -> Array:
    """Pearson correlation of `x` and `y` matched as vectors."""
    return cosine_similarity(
        x - jnp.mean(x, keepdims=True),
        y - jnp.mean(y, keepdims=True),
    )


def correlation_distance(x: Array, y: Array) -> Array:
    """The deviation of the Pearson correlation from 1."""
    return 1 - pearson_correlation(x, y)
