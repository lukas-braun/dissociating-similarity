# ruff: noqa: F722
import jax
import jax.numpy as jnp

from jaxtyping import Array, Float, PRNGKeyArray


def scale_neurons(
    w_in: Float[Array, "hidden in"],
    w_out: Float[Array, "out hidden"],
    scale_factor: float = 0.1,
) -> tuple[Float[Array, "hidden in"], Float[Array, "out hidden"]]:
    """Scale the input and output weights of a neuron per Dinh et al. (2017)."""
    w_in_scaled = scale_factor * w_in
    w_out_scaled = w_out / scale_factor
    return w_in_scaled, w_out_scaled


def duplicate_neurons(
    w_in: Float[Array, "hidden in"],
    w_out: Float[Array, "out hidden"],
    num_duplicates: int = 5,
) -> tuple[Float[Array, "hidden in"], Float[Array, "out hidden"]]:
    """Add duplicate-type neurons per Simsek et al. (2021)."""
    hidden_dim, in_dim = w_in.shape
    out_dim, _ = w_out.shape
    w_in_dup = jnp.repeat(w_in, num_duplicates, axis=0)
    w_out_dup = jnp.repeat(w_out / num_duplicates, num_duplicates, axis=1)
    return w_in_dup, w_out_dup


def add_random_zero_neurons(
    w_in: Float[Array, "hidden in"],
    w_out: Float[Array, "out hidden"],
    num_zero_groups: int = 3,
    neurons_per_group: int = 4,
    *,
    key: PRNGKeyArray,
) -> tuple[Float[Array, "hidden in"], Float[Array, "out hidden"]]:
    """Add randomized zero-type neurons per Simsek et al. (2021)."""
    hidden_dim, in_dim = w_in.shape
    out_dim, _ = w_out.shape

    zero_w_in = jax.random.normal(key, (1, in_dim))
    zero_w_in = jnp.repeat(zero_w_in, num_zero_groups * neurons_per_group, axis=0)

    zero_w_out = jnp.zeros((w_out.shape[0], num_zero_groups * neurons_per_group))
    group_weights = jnp.array([1.0, -1.0] + [0.0] * (neurons_per_group - 2))
    zero_w_out = jnp.tile(group_weights, (out_dim, num_zero_groups))

    new_w_in = jnp.concatenate([w_in, zero_w_in], axis=0)
    new_w_out = jnp.concatenate([w_out, zero_w_out], axis=1)

    return new_w_in, new_w_out


def add_phantom_zero_neurons(
    w_in: Float[Array, "hidden in"],
    w_out: Float[Array, "out hidden"],
    preacts: Float[Array, "batch hidden"],
    pattern: Float[Array, "batch pattern"],
) -> tuple[Float[Array, "hidden in"], Float[Array, "out hidden"]]:
    """Add zero-type neurons to create a phantom pattern over preactivations."""
    hidden_dim, in_dim = w_in.shape
    out_dim, _ = w_out.shape
    _, phantom_dim = pattern.shape

    mapping, *_ = jnp.linalg.lstsq(preacts, pattern)
    phantom_w_in, *_ = jnp.linalg.lstsq(w_in, mapping)
    phantom_w_in = phantom_w_in.T
    phantom_w_out = jnp.zeros((out_dim, phantom_dim))

    new_w_in = jnp.concatenate([w_in, phantom_w_in], axis=0)
    new_w_out = jnp.concatenate([w_out, phantom_w_out], axis=1)

    return new_w_in, new_w_out
