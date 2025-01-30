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
    duplicate_multiplier: float = 1.2,
) -> tuple[Float[Array, "hidden in"], Float[Array, "out hidden"]]:
    """Add duplicate-type neurons per Simsek et al. (2021)."""
    hidden_dim, in_dim = w_in.shape
    out_dim, _ = w_out.shape

    int_dups = int(duplicate_multiplier)
    frac_neurons = int((duplicate_multiplier % 1) * hidden_dim)

    w_in_dup = jnp.repeat(w_in, int_dups, axis=0)
    w_out_dup = jnp.repeat(w_out, int_dups, axis=1)
    if frac_neurons > 0:
        w_in_dup = jnp.concatenate([w_in_dup, w_in[:frac_neurons]], axis=0)
        w_out_dup = jnp.concatenate([w_out_dup, w_out[:, :frac_neurons]], axis=1)

    scale = jnp.ones(hidden_dim) * int_dups
    scale = scale.at[:frac_neurons].add(1)
    scale = jnp.repeat(scale, int_dups)
    if frac_neurons > 0:
        scale = jnp.concatenate([scale, scale[:frac_neurons]])
    w_out_dup = w_out_dup / scale[None, :]

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

    if neurons_per_group > 1:
        raise NotImplementedError("neurons_per_group > 1 not yet implemented.")
    # TODO: Counterbalance within groups.
    # group_weights = jnp.array([1.0, -1.0] + [0.0] * (neurons_per_group - 2))
    # zero_w_out = jnp.tile(group_weights, (out_dim, num_zero_groups))

    new_w_in = jnp.concatenate([w_in, zero_w_in], axis=0)
    new_w_out = jnp.concatenate([w_out, zero_w_out], axis=1)

    return new_w_in, new_w_out


# TODO: linear reconstruction for linear nets
# not function-preserving for nonlinear nets.
# def add_linear_reconstruction
# mapping, *_ = jnp.linalg.lstsq(preacts, pattern)
# etched_w_in, *_ = jnp.linalg.lstsq(w_in, mapping)
# etched_w_in = etched_w_in.T


def add_phantom_zero_neurons(
    w_in: Float[Array, "hidden in"],
    w_out: Float[Array, "out hidden"],
    inputs: Float[Array, "batch hidden"],
    pattern: Float[Array, "batch pattern"],
) -> tuple[Float[Array, "hidden in"], Float[Array, "out hidden"]]:
    """Add zero-type neurons to create a phantom pattern over preactivations."""
    hidden_dim, in_dim = w_in.shape
    out_dim, _ = w_out.shape
    _, phantom_dim = pattern.shape

    phantom_w_in, *_ = jnp.linalg.lstsq(inputs, pattern)
    phantom_w_in = phantom_w_in.T
    phantom_w_out = jnp.zeros((out_dim, phantom_dim))

    new_w_in = jnp.concatenate([w_in, phantom_w_in], axis=0)
    new_w_out = jnp.concatenate([w_out, phantom_w_out], axis=1)

    return new_w_in, new_w_out
