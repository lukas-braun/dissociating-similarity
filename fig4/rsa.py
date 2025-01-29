"""Representational similarity analysis comparison measures."""

# ruff: noqa: INP001
from collections.abc import Callable
from functools import partial
from math import prod

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import PRNGKeyArray

from distance import correlation_distance
from distance import cosine_similarity
from distance import difference_norm
from distance import pearson_correlation


def calculate_rdm(
    activations: Array,
    dissimilarity_measure: Callable[[Array, Array], Array],
    return_full: bool = False,
) -> Array:
    """Compute representational dissimilarities for `activations`."""
    compute_pairwise_distances = jax.vmap(
        jax.vmap(dissimilarity_measure, in_axes=(0, None)),
        in_axes=(None, 0),
    )

    rdm_full = compute_pairwise_distances(activations, activations)
    if return_full:
        return rdm_full

    rdm_triu = rdm_full[jnp.triu_indices_from(rdm_full, k=1)]
    return rdm_triu


def _scaled_squared_euclidean_distance(x: Array, y: Array) -> Array:
    """Dimension-scaled squared Euclidean distance between `x` and `y`."""
    squared_distance = jnp.square(difference_norm(x, y, p=2))
    # Scale the distance by the number of feature dimensions.
    return squared_distance / prod(x.shape)


euclidean_rdm = partial(
    calculate_rdm,
    dissimilarity_measure=_scaled_squared_euclidean_distance,
)
correlation_rdm = partial(
    calculate_rdm,
    dissimilarity_measure=correlation_distance,
)


def calculate_rsa(
    x: Array,
    y: Array,
    representational_dissimilarity_measure: Callable[[Array, Array], Array],
    matrix_dissimilarity_measure: Callable[[Array, Array], Array],
) -> Array:
    """Perform representational similarity analysis."""
    _calculate_rdm = partial(
        calculate_rdm, dissimilarity_measure=representational_dissimilarity_measure
    )
    rdm_x = _calculate_rdm(x)
    rdm_y = _calculate_rdm(y)

    return matrix_dissimilarity_measure(rdm_x, rdm_y)


def cosine_rsa_correlation_rdm(
    x: Array,
    y: Array,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Compare representations by cosine similarity of correlations."""
    del key
    return calculate_rsa(
        x=x,
        y=y,
        representational_dissimilarity_measure=correlation_distance,
        matrix_dissimilarity_measure=cosine_similarity,
    )


def correlation_rsa_correlation_rdm(
    x: Array,
    y: Array,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Compare representations by correlation of correlations."""
    del key
    return calculate_rsa(
        x=x,
        y=y,
        representational_dissimilarity_measure=correlation_distance,
        matrix_dissimilarity_measure=pearson_correlation,
    )


def cosine_rsa_euclidean_rdm(
    x: Array,
    y: Array,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Compare representations by cosine similarity of Euclidean distances."""
    del key
    return calculate_rsa(
        x=x,
        y=y,
        representational_dissimilarity_measure=_scaled_squared_euclidean_distance,
        matrix_dissimilarity_measure=cosine_similarity,
    )


def correlation_rsa_euclidean_rdm(
    x: Array,
    y: Array,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Compare representations by correlation of Euclidean representational distances."""
    del key
    return calculate_rsa(
        x=x,
        y=y,
        representational_dissimilarity_measure=_scaled_squared_euclidean_distance,
        matrix_dissimilarity_measure=pearson_correlation,
    )
