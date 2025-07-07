# type: ignore[reportMissingParameterType]
"""Mash-up of `eqx.filter_vmap` and `jax.lax.map`."""

from collections.abc import Callable
from collections.abc import Hashable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._filters import combine
from equinox._filters import is_array
from equinox._filters import partition
from equinox._module import Static
from equinox._module import module_update_wrapper
from equinox._vmap_pmap import AxisSpec
from equinox._vmap_pmap import _is_none  # type: ignore[reportPrivateUsage]
from equinox._vmap_pmap import _moveaxis  # type: ignore[reportPrivateUsage]
from equinox._vmap_pmap import _named_in_axes  # type: ignore[reportPrivateUsage]
from equinox._vmap_pmap import _resolve_axes  # type: ignore[reportPrivateUsage]
from equinox._vmap_pmap import _VmapWrapper  # type: ignore[reportPrivateUsage]
from jaxtyping import Array
from jaxtyping import PyTree


def _array_or_scalar_axis_size(x: Any, axis: int):
    if isinstance(x, Array):
        if x.shape:
            return x.shape[axis]
        else:
            return 0
    return None


def _flatten_batch_dim(x: Array) -> Array:
    return x.reshape(-1, *x.shape[2:])


def _unbatch(x: Any, y: Any) -> Any:
    if isinstance(x, Array):
        return jnp.concatenate([_flatten_batch_dim(x), y], axis=0)
    return x


def _batch_and_remainder(x: PyTree, axis_spec: PyTree[int] | int, batch_size: int):
    leaves, treedef = jax.tree_util.tree_flatten(x)
    axis_leaves, _ = jax.tree_util.tree_flatten(axis_spec)

    scan_leaves = []
    remainder_leaves = []

    for leaf, axis in zip(leaves, axis_leaves):
        if not isinstance(leaf, Array) or axis is None:
            scan_leaves.append(leaf)
            remainder_leaves.append(leaf)
            continue

        if not leaf.shape:  # scalar; broadcast like `jax.vmap`
            broadcast_shape = [
                batch_size if i == axis else 1 for i in range(max(axis + 1, 2))
            ]
            scan_leaves.append(
                jax.numpy.broadcast_to(leaf, (1,) + tuple(broadcast_shape))
            )
            remainder_leaves.append(jax.numpy.array([]))
            continue

        num_batches, _ = divmod(leaf.shape[axis], batch_size)
        total_batch_elems = num_batches * batch_size

        # Principal.
        idx = tuple(
            slice(None) if i != axis else slice(total_batch_elems)
            for i in range(len(leaf.shape))
        )
        main = leaf[idx]
        reshape_dims = list(main.shape)
        reshape_dims[axis : axis + 1] = [num_batches, batch_size]
        batched = main.reshape(reshape_dims)
        scan_leaves.append(jax.numpy.moveaxis(batched, axis, 0))

        # Remainder.
        rem_idx = tuple(
            slice(None) if i != axis else slice(total_batch_elems, None)
            for i in range(len(leaf.shape))
        )
        remainder_leaves.append(leaf[rem_idx])

    scan_tree = treedef.unflatten(scan_leaves)
    remainder_tree = treedef.unflatten(remainder_leaves)
    return scan_tree, remainder_tree


class _BmapWrapper(_VmapWrapper):
    _batch_size: int | None

    def __call__(self, /, *args, **kwargs):
        if len(kwargs) != 0:
            raise RuntimeError(
                "keyword arguments cannot be used with functions wrapped with " "`bmap`"
            )
        del kwargs

        in_axes = _named_in_axes(self._fun, self._in_axes, args)
        in_axes = _resolve_axes(args, in_axes)
        unmapped_axis = jax.tree.map(_is_none, in_axes, is_leaf=_is_none)
        static_args, dynamic_args = partition(args, unmapped_axis)

        # If `axis_size` is not provided, determine the size of the vmap axis.
        if self._axis_size is None:
            max_size = max(
                jax.tree.leaves(
                    jax.tree.map(_array_or_scalar_axis_size, dynamic_args, in_axes)
                )
            )
        else:
            max_size = self._axis_size

        def _fun_wrapper(_dynamic_args):
            _args = combine(_dynamic_args, static_args)
            _out = self._fun(*_args)
            _out_axes = _resolve_axes(_out, self._out_axes)
            _none_axes = jax.tree.map(_is_none, _out_axes, is_leaf=_is_none)
            _nonvmapd, _vmapd = partition(_out, _none_axes, is_leaf=_is_none)
            _nonvmapd_arr, _nonvmapd_static = partition(_nonvmapd, is_array)
            return _vmapd, _nonvmapd_arr, Static((_nonvmapd_static, _out_axes))

        if len(jax.tree.leaves(in_axes)) == 0 and self._axis_size is None:
            vmapd, nonvmapd_arr, static = _fun_wrapper(dynamic_args)
            if len(jax.tree.leaves(vmapd)) != 0:
                raise ValueError(
                    "Cannot resolve batch dimension. Non-`None` `out_axes` requires "
                    "either `in_axes` or `axis_size` to be not `None`."
                )

        elif (
            len(jax.tree.leaves(in_axes)) == 0
            or self._batch_size is None
            or self._batch_size >= max_size
        ):
            # Fall back to the `eqx.filter_vmap` solution.
            vmapd, nonvmapd_arr, static = jax.vmap(
                _fun_wrapper,
                in_axes=(in_axes,),
                out_axes=(0, None, None),
                axis_name=self._axis_name,
                axis_size=self._axis_size,
                **self._vmapkwargs,
            )(dynamic_args)

        else:
            if len(dynamic_args) > 0:
                scan_dynamic, remainder_dynamic = zip(
                    *[
                        _batch_and_remainder(arg, in_axis, self._batch_size)
                        for arg, in_axis in zip(dynamic_args, in_axes)
                    ]
                )

                def scan_f(_, batched_dynamic):
                    return (), jax.vmap(
                        _fun_wrapper,
                        in_axes=(in_axes,),
                        out_axes=(0, None, None),
                        axis_name=self._axis_name,
                        **self._vmapkwargs,
                    )(batched_dynamic)

                _, (scan_vmapd, _, _) = jax.lax.scan(scan_f, (), scan_dynamic)

                remainder_vmapd, remainder_nonvmapd_arr, remainder_static = jax.vmap(
                    _fun_wrapper,
                    in_axes=(in_axes,),
                    out_axes=(0, None, None),
                    axis_name=self._axis_name,
                    **self._vmapkwargs,
                )(remainder_dynamic)

                vmapd = jax.tree.map(
                    _unbatch,
                    scan_vmapd,
                    remainder_vmapd,
                )
                nonvmapd_arr = remainder_nonvmapd_arr
                static = remainder_static

            else:
                vmapd, nonvmapd_arr, static = jax.vmap(
                    _fun_wrapper,
                    in_axes=(in_axes,),
                    out_axes=(0, None, None),
                    axis_name=self._axis_name,
                    axis_size=self._axis_size,
                    **self._vmapkwargs,
                )(dynamic_args)

        nonvmapd_static, out_axes = static.value
        nonvmapd = combine(nonvmapd_arr, nonvmapd_static)

        assert jax.tree.structure(vmapd) == jax.tree.structure(out_axes)
        vmapd = jax.tree.map(_moveaxis, vmapd, out_axes)

        return combine(vmapd, nonvmapd)


def bmap(
    fun: Callable[..., Any],
    *,
    in_axes: PyTree[AxisSpec] = eqx.if_array(0),
    out_axes: PyTree[AxisSpec] = eqx.if_array(0),
    axis_name: Hashable = None,
    axis_size: int | None = None,
    batch_size: int | None = None,
    **vmapkwargs,
) -> Callable[..., Any]:
    """Batched map to save memory.

    `bmap` splits the `in_axes` dimension of the input into batches of size
    `batch_size` and applies the function `vmap(f)` to each batch inside a
    `scan`.

    Example:
      >>> def f(x):
      ...   assert x.shape == ()  # `x` is a scalar
      ...   return x ** 2
      >>> x = np.arange(8)
      >>> y = bmap(f, batch_size=2)(x)
      >>> y
      Array([ 0,  1,  4,  9, 16, 25, 36, 49], dtype=int32)

    Wrapping `fun` with `jax.remat()` is recommended if you want to use `bmap`
    inside gradient calculations. Otherwise, you won't realize any memory savings
    because JAX will save all the forward activations.

    If your code looks like `value_and_grad(mean(vmap(...)))` then you probably want
    to use microbatching instead, which is more efficient because you don't need
    to use `jax.remat`.
    """
    if axis_name is not None and batch_size is not None:
        raise ValueError("Batching with `axis_name` is not supported.")

    bmap_wrapper = _BmapWrapper(
        _fun=fun,
        _in_axes=in_axes,
        _out_axes=out_axes,
        _axis_name=axis_name,
        _axis_size=axis_size,
        _batch_size=batch_size,
        _vmapkwargs=vmapkwargs,
    )
    return module_update_wrapper(bmap_wrapper)
