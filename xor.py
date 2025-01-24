import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

from jaxtyping import Array


def mse(pred_y, y):
    return jnp.mean(jnp.sum(jnp.square((pred_y - y)), axis=1), axis=0)


def batch_objective(params, args):
    static, X, y = args
    model = eqx.combine(params, static)
    pred_y = eqx.filter_vmap(model)(X)
    objective_value = mse(pred_y, y)
    aux = None
    return objective_value, aux


def train_xor() -> tuple[eqx.nn.MLP, Array, Array]:
    N = 2
    K = 2

    X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = jnp.array([[0, 1, 1, 0]]).T

    model = eqx.nn.MLP(
        in_size=N,
        width_size=K,
        out_size=1,
        depth=1,
        use_bias=False,
        use_final_bias=False,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(37),
    )

    optimizer = optx.GradientDescent(learning_rate=1e-1, rtol=1e-4, atol=1e-4)
    options = None
    f_struct = jax.ShapeDtypeStruct((), jnp.float32)
    aux_struct = None
    tags = frozenset()

    init = eqx.Partial(
        optimizer.init,
        fn=batch_objective,
        options=options,
        f_struct=f_struct,
        aux_struct=aux_struct,
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

    while not done:
        params, static = eqx.partition(model, eqx.is_array)
        params, state, _ = step(y=params, args=(static, X, y), state=state)
        done, result = terminate(y=params, args=(static, X, y), state=state)
        model = eqx.combine(params, static)
        loss, _ = batch_objective(params, (static, X, y))
        print(f"Evaluating iteration with loss value {loss}.")

    if result != optx.RESULTS.successful:
        print("Failed!")

    params, static = eqx.partition(model, eqx.is_array)
    params, _, _ = postprocess(
        y=params,
        aux=None,
        args=(static, X, y),
        state=state,
        result=result,
    )

    loss, _ = batch_objective(params, (static, X, y))
    print(f"Found solution with loss value {loss}.")
    model = eqx.combine(params, static)

    return model


if __name__ == "__main__":
    model = train_xor()
