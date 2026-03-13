from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import tqdm

from rala.models import ExtraParamsWrapper


def split(X: jax.Array, *ys: jax.Array, p: float = 0.7, key: jax.Array):
    size = len(X)
    indices = jnp.arange(size)
    indices_shuffled = jax.random.permutation(key, indices)

    size_train = int(p * size)
    indices_left = indices_shuffled[:size_train]
    indices_right = indices_shuffled[size_train:]

    split_train = [X[indices_left]] + [y[indices_left] for y in ys]
    split_test = [X[indices_right]] + [y[indices_right] for y in ys]
    return split_train + split_test


def create_opt(
    model: nnx.Module,
    lr: float = 1e-3,
    should_clip: bool = False,
    base: optax.GradientTransformation = optax.sgd,
):

    params = nnx.state(model, nnx.Any(nnx.Param, ExtraParamsWrapper))

    # JAX's way of labeling parameters, which allows different optimizers
    # to be used for different parameter groups. Quite unreadable.
    def label_params(path, _):
        if "extra_params" in path:
            return "extra_params"
        return "default"

    param_labels = nnx.map_state(label_params, params)

    clip_default = (
        optax.clip_by_global_norm(10.0) if should_clip else optax.identity()
    )
    tx = optax.multi_transform(
        {
            "default": optax.chain(clip_default, base(learning_rate=lr)),
            "extra_params": optax.chain(
                clip_default, optax.sgd(learning_rate=lr, nesterov=True)
            ),
        },
        param_labels,
    )

    return nnx.Optimizer(
        model, tx=tx, wrt=nnx.Any(nnx.Param, ExtraParamsWrapper)
    )


def make_indices(size: int, batch_size: int, key: jax.Array):
    all_samples = jnp.arange(size)
    key, subkey = jax.random.split(key, 2)
    all_samples_shuffled = jax.random.permutation(subkey, all_samples)
    n_batches = size // batch_size
    size_in_batches = n_batches * batch_size
    size_in_excess = size - size_in_batches

    samples_per_index = all_samples_shuffled[:size_in_batches]
    if size_in_excess > 0:
        samples_not_included = all_samples_shuffled[size_in_batches:]
        extra_samples = all_samples_shuffled[: batch_size - size_in_excess]
        extra_samples = jnp.hstack([samples_not_included, extra_samples])
        samples_per_index = jnp.hstack([samples_per_index, extra_samples])

    return samples_per_index.reshape(
        (n_batches + (size_in_excess > 0), batch_size)
    ), key


def update(
    carry: tuple[nnx.Module, nnx.Optimizer, jax.Array],
    indices: jax.Array,
    X: jax.Array,
    y: jax.Array,
    loss_fn: callable,
):
    model, opt, curr_loss = carry

    def step(model):
        X_batch = jnp.take(X, indices, axis=0)
        y_batch = jnp.take(y, indices, axis=0)
        y_pred = model(X_batch)
        return loss_fn(y_pred, y_batch, model)

    loss, grads = nnx.value_and_grad(
        step, argnums=nnx.DiffState(0, nnx.Any(nnx.Param, ExtraParamsWrapper))
    )(model)
    opt.update(model, grads)

    # Compute the L2 norm of the gradients
    grads_norm = jax.tree_util.tree_reduce(
        lambda acc, p: acc + jnp.sum(p**2), grads, initializer=0.0
    )
    return (model, opt, curr_loss + loss), grads_norm


@partial(nnx.jit, static_argnames=("loss_fn",))
def train_step(
    model: nnx.Module,
    X: jax.Array,
    y: jax.Array,
    indices: jax.Array,
    loss_fn: callable,
    key: jax.Array,
    opt: nnx.Optimizer,
):
    # Evaluate the model
    (model, opt, loss), grads_norm = jax.lax.scan(
        f=partial(update, X=X, y=y, loss_fn=loss_fn),
        xs=indices,
        init=(model, opt, jnp.array(0.0)),
        length=len(indices),
    )

    return model, grads_norm.mean(), loss / len(indices), key


def train(
    model: nnx.Module,
    epochs: int,
    X: jax.Array,
    y: jax.Array,
    loss_fn: callable,
    batch_size: int,
    key: jax.Array,
    lr: float = 1e-3,
    should_clip: bool = True,
):
    opt = create_opt(model, lr=lr, should_clip=should_clip)

    losses = []
    for _ in (pbar := tqdm.trange(epochs)):
        indices, key = make_indices(X.shape[0], batch_size, key)
        model, grads_norm, loss, key = train_step(
            model, X, y, indices, loss_fn, key, opt
        )
        pbar.set_postfix(loss=f"{loss:.2e}", grads_norm=f"{grads_norm:.2e}")
        losses.append(loss)

    return model, losses, key
