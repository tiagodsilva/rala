from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import tqdm

from rala.models import LogPosterior


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
    lr: 1e-3,
    should_clip: bool = False,
    base: optax.GradientTransformation = optax.contrib.muon,
    weight_decay: float = 1e-4,
):
    clip_default = optax.clip_by_global_norm(1.0) if should_clip else optax.identity()
    tx = optax.chain(clip_default, base(learning_rate=lr, weight_decay=weight_decay))
    return nnx.Optimizer(model, tx=tx, wrt=nnx.Param)


def make_indices(size: int, batch_size: int, key: jax.Array):
    all_samples = jnp.arange(size)
    # We arbitrarily prune `all_samples` to ensure its size is a multiple of batch_size
    key, subkey = jax.random.split(key, 2)
    all_samples_shuffled = jax.random.permutation(subkey, all_samples)
    n_batches = size // batch_size
    all_samples_shuffled = all_samples_shuffled[: n_batches * batch_size]
    return all_samples_shuffled.reshape((n_batches, batch_size)), key


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
        return loss_fn(y_pred, y_batch)

    loss, grads = nnx.value_and_grad(step, argnums=0)(model)
    opt.update(model, grads)

    return (model, opt, curr_loss + loss), None


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
    (model, opt, loss), _ = jax.lax.scan(
        f=partial(update, X=X, y=y, loss_fn=loss_fn),
        xs=indices,
        init=(model, opt, jnp.array(0.0)),
        length=len(indices),
    )

    return model, loss / len(indices), key


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

    # Get the data indices
    indices, key = make_indices(X.shape[0], batch_size, key)

    losses = []
    for _ in (pbar := tqdm.trange(epochs)):
        model, loss, key = train_step(model, X, y, indices, loss_fn, key, opt)
        pbar.set_postfix(loss=loss)
        losses.append(loss)

    return model, losses, key
