from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import tqdm


def create_opt(
    model: nnx.Module,
    lr: 1e-3,
    should_clip: bool = False,
):

    clip_default = optax.clip_by_global_norm(1.0) if should_clip else optax.identity()

    tx = optax.chain(clip_default, optax.contrib.muon(learning_rate=lr))

    return nnx.Optimizer(model, tx=tx, wrt=nnx.Param)


def make_indices(size: int, batch_size: int, key: jax.Array):
    all_samples = jnp.arange(size)
    # We arbitrarily expand `all_samples` to ensure its size is a multiple of batch_size
    key, subkey = jax.random.split(key, 2)
    all_samples_shuffled = jax.random.permutation(subkey, all_samples)
    n_batches = size // batch_size
    all_samples_shuffled = all_samples_shuffled[n_batches * batch_size]
    return n_batches, all_samples_shuffled.view(n_batches, batch_size), key


def update(
    carry: tuple[nnx.Module, jax.Array],
    indices: jax.Array,
    X: jax.Array,
    y: jax.Array,
    loss_fn: callable,
    opt: nnx.Optimizer,
):
    model, curr_loss = carry

    def loss_fn(model: nnx.Module):
        y_pred = model(X[indices])
        loss = loss_fn(y_pred, y[indices])
        return loss

    loss, grads = nnx.value_and_grad(loss_fn, argnums=0)
    opt.update(model, grads)

    return (model, curr_loss + loss), None


@jax.jit
def train_step(
    model: nnx.Module,
    X: jax.Array,
    y: jax.Array,
    loss_fn: callable,
    batch_size: int,
    key: jax.Array,
    opt: nnx.Optimizer,
):
    size = X.shape[0]
    # Get the data indices
    n_batches, indices, key = make_indices(size, batch_size, key)

    # Evaluate the model
    (model, loss), _ = jax.lax.scan(
        f=partial(update, X=X, y=y, loss_fn=loss_fn, opt=opt),
        xs=indices,
        init=(model, jnp.array(0.0)),
        length=n_batches,
    )

    return model, loss / size, key


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
        model, loss, key = train_step(model, X, y, loss_fn, batch_size, key, opt)
        pbar.set_postfix(loss=loss)
        losses.append(loss)

    return model, losses
