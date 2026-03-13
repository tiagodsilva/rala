from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import typer
from flax.nnx.graph import GraphDef
from sklearn.datasets import make_classification

from rala.laplace import LaplaceMethod, laplace_approximation
from rala.models import MLP, MLPLastLayer
from rala.train import split, train

app = typer.Typer(pretty_exceptions_enable=False)


# @jax.jit
def loss_fn(y_pred: jax.Array, y_true: jax.Array, _: MLP):
    y_true_onehot = jax.nn.one_hot(y_true, y_pred.shape[1])
    loss = optax.softmax_cross_entropy(y_pred, y_true_onehot).mean()
    return loss


def acc(model: MLP, X: jax.Array, y: jax.Array):
    y_pred = model(X)
    y_pred = jnp.argmax(y_pred, axis=1)
    return (y_pred == y).mean()


@partial(jax.vmap, in_axes=(0, None, None), out_axes=0)
def pm(samples: jax.Array, graphdef: GraphDef, X: jax.Array):
    model = nnx.merge(graphdef, samples)
    logits = model(X)
    logits = nnx.log_softmax(logits, axis=1)
    return logits


def acc_from_pm(pred_m: jax.Array, y: jax.Array):
    pred_m = jax.nn.logsumexp(pred_m, axis=0) - jnp.log(pred_m.shape[0])
    pred = pred_m.argmax(axis=1)
    return (pred == y).mean()


@jax.jit
def log_p(model: nnx.Module, X: jax.Array, y: jax.Array, sigma_p: float = 5):
    # We compute the likelihood for data point
    logits = model(X)
    logits = nnx.log_softmax(logits, axis=1)
    rows = jnp.arange(len(logits))
    log_likelihood = logits[rows, y]

    # We only want to regularize 'Param' types (weights/biases), not 'BatchStat'
    params = nnx.state(model, nnx.Param)

    l2_norm = jax.tree_util.tree_reduce(
        lambda acc, p: acc + jnp.sum(p**2),
        params,
        initializer=0.0,
    )

    log_prior = -0.5 * (1 / sigma_p**2) * l2_norm

    return (log_likelihood + log_prior).sum()


@app.command()
def main(
    size: int = 1000,
    feat: int = 20,
    classes: int = 2,
    dmid: int = 64,
    epochs: int = 50,
    batch_size: int = 64,
    num_samples: int = 512,
    seed: int = 42,
    method: LaplaceMethod = LaplaceMethod.STANDARD,
    last_layer: bool = False,
):
    X, y = make_classification(
        n_samples=size, n_features=feat, n_classes=classes, random_state=seed
    )

    rngs = nnx.Rngs(seed)
    X_train, y_train, X_test, y_test = split(X, y, key=rngs())

    if last_layer:
        model = MLPLastLayer(feat, dmid, classes, rngs=rngs)
    else:
        model = MLP(feat, dmid, classes, rngs=rngs)

    model, _, key = train(
        model, epochs, X_train, y_train, loss_fn, batch_size, rngs()
    )

    # Compute accuracy on the test set
    jax.debug.print("{}", acc(model, X_test, y_test))

    samples, graphdef, _ = laplace_approximation(
        partial(log_p, X=X_train, y=y_train),
        model,
        key,
        num_samples=num_samples,
        method=method,
    )

    # Compute accuracy from the predictive marginal
    jax.debug.print("{}", acc_from_pm(pm(samples, graphdef, X_test), y_test))

    return model


if __name__ == "__main__":
    app()
