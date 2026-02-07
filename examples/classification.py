from functools import partial

import flax.nnx as nnx
import flax.struct as struct
import jax
import jax.numpy as jnp
import optax
import typer
from flax.nnx.graph import GraphDef
from sklearn.datasets import make_classification

from rala.laplace import laplace_approximation
from rala.models import MLP
from rala.train import split, train

app = typer.Typer(pretty_exceptions_enable=False)


# @jax.jit
def loss_fn(y_pred: jax.Array, y_true: jax.Array):
    y_true_onehot = jax.nn.one_hot(y_true, y_pred.shape[1])
    loss = optax.softmax_cross_entropy(y_pred, y_true_onehot).mean()
    return loss


def reconstruct_model(last_layer: struct.PyTreeNode, model: nnx.Module):
    state = nnx.state(model)
    graphdef = nnx.graphdef(model)
    state["linear_out"] = last_layer
    model_from_layer = nnx.merge(graphdef, state)
    return model_from_layer


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


@partial(jax.vmap, in_axes=(0, None, None), out_axes=0)
def pm_from_last_layer(samples_layer: jax.Array, model: jax.Array, X: jax.Array):
    model = reconstruct_model(samples_layer, model)
    logits = model(X)
    logits = nnx.log_softmax(logits, axis=1)
    return logits


def acc_from_pm(pred_m: jax.Array, y: jax.Array):
    pred_m = jax.nn.logsumexp(pred_m, axis=0) - jnp.log(pred_m.shape[0])
    pred = pred_m.argmax(axis=1)
    return (pred == y).mean()


@jax.jit
def log_p(model: nnx.Module, X: jax.Array, y: jax.Array, sigma_p: float = 1):
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


@jax.jit
def log_p_from_last_layer(last_layer: nnx.Module, model: nnx.Module, X: jax.Array, y: jax.Array):
    last_layer_state = nnx.state(last_layer)
    model_from_layer = reconstruct_model(last_layer_state, model)
    return log_p(model_from_layer, X, y)


@app.command()
def main(
    size: int = 500,
    feat: int = 20,
    classes: int = 2,
    dmid: int = 64,
    epochs: int = 50,
    batch_size: int = 64,
    seed: int = 42,
):
    X, y = make_classification(n_samples=size, n_features=feat, n_classes=classes, random_state=seed)

    rngs = nnx.Rngs(seed)
    X_train, y_train, X_test, y_test = split(X, y, key=rngs())

    model = MLP(feat, dmid, classes, rngs=rngs)
    model, _, _ = train(model, epochs, X_train, y_train, loss_fn, batch_size, rngs())

    # Compute accuracy on the test set
    jax.debug.print("{}", acc(model, X_test, y_test))

    key1, key2 = jax.random.split(rngs(), 2)

    # Compute Laplace approximation for the last layer
    samples_layer, _, _ = laplace_approximation(
        partial(log_p_from_last_layer, model=model, X=X_train, y=y_train),
        model.linear_out,
        key1,
        num_samples=512,
    )

    # Compute the accuracy based on the Laplace approximation for the last layer
    jax.debug.print("{}", acc_from_pm(pm_from_last_layer(samples_layer, model, X_test), y_test))

    samples, graphdef, _ = laplace_approximation(
        partial(log_p, X=X_train, y=y_train),
        model,
        key2,
        num_samples=512,
    )

    # Compute accuracy from the predictive marginal
    jax.debug.print("{}", acc_from_pm(pm(samples, graphdef, X_test), y_test))

    return model


if __name__ == "__main__":
    app()
