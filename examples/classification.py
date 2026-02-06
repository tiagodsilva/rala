import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import typer
from sklearn.datasets import make_classification

from rala.models import MLP
from rala.train import train

app = typer.Typer(pretty_exceptions_enable=False)


# @jax.jit
def loss_fn(y_pred: jax.Array, y_true: jax.Array):
    y_true_onehot = jax.nn.one_hot(y_true, y_pred.shape[1])
    loss = optax.softmax_cross_entropy(y_pred, y_true_onehot).mean()
    return loss


def acc(model: MLP, X: jax.Array, y: jax.Array):
    y_pred = model(X)
    y_pred = jnp.argmax(y_pred, axis=1)
    return (y_pred == y).mean()


@app.command()
def main(
    size: int = 500,
    feat: int = 20,
    classes: int = 2,
    dmid: int = 64,
    epochs: int = 200,
    batch_size: int = 64,
    seed: int = 42,
):
    X, y = make_classification(n_samples=size, n_features=feat, n_classes=classes, random_state=seed)

    rngs = nnx.Rngs(seed)
    model = MLP(feat, dmid, classes, rngs=rngs)
    model, _, _ = train(model, epochs, X, y, loss_fn, batch_size, rngs())

    # Compute accuracy on the training set
    jax.debug.print("{}", acc(model, X, y))

    return model


if __name__ == "__main__":
    app()
