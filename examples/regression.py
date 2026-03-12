import pathlib
from enum import Enum
from functools import partial

import flax.nnx as nnx
import flax.struct as struct
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer

from rala.laplace import LaplaceMethod, laplace_approximation
from rala.models import MLP, ExtraParamWrapper
from rala.train import train
from rala.utils import show_kitty

app = typer.Typer(pretty_exceptions_enable=False)

HERE = pathlib.Path(__file__).parent


class DataEnum(Enum):
    SNELSON = "snelson"


@struct.dataclass
class ExtraParams:
    log_sigma: jax.Array


def make_snelson_data():
    import numpy as np  # only used here

    data = np.loadtxt(HERE / "data/snelson_inout.dat")
    data = jnp.asarray(data)

    X = data[:, [0]]  # (N, 1)
    y = data[:, 1]  # (N,)

    return X, y


# @jax.jit
def log_lik_fn(y_pred: jax.Array, y_true: jax.Array, model: MLP[ExtraParams]):
    log_sigma = model.extra_params.log_sigma
    y_pred = y_pred.squeeze(axis=1)

    loglik = (
        -0.5 * ((y_pred - y_true) ** 2) / jnp.exp(2 * log_sigma) - log_sigma
    )

    return loglik.sum()


def log_prior_fn(model: MLP[ExtraParams], sigma_p: float = 1e2):
    # We only want to regularize 'Param' types (weights/biases), not 'BatchStat'
    params = nnx.state(model, nnx.Param, nnx.Not(ExtraParamWrapper))

    # Compute the prior distribution
    l2_norm = jax.tree_util.tree_reduce(
        lambda acc, p: acc + jnp.sum(p**2),
        params,
        initializer=0.0,
    )
    log_prior = -0.5 * (1 / sigma_p**2) * l2_norm

    return log_prior


@nnx.jit
def log_p(model: nnx.Module, X: jax.Array, y: jax.Array):
    # We compute the likelihood for data point
    y_pred = model(X)
    log_lik = log_lik_fn(y_pred, y, model)
    log_prior = log_prior_fn(model)

    return (log_lik + log_prior).sum()


def plot_model_1d(model: MLP[ExtraParams], x_obs: jax.Array):
    # Create a grid to plot
    x_grid = jnp.linspace(x_obs.min() - 1, x_obs.max() + 1, num=100)
    x_grid = jnp.expand_dims(x_grid, axis=1)
    y_pred = model(x_grid)

    plt.plot(x_grid, y_pred, label="model")


@app.command()
def main(
    dmid: int = 32,
    epochs: int = 50,
    batch_size: int = 64,
    dataset: DataEnum = DataEnum.SNELSON,
    num_samples: int = 1000,
    seed: int = 42,
    method: LaplaceMethod = LaplaceMethod.STANDARD,
):

    match dataset:
        case DataEnum.SNELSON:
            X, y = make_snelson_data()

    extra_params = ExtraParams(log_sigma=-1.0)

    rngs = nnx.Rngs(seed)
    model = MLP(X.shape[1], dmid, 1, extra_params=extra_params, rngs=rngs)

    # Find the MAP (minimize the negative log posterior)
    def loss_fn(X: jax.Array, y: jax.Array, model: MLP[ExtraParams]):
        return -log_p(model, X, y)

    model, _, _ = train(model, epochs, X, y, loss_fn, batch_size, rngs())

    jax.debug.print("{}", model.extra_params.log_sigma)
    key1, key2 = jax.random.split(rngs(), 2)

    # if isinstance(model, MLP):
    #     # Compute Laplace approximation for the last layer
    #     samples_layer, _, _ = laplace_approximation(
    #         partial(log_p_from_last_layer, model=model, X=X_train, y=y_train),
    #         model.linear_out,
    #         key1,
    #         num_samples=num_samples,
    #         method=method,
    #     )

    #     # Compute the accuracy based on the Laplace approximation for the last layer
    #     jax.debug.print(
    #         "{}",
    #         acc_from_pm(
    #             pm_from_last_layer(samples_layer, model, X_test), y_test
    #         ),
    #     )

    samples, graphdef, _ = laplace_approximation(
        partial(log_p, X=X, y=y),
        model,
        key2,
        num_samples=num_samples,
        method=method,
    )

    match dataset:
        case DataEnum.SNELSON:
            x_obs = X.squeeze(axis=1)
            for i in range(num_samples):
                sample = jax.tree_util.tree_map(lambda x: x[i], samples)
                model_from_sample = nnx.merge(graphdef, sample)
                plot_model_1d(model, x_obs)
            plt.scatter(x_obs, y)
            show_kitty()

    # # Compute accuracy from the predictive marginal
    # jax.debug.print("{}", acc_from_pm(pm(samples, graphdef, X_test), y_test))

    # return model


if __name__ == "__main__":
    app()
