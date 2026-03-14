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
from rala.models import MLP, Buffer, ExtraParamsWrapper, MLPLastLayer
from rala.train import train
from rala.utils import show_kitty

# python examples/regression.py --epochs 15000 --last-layer --seed 84 --batch-size 200
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


def log_prior_fn(
    model: MLP[ExtraParams], sigma_p: float = 10, tau_p: float = 10
):
    # We only want to regularize 'Param' types (weights/biases), not 'BatchStat'
    params = nnx.state(model, nnx.Param)

    # Compute the prior distribution
    def acc_fn(acc: jax.Array, p: jax.Array):
        return acc + jnp.sum(p**2)

    l2_norm = jax.tree_util.tree_reduce(
        acc_fn,
        params,
        initializer=0.0,
    )
    log_prior = (
        -0.5 * (1 / sigma_p**2) * l2_norm
        # log-normal prior over the observational variance
        - 0.5 * (1 / tau_p**2) * model.extra_params.log_sigma**2
    )

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
    x_grid = jnp.linspace(x_obs.min(), x_obs.max(), num=200)
    x_grid = jnp.expand_dims(x_grid, axis=1)
    y_pred = model(x_grid)

    plt.plot(x_grid, y_pred, label="model")


@app.command()
def main(
    dmid: int = 10,
    epochs: int = 50,
    batch_size: int = 128,
    dataset: DataEnum = DataEnum.SNELSON,
    num_samples: int = 1000,
    seed: int = 42,
    method: LaplaceMethod = LaplaceMethod.STANDARD,
    last_layer: bool = False,
    rwmc: bool = False,
):

    match dataset:
        case DataEnum.SNELSON:
            X, y = make_snelson_data()

    extra_params = ExtraParams(log_sigma=jnp.log(y.std()))

    rngs = nnx.Rngs(seed)
    if last_layer:
        model = MLPLastLayer(
            X.shape[1], dmid, 1, extra_params=extra_params, rngs=rngs
        )
    else:
        model = MLP(X.shape[1], dmid, 1, extra_params=extra_params, rngs=rngs)

    model.set_scale(X, y)

    # Find the MAP (minimize the negative log posterior)
    total_samples = X.shape[0]

    def loss_fn(y_pred: jax.Array, y: jax.Array, model: MLP[ExtraParams]):
        log_lik = log_lik_fn(y_pred, y, model)
        log_prior = log_prior_fn(model)
        log_lik = log_lik * (total_samples / y_pred.shape[0])

        return -(log_lik + log_prior)

    model, _, _ = train(
        model, epochs, X, y, loss_fn, batch_size, rngs(), should_clip=True
    )

    jax.debug.print("{}", model.extra_params.log_sigma)
    _, subkey = jax.random.split(rngs(), 2)

    extra_state = nnx.state(model, nnx.Any(ExtraParamsWrapper, Buffer))

    samples, graphdef, _ = laplace_approximation(
        partial(log_p, X=X, y=y),
        model,
        subkey,
        num_samples=num_samples,
        method=method,
        extra_state=extra_state,
        rwmc_refine=rwmc,
        min_eigenvalue=1e-6,
    )

    match dataset:
        case DataEnum.SNELSON:
            x_obs = X.squeeze(axis=1)
            for i in range(num_samples):
                sample = jax.tree_util.tree_map(lambda x: x[i], samples)
                model_from_sample = nnx.merge(graphdef, sample, extra_state)
                plot_model_1d(model_from_sample, x_obs)
            plt.scatter(x_obs, y)
            show_kitty()

    # # Compute accuracy from the predictive marginal
    # jax.debug.print("{}", acc_from_pm(pm(samples, graphdef, X_test), y_test))

    # return model


if __name__ == "__main__":
    app()
