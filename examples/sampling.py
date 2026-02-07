from enum import Enum

import jax
import jax.numpy as jnp
import jax.scipy.optimize as jspo
import matplotlib.pyplot as plt
import typer

from rala.laplace import laplace_approximation
from rala.models import LogPosterior
from rala.utils import plot_potential, show_kitty


def banana_log_density(x, b: float = 4):
    x_sq = x * x
    log_p = x_sq[0] + (x[1] - b * (x_sq[0] - 1)) ** 2
    return -log_p


def squiggle_log_density(x, a: float = 1.5, cov: jax.Array = jnp.array([[5, 0], [0, 0.05]])):
    y = jnp.array([x[0], x[1] + jnp.sin(a * x[0])])
    y_inv = jnp.linalg.solve(cov, y)
    return -jnp.dot(y, y_inv) / 2


def plot_samples(samples: jax.Array, x_map: jax.Array, logp_fn: callable):
    # Plot the samples
    xmin, xmax = samples[:, 0].min(), samples[:, 0].max()
    ymin, ymax = samples[:, 1].min(), samples[:, 1].max()

    ax = plt.subplot(1, 2, 1)
    plot_potential(xmin, xmax, ymin, ymax, res=2000, logp_fn=logp_fn, ax=ax, alpha=0.4)
    ax.scatter(samples[:, 0], samples[:, 1])
    ax = plt.subplot(1, 2, 2)
    plot_potential(xmin, xmax, ymin, ymax, res=2000, logp_fn=logp_fn, ax=ax)
    ax.scatter(x_map[0], x_map[1], c="red")

    show_kitty()


class DistType(Enum):
    BANANA = "banana"
    SQUIGGLE = "squiggle"

    def get(self):
        if self == DistType.BANANA:
            return banana_log_density, 2
        elif self == DistType.SQUIGGLE:
            return squiggle_log_density, 2


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(dist: DistType, epochs: int = 50, seed: int = 42, num_samples: int = 1000):
    logp_fn, dim = dist.get()

    # Find MAP by maximizing the log-likelihood
    x_map = jax.jit(jspo.minimize, static_argnames=("fun", "method"))(
        fun=lambda x: -logp_fn(x),
        x0=jnp.zeros((dim,)),
        method="BFGS",
    ).x

    # Instantiate the distribution
    model = LogPosterior(x_map, logp_fn)

    # Compute the Laplace Approximation
    key = jax.random.key(seed)
    samples, _, key = laplace_approximation(
        # Here there is no separation between model and data.
        # Hence the posterior distribution is simply computed by calling `model`.
        lambda model: model(),
        model,
        key=key,
        num_samples=num_samples,
    )
    samples = samples["theta"].get_value()

    plot_samples(samples, x_map, logp_fn)


if __name__ == "__main__":
    app()
