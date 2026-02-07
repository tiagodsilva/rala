from enum import Enum

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import typer

from rala.utils import plot_potential, show_kitty


def banana_log_density(x, b: float = 4):
    x_sq = x * x
    log_p = x_sq[0] + (x[1] - b * (x_sq[0] - 1)) ** 2
    return log_p


def squiggle_log_density(x, a: float = 1.5, cov: jax.Array = jnp.array([[5, 0], [0, 0.05]])):
    y = jnp.array([x[0], x[1] + jnp.sin(a * x[0])])
    return jsp.stats.multivariate_normal.logpdf(
        y,
        mean=jnp.zeros_like(y),
        cov=cov,
    )


class DistType(Enum):
    BANANA = "banana"
    SQUIGGLE = "squiggle"

    def get(self):
        if self == DistType.BANANA:
            return banana_log_density
        elif self == DistType.SQUIGGLE:
            return squiggle_log_density


app = typer.Typer()


@app.command()
def main(dist: DistType, epochs: int = 50, seed: int = 42):
    pot = dist.get()

    plot_potential(-10, 10, -2, 2, res=2000, potential=pot, ax=plt.gca())
    show_kitty()
    # Instantiate the distribution

    # Search for the MAP

    # Compute the Laplace Approximation

    # Plot the samples

    return


if __name__ == "__main__":
    app()
