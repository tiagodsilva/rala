import os
import subprocess
from io import BytesIO

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def is_kitty():
    return os.environ.get("TERM", "").startswith("xterm-kitty")


def show_kitty():
    # Utility for displaying plots in kitty terminal
    # (instead of creating a window)
    if not is_kitty():
        return

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Pipe directly to kitty
    subprocess.run(["kitty", "+kitten", "icat"], input=buf.getvalue())
    plt.close()


def plot_potential(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    res: float,
    potential: callable,
    ax: plt.Axes,
    alpha: float = 1.0,
):
    pot_vmap = jax.vmap(jax.vmap(potential, 1, 0), 1, 0)

    x = jnp.linspace(xmin, xmax, endpoint=True, num=res)
    y = jnp.linspace(ymin, ymax, endpoint=True, num=res)

    xy = jnp.meshgrid(x, y)
    xy = jnp.stack(xy)

    log_p = -pot_vmap(xy)

    # As we simply want the shape, we normalize log_p
    log_p = log_p - jax.nn.logsumexp(log_p, axis=(0, 1))
    p = jnp.exp(log_p)

    ax.contourf(xy[0], xy[1], p, alpha=alpha)

    return p
