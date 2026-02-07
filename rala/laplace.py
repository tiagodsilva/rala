from enum import Enum
from functools import partial
from typing import Callable

import flax.nnx as nnx
import flax.struct as struct
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as tree_util
from diffrax import Dopri5, ODETerm, diffeqsolve
from jax.flatten_util import ravel_pytree


class LaplaceMethod:
    STANDARD = "standard"
    RIEMANN = "riemann"


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None), out_axes=0)
def tree_multivariate_normal(
    key: jax.Array,
    mean_tree: struct.PyTreeNode,
    cov_L_tree: struct.PyTreeNode,
):
    def sample_leaf(key: jax.Array, mean: jax.Array, cov_L: jax.Array):
        z = jax.random.normal(key, mean.shape)
        flat_mean = mean.ravel()
        # jnp.linalg.cholesky return L s.t. H = LL^{T}
        # so L^{T} x = z implies x = L^{-T} z and x ~ N(0, L^{-T} L^{-1}) = N(0, (L L^{T})^{-1})
        delta = jsp.linalg.solve_triangular(cov_L, z, lower=True, trans=1)
        return (flat_mean + delta).reshape(mean.shape)

    # Convert keys into the structure of `mean_tree`
    treedef = tree_util.tree_structure(mean_tree)
    keys = jax.random.split(key, treedef.num_leaves)
    key_tree = tree_util.tree_unflatten(treedef, keys)

    theta = sample_leaf(key_tree, mean_tree, cov_L_tree)
    return theta


@partial(jax.jit, static_argnames=("logp_fn_flat", "dim"))
@partial(jax.vmap, in_axes=(0, None, None, None), out_axes=0)
def expmap(sample: jax.Array, theta_map: jax.Array, logp_fn_flat: Callable, dim: int, dt: float = 0.1):
    def geodesic_ode(t, y, args):
        c = y[:dim]
        c_prime = y[dim:]

        grad_fn = jax.grad(logp_fn_flat)
        grad_c, hvp_c = jax.jvp(grad_fn, (c,), (c_prime,))

        rhs_c = c_prime
        denominator = 1 + jnp.sum(grad_c**2)
        rhs_c_prime = -grad_c * jnp.dot(grad_c, hvp_c) / denominator

        return jnp.concatenate([rhs_c, rhs_c_prime])

    # We start at the MAP and move toward the sample in tangent space
    velocity = sample - theta_map
    y0 = jnp.concatenate([theta_map, velocity])

    term = ODETerm(geodesic_ode)
    solver = Dopri5()

    solution = diffeqsolve(term, solver, t0=0, t1=1, dt0=dt, y0=y0)

    # Return the position at t=1 (the mapped sample)
    return solution.ys[-1, :dim]


def laplace_approximation(
    logp_fn: Callable[[nnx.Module], float],
    model: nnx.Module,
    key: jax.Array,
    num_samples: int,
    method: LaplaceMethod = LaplaceMethod.STANDARD,
) -> tuple[nnx.State, nnx.GraphDef, jax.Array]:
    state = nnx.state(model)
    graphdef = nnx.graphdef(model)
    theta, unravel = ravel_pytree(state)

    @jax.jit
    def log_p_flat(theta):
        state_unflat = unravel(theta)
        model_reconstructed = nnx.merge(graphdef, state_unflat)
        return -logp_fn(model_reconstructed)

    hessian = jax.hessian(log_p_flat)(theta)
    hessian_L = jnp.linalg.cholesky(hessian)
    dim, _ = hessian.shape

    key, *keys = jax.random.split(key, num_samples + 1)
    keys = jnp.array(keys)

    samples = tree_multivariate_normal(keys, theta, hessian_L)

    match method:
        case LaplaceMethod.RIEMANN:
            samples = expmap(samples, theta, log_p_flat, dim)

    # Apply jax.vmap to unravel so we can unravel multiple samples at the same time
    unravel = jax.vmap(unravel)
    samples = unravel(samples)

    return samples, graphdef, key
