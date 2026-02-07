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


@partial(jax.jit, static_argnames=("hvp_fn", "dim", "dt"))
@partial(jax.vmap, in_axes=(0, None, None), out_axes=0)
def expmap(c_i: jax.Array, hvp_fn: callable, dim: int, dt: float = 0.1):
    # Integrate from c_i to c_e
    def geodesic_ode(t: float, c_c_prime: jax.Array, args):
        dim, hvp_fn = args
        c = c_c_prime[:dim]
        c_prime = c_c_prime[dim:]

        grad_c, hvp_c = hvp_fn(c, c_prime)

        rhs_c = c_prime
        rhs_c_prime = -grad_c * jnp.dot(c_prime, hvp_c) / (1 + jnp.sum(grad_c**2))

        return jnp.hstack([rhs_c, rhs_c_prime])

    term = ODETerm(geodesic_ode)
    solver = Dopri5()
    solution = diffeqsolve(term, solver, t0=0, t1=1, dt0=dt, y0=c_i, args=(dim, hvp_fn))
    return solution.ys[0, :dim]


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
        return logp_fn(model_reconstructed)

    hessian = -jax.hessian(log_p_flat)(theta)
    hessian_L = jnp.linalg.cholesky(hessian)
    dim, _ = hessian.shape

    key, *keys = jax.random.split(key, num_samples + 1)
    keys = jnp.array(keys)

    samples = tree_multivariate_normal(keys, theta, hessian_L)

    match method:
        case LaplaceMethod.RIEMANN:
            grad_fn = jax.grad(log_p_flat)

            def hvp_fn(theta, v):
                return jax.jvp(grad_fn, (theta,), (v,))

            ones = jnp.ones((dim,))
            jax.debug.print("{} {}", grad_fn(ones), hvp_fn(ones, ones))

            samples = expmap(samples, hvp_fn, dim)

    # Apply jax.vmap to unravel so we can unravel multiple samples at the same time
    unravel = jax.vmap(unravel)
    samples = unravel(samples)

    return samples, graphdef, key
