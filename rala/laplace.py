from functools import partial
from typing import Callable

import flax.nnx as nnx
import flax.struct as struct
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as tree_util
from jax.flatten_util import ravel_pytree


@partial(jax.jit, static_argnames=("unravel",))
@partial(jax.vmap, in_axes=(0, None, None, None), out_axes=0)
def tree_multivariate_normal(
    key: jax.Array,
    mean_tree: struct.PyTreeNode,
    cov_L_tree: struct.PyTreeNode,
    unravel: callable,
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
    return unravel(theta)


def laplace_approximation(
    log_p: Callable[[nnx.Module], float],
    model: nnx.Module,
    key: jax.Array,
    num_samples: int,
):
    state = nnx.state(model)
    graphdef = nnx.graphdef(model)
    theta, unravel = ravel_pytree(state)

    @jax.jit
    def log_p_flat(theta):
        state_unflat = unravel(theta)
        model_reconstructed = nnx.merge(graphdef, state_unflat)
        return log_p(model_reconstructed)

    hessian = -jax.hessian(log_p_flat)(theta)
    hessian_L = jnp.linalg.cholesky(hessian)

    key, *keys = jax.random.split(key, num_samples + 1)
    keys = jnp.array(keys)

    return tree_multivariate_normal(keys, theta, hessian_L, unravel), graphdef, key
