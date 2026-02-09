from enum import Enum
from functools import partial
from typing import Callable

import flax.nnx as nnx
import flax.struct as struct
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as tree_util
from diffrax import Dopri5, ODETerm, PIDController, diffeqsolve
from jax.flatten_util import ravel_pytree
from jaxhmc.mcmc import RandomWalkConfig, random_walk


class LaplaceMethod(Enum):
    STANDARD = "standard"
    RIEMANN = "riemann"
    MONGE = "monge"
    FISHER = "fisher"
    RANDOM_WALK = "random-walk"


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None), out_axes=0)
def sample_from_gaussian(
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
def rexpmap(
    sample: jax.Array,
    theta_map: jax.Array,
    logp_fn_flat: Callable,
    dim: int,
    dt: float = None,
):
    # Exponential map for Riemann
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

    solution = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=1,
        dt0=dt,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-6),
        y0=y0,
    )

    # Return the position at t=1 (the mapped sample)
    return solution.ys[-1, :dim]


@partial(jax.jit, static_argnames=("metric_fn", "dim", "dt"))
@partial(jax.vmap, in_axes=(0, None, None, None), out_axes=0)
def fexpmap(
    sample: jax.Array,
    theta_map: jax.Array,
    metric_fn: callable,
    dim: int,
    dt: float = None,
):
    def geodesic_ode(t, y, args):
        theta = y[:dim]
        theta_prime = y[dim:]

        G = metric_fn(theta)
        # This seems to be needed; however, it is probably computationally prohibitive in large dimensions
        G_inv = jnp.linalg.inv(G)

        # Compute the Christoffel contraction
        # \theta'^{T} G \theta'
        def v_G_v(theta_val):
            return jnp.dot(theta_prime, jnp.dot(metric_fn(theta_val), theta_prime))

        # The gradient of this function is \sum_{j} v_{i} \partial_{i} G(\theta) v_{j}
        # The Christoffel symbols are defined as
        # .5 \sum_{l} G^{kl} (\partial_{i} G_{jl} + \partial_{j} G_{il} - \partial_{l} G_{ij})
        # This is \Gamma_{ij}^{k}. Summing over v_i, v_j, we get (since G is symmetric)
        # .5 \sum_{i, j} v_{i} \sum_{l} G^{kl} (\partial_{i} G_{jl} + \partial_{j} G_{il} - \partial_{l} G_{ij}) v_{j}
        # = \sum_{i, j} v_{i} \cdot ( \sum_{l} G^{kl} (\partial_{i} G_{jl}) - .5 \sum_{l} \partial_{l} G_{ij} ) v_{j}
        # = \sum_{i, j} v_{i} v_{j} \sum_{l} \partial_{i} G_{jl} - .5 * v_{i} v_{j} \sum_{l} \partial_{l} G_{ij}

        # The differential of G is a R^{d d d} matrix
        # Let i, j, k be the indices
        # This represents the i, j matrix with entries \sum_{k} \partial_{k} G_{ij}(\theta) v_{k}
        _, term1 = jax.jvp(metric_fn, (theta,), (theta_prime,))
        # This is a vector s.t. the j-th entry is \sum_{i, k} v_{i} \partial_{k} G_{ij}(\theta) v_{k}
        # Re-indexing, this becomes (k <-> j)
        # \sum_{i, j} v_{i} \partial_{j} G_{ik}(\theta) v_{j},
        # which is exactly the term we needed above for the Christoffel contraction.
        term1 = term1 @ theta_prime
        # And this is a vector with entries \sum_{j} v_{i} v_{j} \partial_{i} G_{ij}(\theta) with index i
        term2 = 0.5 * jax.grad(v_G_v)(theta)
        acceleration = -G_inv @ (term1 - term2)

        return jnp.concatenate([theta_prime, acceleration])

    # Again, we sample the velocity from the shifted Laplace approximation
    velocity = sample - theta_map
    y0 = jnp.concatenate([theta_map, velocity])

    # We use the Dormand-Prince 5/4 method for the corresponding ODE
    term = ODETerm(geodesic_ode)
    solver = Dopri5()
    solution = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=1,
        dt0=dt,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-6),
        y0=y0,
    )

    return solution.ys[-1, :dim]


def laplace_approximation(
    logp_fn: Callable[[nnx.Module], float],
    model: nnx.Module,
    key: jax.Array,
    num_samples: int,
    method: LaplaceMethod = LaplaceMethod.STANDARD,
    metric_fn: callable = None,
) -> tuple[nnx.State, nnx.GraphDef, jax.Array]:
    state = nnx.state(model)
    graphdef = nnx.graphdef(model)
    theta_map, unravel = ravel_pytree(state)

    @jax.jit
    def log_p_flat(theta):
        state_unflat = unravel(theta)
        model_reconstructed = nnx.merge(graphdef, state_unflat)
        return -logp_fn(model_reconstructed)

    hessian = jax.hessian(log_p_flat)(theta_map)
    hessian_L = jnp.linalg.cholesky(hessian)
    dim, _ = hessian.shape

    key, *keys = jax.random.split(key, num_samples + 1)
    keys = jnp.array(keys)

    samples = sample_from_gaussian(keys, theta_map, hessian_L)

    match method:
        case LaplaceMethod.RIEMANN:
            samples = rexpmap(samples, theta_map, log_p_flat, dim)
        case LaplaceMethod.MONGE:

            def metric_fn(theta):
                grad_log_p = jax.grad(log_p_flat)(theta)
                return jnp.eye(dim) + jnp.outer(grad_log_p, grad_log_p)

            samples = fexpmap(samples, theta_map, metric_fn, dim)
        case LaplaceMethod.FISHER:
            assert metric_fn is not None, "`metric_fn` should be provided when `FISHER` is used"
            samples = fexpmap(samples, theta_map, metric_fn, dim)
        case LaplaceMethod.RANDOM_WALK:
            config = RandomWalkConfig(key=key, iterations=50, tuning_steps=10)
            samples = random_walk(log_p_flat, samples, config)
            samples = samples[-1]

    # Apply jax.vmap to unravel so we can unravel multiple samples at the same time
    unravel = jax.vmap(unravel)
    samples = unravel(samples)

    return samples, graphdef, key
