from functools import partial
from typing import Generic, Optional, TypeVar

import flax.nnx as nnx
import flax.struct as struct
import jax
import jax.numpy as jnp

ExtraParamsType = TypeVar("ExtraParamsType", bound=nnx.Variable)


class ExtraParamsWrapper(nnx.Variable):
    # Contains optimizable variables which are not used in the Laplace approximation
    # (e.g., variance, or internal layers of an MLP for certain models)
    pass


class Buffer(nnx.Variable):
    # Non-trainable data.
    pass


@struct.dataclass
class DataNorm:
    X_mu: jax.Array
    X_std: jax.Array
    y_mu: jax.Array
    y_std: jax.Array

    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def norm(self, x: jax.Array):
        return (x - self.X_mu) / self.X_std

    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def denorm(self, x: jax.Array):
        return x * self.y_std + self.y_mu


def forward_modules(x: jax.Array, layer: nnx.Module):
    y = nnx.tanh(x)
    y = layer(y)
    return y, None


def forward_weights(x: jax.Array, layer: tuple[jax.Array, jax.Array]):
    weights, bias = layer
    y = nnx.tanh(x)  # (b, dmid)
    y = jnp.dot(y, weights) + bias[None, ...]  # (b, dmid)
    return y, None


class MLP(nnx.Module, Generic[ExtraParamsType]):
    extra_params: Optional[ExtraParamsType]

    def __init__(
        self,
        din: int,
        dmid: int,
        dout: int,
        nlayers: int = 1,
        extra_params: Optional[ExtraParamsType] = None,
        *,
        rngs: nnx.Rngs,
    ):
        # nlayers = 0 implies linear regression
        self.din = din
        self.dmid = dmid if nlayers > 0 else din
        self.dout = dout
        self.nlayers = nlayers
        self.extra_params = (
            ExtraParamsWrapper(extra_params) if extra_params else None
        )

        initializer = nnx.initializers.lecun_uniform()

        @nnx.split_rngs(splits=nlayers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs: nnx.Rngs):
            return nnx.Linear(
                self.dmid, self.dmid, rngs=rngs, kernel_init=initializer
            )

        self.linear_in = nnx.Linear(
            self.din, self.dmid, rngs=rngs, kernel_init=initializer
        )
        self.layers = create_layer(rngs)
        self.linear_out = nnx.Linear(
            self.dmid, self.dout, rngs=rngs, kernel_init=initializer
        )

        self.data_norm = Buffer(
            DataNorm(
                X_mu=jnp.zeros((din,)),
                X_std=jnp.ones((din,)),
                y_mu=jnp.zeros((dout,)),
                y_std=jnp.ones((dout,)),
            )
        )

    def __call__(self, x: jax.Array):
        # x_norm = self.data_norm.norm(x)
        y = self.linear_in(x)
        y, _ = jax.lax.scan(
            f=forward_modules,
            init=y,
            xs=self.layers,
            length=self.nlayers,
        )
        y = self.linear_out(y)
        # y = self.data_norm.denorm(y)
        return y

    def set_scale(self, X: jax.Array, y: jax.Array):
        self.data_norm.value = self.data_norm.value.replace(
            X_mu=X.mean(0),
            X_std=X.std(0),
            y_mu=y.mean(0),
            y_std=y.std(0),
        )


class MLPLastLayer(nnx.Module, Generic[ExtraParamsType]):
    extra_params: Optional[ExtraParamsType] = None

    def __init__(
        self,
        din: int,
        dmid: int,
        dout: int,
        nlayers: int = 1,
        extra_params: Optional[ExtraParamsType] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.din = din
        self.dmid = dmid if nlayers > 0 else din
        self.dout = dout
        self.nlayers = nlayers
        self.extra_params = (
            ExtraParamsWrapper(extra_params) if extra_params else None
        )

        # JAX standard initializer.
        initializer = nnx.initializers.lecun_uniform()

        @nnx.split_rngs(splits=nlayers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs: nnx.Rngs):
            return ExtraParamsWrapper(
                initializer(rngs(), (self.dmid, self.dmid))
            )

        # Create the layers
        self.linear_in = ExtraParamsWrapper(
            initializer(rngs(), shape=(self.din, self.dmid))
        )
        self.layers = create_layer(rngs)
        self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

        @nnx.split_rngs(splits=nlayers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_biases(_: nnx.Rngs):
            return ExtraParamsWrapper(jnp.zeros((self.dmid,)))

        self.b_in = ExtraParamsWrapper(jnp.zeros((self.dmid,)))
        self.b_layers = create_biases(rngs)

        self.data_norm = Buffer(
            DataNorm(
                X_mu=jnp.zeros((din,)),
                X_std=jnp.ones((din,)),
                y_mu=jnp.zeros((dout,)),
                y_std=jnp.ones((dout,)),
            )
        )

    def __call__(self, x: jax.Array):
        x_norm = self.data_norm.norm(x)
        y = jnp.dot(x_norm, self.linear_in) + self.b_in[None, ...]  # (b, dmid)
        y, _ = jax.lax.scan(
            f=forward_weights,
            init=y,
            xs=(self.layers, self.b_layers),
            length=self.nlayers,
        )
        y = self.linear_out(y)
        y = self.data_norm.denorm(y)
        return y

    def set_scale(self, X: jax.Array, y: jax.Array):
        self.data_norm.value = self.data_norm.value.replace(
            X_mu=X.mean(0),
            X_std=X.std(0),
            y_mu=y.mean(0),
            y_std=y.std(0),
        )


# General model for a log posterior distribution.
# Created for compatibility with MLP, nnx.Linear, etc., i.e., other nnx modules.
# logp_fn takes an input and returns a scalar log probability.
class LogPosterior(nnx.Module):
    def __init__(self, theta: jax.Array):
        self.dim = theta.shape[0]
        self.theta = nnx.Param(theta)
