from typing import Generic, Optional, TypeVar

import flax.nnx as nnx
import jax
import jax.numpy as jnp

ExtraParamsType = TypeVar("ExtraParams", bound=nnx.Variable)


class ExtraParamsWrapper(nnx.Variable):
    # Contains optimizable variables which are not used in the Laplace approximation
    # (e.g., variance, or internal layers of an MLP for certain models)
    pass


def forward_modules(x: jax.Array, layer: nnx.Module):
    y = nnx.tanh(x)
    y = layer(y)
    return y, None


def forward_weights(x: jax.Array, layer: jax.Array):
    y = nnx.tanh(x)  # (b, dmid)
    y = jnp.dot(y, layer)  # (b, dmid)
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

        @nnx.split_rngs(splits=nlayers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs: nnx.Rngs):
            return nnx.Linear(self.dmid, self.dmid, rngs=rngs)

        self.linear_in = nnx.Linear(self.din, self.dmid, rngs=rngs)
        self.layers = create_layer(rngs)
        self.linear_out = nnx.Linear(self.dmid, self.dout, rngs=rngs)

    def __call__(self, x: jax.Array):
        y = self.linear_in(x)
        y, _ = jax.lax.scan(
            f=forward_modules,
            init=y,
            xs=self.layers,
            length=self.nlayers,
        )
        y = self.linear_out(y)
        return y


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

        initializer = nnx.initializers.lecun_uniform()

        # Create the initial layer
        self.linear_in = ExtraParamsWrapper(
            initializer(rngs(), shape=(self.din, self.dmid))
        )

        @nnx.split_rngs(splits=nlayers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs: nnx.Rngs):
            return ExtraParamsWrapper(
                initializer(rngs(), (self.dmid, self.dmid))
            )

        self.layers = create_layer(rngs)
        self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x: jax.Array):
        y = jnp.dot(x, self.linear_in)  # (b, dmid)
        y, _ = jax.lax.scan(
            f=forward_weights,
            init=y,
            xs=self.layers,
            length=self.nlayers,
        )
        y = self.linear_out(y)
        return y


# General model for a log posterior distribution.
# Created for compatibility with MLP, nnx.Linear, etc., i.e., other nnx modules.
# logp_fn takes an input and returns a scalar log probability.
class LogPosterior(nnx.Module):
    def __init__(self, theta: jax.Array):
        self.dim = theta.shape[0]
        self.theta = nnx.Param(theta)
