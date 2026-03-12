from typing import Generic, Optional, TypeVar

import flax.nnx as nnx
import flax.struct as struct
import jax

ExtraParams = TypeVar("ExtraParams", bound=struct.PyTreeNode)


def forward(x: jax.Array, layer: nnx.Module):
    y = nnx.leaky_relu(x)
    y = layer(y)
    return y, None


class MLP(nnx.Module, Generic[ExtraParams]):
    extra_params: Optional[ExtraParams]

    def __init__(
        self,
        din: int,
        dmid: int,
        dout: int,
        nlayers: int = 1,
        extra_params: Optional[ExtraParams] = None,
        *,
        rngs: nnx.Rngs,
    ):
        # nlayers = 0 implies linear regression
        self.din = din
        self.dmid = dmid if nlayers > 0 else din
        self.dout = dout
        self.nlayers = nlayers
        self.extra_params = nnx.Param(extra_params) if extra_params else None

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
            f=forward,
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


def reconstruct_model(last_layer: struct.PyTreeNode, model: nnx.Module):
    state = nnx.state(model)
    graphdef = nnx.graphdef(model)
    state["linear_out"] = last_layer
    model_from_layer = nnx.merge(graphdef, state)
    return model_from_layer
