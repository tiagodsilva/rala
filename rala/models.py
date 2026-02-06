import flax.nnx as nnx
import jax


def forward(x: jax.Array, layer: nnx.Module):
    y = nnx.leaky_relu(x)
    y = layer(y)
    return y, None


class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, nlayers: int = 1, *, rngs: nnx.Rngs):
        # nlayers = 0 implies linear regression
        self.din = din
        self.dmid = dmid if nlayers > 0 else din
        self.dout = dout
        self.nlayers = nlayers

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


class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        self.din = din
        self.dout = dout
        self.linear = nnx.Linear(self.din, self.dout, rngs=rngs)

    def __call__(self, x: jax.Array):
        return self.linear(x)
