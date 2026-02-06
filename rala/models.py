import flax.nnx as nnx
import jax


def forward(x: jax.Array, layer: nnx.Module):
    y = nnx.leaky_relu(x)
    y = layer(y)
    return y, None


class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, nlayers: int = 0, *, rngs: nnx.Rngs):
        # nlayers = 0 implies linear regression
        self.din = din
        self.dmid = dmid if nlayers > 0 else din
        self.dout = dout
        self.nlayers = nlayers

        @nnx.split_rngs(splits=nlayers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs: nnx.Rngs):
            return nnx.Linear(self.dmid, self.dmid, rngs=rngs)

        self.linear_in = nnx.Linear(self.din, self.dmid, rngs=rngs) if nlayers > 0 else nnx.identity
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


if __name__ == "__main__":
    # Create and MLP with more than one layer
    rngs = nnx.Rngs(42)
    model = MLP(din=10, dmid=32, dout=1, nlayers=2, rngs=rngs)
    x = jax.numpy.ones((10,))
    y = model(x)
    assert y.shape == (1,)

    # Create an MLP with less than one layer
    model = MLP(din=10, dmid=32, dout=1, nlayers=0, rngs=rngs)
    x = jax.numpy.ones((10,))
    y = model(x)
    assert y.shape == (1,)
