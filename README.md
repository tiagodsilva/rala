# Riemann Laplace (rala)

Implementation of Riemann Laplace variants for Bayesian sampling.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
python examples/sampling.py banana --method monge --hmc
```

![Banana distribution](./figures/banana.png)

## Methods

- **Standard** — Gaussian approximation via Hessian at MAP
- **Riemann** — Riemannian geometry with geodesic flows
- **Monge** — Optimal transport using `G(θ) = I + ∇log p(θ) ∇log p(θ)^T`
- **Fisher** — Fisher-Rao metric (requires custom metric)
- **HMC** — Hamiltonian Monte Carlo refinement

## Usage

```python
from rala.laplace import laplace_approximation, LaplaceMethod
from rala.models import LogPosterior
import jax.numpy as jnp

logp_fn = lambda theta: -jnp.sum(theta ** 2) / 2
model = LogPosterior(jnp.zeros(2))

key = jax.random.key(42)
samples, _, key = laplace_approximation(
    lambda m: logp_fn(m.theta), model, key=key, 
    num_samples=1000, method=LaplaceMethod.MONGE,
)
```

## Examples

- `examples/sampling.py` — Banana and Squiggle distributions
- `examples/classification.py` — Neural network classification
