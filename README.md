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

```bash
python examples/sampling.py squiggle --method monge --hmc
```

![Squiggle distribution](./figures/squiggle.png)

## Methods

This package implements several Laplace approximation variants for sampling from posterior distributions:

### Standard Laplace
Basic Gaussian approximation using the Hessian of the log-posterior at the MAP estimate.

### Riemann
Uses Riemannian geometry with geodesic flows to transport samples from the Gaussian approximation to the true posterior.

### Monge
Optimal transport approach using the Monge metric:
```
G(θ) = I + ∇log p(θ) ∇log p(θ)^T
```

### Fisher
Fisher-Rao metric (requires custom metric function).

### HMC Refinement
Hamiltonian Monte Carlo refinement step to improve sample quality.

## Usage

```python
from rala.laplace import laplace_approximation, LaplaceMethod
from rala.models import LogPosterior
import jax
import jax.numpy as jnp

# Define your log posterior
def logp_fn(theta):
    # Example: simple 2D Gaussian
    return -jnp.sum(theta ** 2) / 2

# Find MAP
x_map = jnp.zeros(2)  # or use optimization

# Create model
model = LogPosterior(x_map)

# Sample
key = jax.random.key(42)
samples, _, key = laplace_approximation(
    lambda model: logp_fn(model.theta),
    model,
    key=key,
    num_samples=1000,
    method=LaplaceMethod.MONGE,
)
```

## Examples

- `examples/sampling.py` - Demo with Banana and Squiggle distributions
- `examples/classification.py` - Classification example with neural networks

## Requirements

- Python >= 3.13
- JAX >= 0.9.0.1
- Flax >= 0.12.3
- Diffrax >= 0.7.1
- optax >= 0.2.7
- jaxhmc
- matplotlib >= 3.10.8
- scikit-learn >= 1.8.0
- tqdm >= 4.67.3
- typer >= 0.21.1
