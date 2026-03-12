# Agent Guidelines for RALA Repository

This document outlines the guidelines for AI agents operating within this repository. Adherence to these standards ensures consistency, maintainability, and efficient development.

## 1. Project Overview

RALA (Riemannian Approximate Laplace Approximation) is a JAX/Flax-based library for Bayesian neural network approximation using Laplace methods. The project uses:
- **Python**: 3.13+
- **Build**: setuptools
- **ML Framework**: JAX, Flax (nnx)
- **ODE Solving**: Diffrax
- **Optimization**: Optax
- **CLI**: Typer

## 2. Build, Lint, and Test Commands

### Build Commands

```bash
# Build source distribution
python -m build --sdist

# Build wheel distribution
python -m build --wheel

# Install in editable mode
pip install -e .
```

### Linting Commands

```bash
# Ruff (if installed)
ruff check .

# Flake8 (if installed)
flake8 .
```

### Testing Commands

```bash
# Run all tests
pytest

# Run tests in a specific file
pytest rala/tests/test_laplace.py

# Run a specific test function
pytest rala/tests/test_laplace.py::test_sampling

# Run with verbose output
pytest -v
```

## 3. Code Style Guidelines

### Imports

- Imports grouped by PEP 8: standard library, third-party, local/project
- Use explicit imports: `import jax.numpy as jnp`
- Minimize imports, only include necessary modules
- Example import order:
  ```python
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
  
  from rala.laplace import LaplaceMethod, laplace_approximation
  from rala.models import MLP
  ```

### Formatting

- Follow PEP 8: 4 spaces for indentation, 88 character line limit (Black default)
- Use meaningful blank lines to separate logical code blocks
- Use trailing commas in multi-line collections

### Types

- Use type hints extensively: `jax.Array`, `nnx.Module`, `Callable`, `float`, `int`
- Use JAX/Flax-specific types from `typing` module
- Generic types with TypeVar:
  ```python
  from typing import Generic, TypeVar
  ExtraParams = TypeVar("ExtraParams", bound=struct.PyTreeNode)
  ```

### Naming Conventions

- **Variables/Functions**: `snake_case` (e.g., `logp_fn`, `train_step`, `laplace_approximation`)
- **Classes**: `CamelCase` (e.g., `MLP`, `LogPosterior`, `LaplaceMethod`)
- **Constants/Enums**: `UPPER_SNAKE_CASE` (e.g., `LaplaceMethod.STANDARD`)
- **Files**: `snake_case.py`

### Error Handling

- Minimal error handling; rely on JAX runtime checks
- Use `assert` statements for validation with descriptive messages:
  ```python
  assert metric_fn is not None, "`metric_fn` should be provided when `FISHER` is used"
  ```
- Use `match/case` for enum-based dispatch

### JAX and Flax Conventions

- Use JAX transformations: `@jax.jit`, `@partial(jax.vmap, ...)`, `jax.grad`, `jax.lax.scan`
- Use `flax.nnx` for neural network modules
- Use `@partial` decorator for jit with static arguments
- Use `nnx.jit` for training functions with models
- Use `optax` for optimizers and learning rate schedules
- State management: `nnx.state()`, `nnx.graphdef()`, `nnx.merge()`
- Use `jax.random` with explicit keys: `jax.random.split(key, n)`
- Use `jax.vmap` for vectorization, typically with `in_axes`/`out_axes`

### Documentation

- Docstrings: concise, explain purpose, args, and return values
- Inline comments: sparingly, for complex/non-obvious logic
- Example docstring:
  ```python
  def laplace_approximation(
      logp_fn: Callable[[nnx.Module], float],
      model: nnx.Module,
      key: jax.Array,
      num_samples: int,
      method: LaplaceMethod = LaplaceMethod.STANDARD,
  ) -> tuple[nnx.State, nnx.GraphDef, jax.Array]:
      """Compute Laplace approximation for a model's posterior."""
  ```

## 4. Project Structure

```
rala/
├── __init__.py          # Package init (minimal)
├── laplace.py           # Core Laplace approximation logic
├── models.py            # Neural network models (MLP, LogPosterior)
├── train.py             # Training utilities
└── utils.py             # Helper functions

examples/
├── classification.py    # Classification example (uses typer CLI)
├── regression.py        # Regression example
└── sampling.py         # Sampling example
```

## 5. Testing Guidelines

- Test files: `test_*.py` or `*_test.py` in `tests/` directory
- Use pytest fixtures for JAX/Flax setup
- Test both eager and jit-compiled functions
- Include edge cases for vmap dimensions

## 6. Cursor and Copilot Rules

No Cursor rules (`.cursor/rules/` or `.cursorrules`) or Copilot instructions (`.github/copilot-instructions.md`) found in repository. Follow general best practices for code quality and security.
