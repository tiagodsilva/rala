import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, diffeqsolve

# consider the ODE y'' + y' = c
# this may be written as z = y', z' + z = c
# or [y z]' = [z \\  c - z]


def f(t, y, args, c=2):
    return jnp.array([y[1], c - y[1]])


term = ODETerm(f)
solver = Dopri5()
y0 = jnp.array([2.0, 3.0])
solution = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=y0)

print(solution.ys)
