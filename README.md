<img src="https://github.com/maroba/odecast/raw/main/docs/odecast-logo.png" alt="Odecast logo" width="120" align="right" />

# odecast

[![PyPI version](https://img.shields.io/pypi/v/odecast.svg)](https://pypi.org/project/odecast/)
[![Tests](https://github.com/maroba/odecast/actions/workflows/test.yml/badge.svg)](https://github.com/maroba/odecast/actions)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://maroba.github.io/odecast)

odecast is a Python library for solving ordinary differential equations of
arbitrary order. Equations are written in their natural form, e.g.
`y'' + 0.3 y' + y = 0`, and odecast performs the reduction to a first-order
system that numerical integrators require. Numerical solutions are computed
with SciPy, closed-form solutions with SymPy.

Full documentation: <https://maroba.github.io/odecast>

## Installation

```bash
pip install odecast
```

This installs NumPy, SciPy and SymPy. Python 3.10 or later is required. The
example scripts additionally need Matplotlib (`pip install "odecast[examples]"`).

## Example

The damped harmonic oscillator `y'' + 0.3 y' + y = 0` with `y(0) = 1`,
`y'(0) = 0`:

```python
from odecast import var, Eq, solve

y = var("y")
eq = Eq(y.d(2) + 0.3 * y.d() + y, 0)

sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 10))

sol.t          # time points
sol[y]         # y(t)
sol[y.d()]     # y'(t)
sol.eval(y, 5.0)
```

For comparison, the same problem written directly against
`scipy.integrate.solve_ivp` requires the state vector to be set up by hand:

```python
def rhs(t, z):
    y, dy = z
    return [dy, -0.3 * dy - y]

solve_ivp(rhs, (0, 10), [1.0, 0.0])
```

This is manageable for a single equation but becomes tedious and error-prone
for coupled systems of mixed order, which is the case odecast is designed for.

## Features

- **Arbitrary order.** The order of each variable is inferred from the
  equations; no manual reduction is needed.
- **Coupled systems of mixed order.** Pass a list of equations; each
  variable may appear with a different highest derivative.
- **Nonlinear equations.** Powers and functions of the unknowns, such as
  `y**2` or `sp.sin(y)` (or `np.sin(y)`), can be used directly.
- **Vector and matrix variables.** `var("u", shape=2)` or
  `var("M", shape=(2, 2))` declares an array-valued unknown. Equations such
  as `Eq(u.d(2) + u, 0)` are expanded component-wise.
- **Initial and boundary value problems.** Two-point boundary value problems
  are solved with `scipy.integrate.solve_bvp`.
- **Symbolic solutions.** The SymPy backend returns closed-form solutions,
  with initial or boundary conditions applied.
- **Validation.** Missing, superfluous or inconsistent conditions are
  reported before solving, with a message naming the offending condition.

### Coupled systems

```python
x, y = var("x"), var("y")
eqs = [
    Eq(x.d() - x * (1 - y), 0),
    Eq(y.d() - y * (x - 1), 0),
]
sol = solve(eqs, ivp={x: 1.0, y: 0.5}, tspan=(0, 15))
```

### Nonlinear and time-dependent terms

The independent variable is available as `odecast.t`:

```python
import sympy as sp
from odecast import t

theta = var("theta")
eq = Eq(theta.d(2) + 0.1 * theta.d() + sp.sin(theta), 0.5 * sp.cos(2 * t))
```

### Vector and matrix variables

```python
u = var("u", shape=2)
eq = Eq(u.d(2) + 0.1 * u.d() + u, 0)
sol = solve(eq, ivp={u: [1.0, 0.0], u.d(): [0.0, 1.0]}, tspan=(0, 10))

sol[u]      # array of shape (2, n)
sol[u[0]]   # array of shape (n,)
```

### Boundary value problems

```python
from odecast import BC

y = var("y")
sol = solve(
    Eq(y.d(2) + y, 0),
    bvp=[BC(y, t=0, value=0), BC(y, t=1, value=1)],
    tspan=(0, 1),
)
sol.eval(y, 0.5)
```

For nonlinear problems an initial guess can be passed, e.g.
`guess={y: lambda t: t * (1 - t)}`.

### Symbolic solutions

```python
sol = solve(Eq(y.d(2) + y, 0), backend="sympy", ivp={y: 1, y.d(): 0})
sol.as_expr(y)       # cos(t)
sol.eval(y, 1.0)     # 0.5403...
```

Without conditions the general solution is returned, e.g.
`C1*sin(t) + C2*cos(t)`.

### Solver options

Additional keyword arguments to `solve` are passed to
`scipy.integrate.solve_ivp` (or `solve_bvp` for boundary value problems):

```python
solve(eq, ivp=ic, tspan=(0, 10), method="LSODA", rtol=1e-9, max_step=0.01)
```

## Limitations

odecast is at an early stage (alpha). In particular:

- Boundary value problems must be two-point problems: all conditions lie at
  the end points of `tspan`. Problems with unknown parameters, such as
  eigenvalue problems, are not supported.
- Functions must be applied to components of vector variables
  (`sp.sin(u[0])`), not to the vector as a whole. Operations on vector and
  matrix variables are element-wise; matrix products are not supported.
- The SymPy backend can solve what `sympy.dsolve` can solve: single
  equations and linear systems, but generally not nonlinear coupled systems.

## Examples

The [`examples/`](examples/) directory contains runnable scripts, e.g.

```bash
python examples/01_ivp_damped_oscillator.py
```

More examples with plots are in the
[documentation](https://maroba.github.io/odecast/examples.html).

## Development

```bash
git clone https://github.com/maroba/odecast.git
cd odecast
pip install -e ".[dev]"
pytest
ruff check .
black --check .
```

See the
[contributing guide](https://maroba.github.io/odecast/contributing.html) for
details. Bug reports and pull requests are welcome.

## License

MIT. See [LICENSE](LICENSE).
