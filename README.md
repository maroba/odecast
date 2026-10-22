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

This installs NumPy, SciPy and SymPy. Python 3.10 or later is required.
Plotting and `to_pandas()` additionally need Matplotlib and pandas
(`pip install "odecast[examples]"`).

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
  as `Eq(u.d(2) + u, 0)` are expanded component-wise; matrix products are
  written `A @ u`.
- **Parameters.** SymPy symbols in the equations take values from
  `params=`. The system is compiled once and reused for further values.
- **Initial and boundary value problems.** Two-point boundary value problems
  are solved with `scipy.integrate.solve_bvp`. Events, such as a body
  reaching the ground, are located during integration.
- **Differential-algebraic equations.** Variables without derivatives are
  algebraic. Constraints such as `Eq(x**2 + y**2, L**2)` are handled by index
  reduction with stabilization; algebraic equations without closed-form
  solution are solved numerically.
- **Stiff problems.** Implicit solvers receive the exact Jacobian.
- **Symbolic solutions.** The SymPy backend returns closed-form solutions,
  with initial or boundary conditions applied.
- **Analysis.** Equilibria and their stability; fitting parameters to
  measured data.
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

### Parameters and matrix products

SymPy symbols in the equations are parameters. Solving again with other
values reuses the compiled system, so parameter studies are fast:

```python
import numpy as np

u = var("u", shape=2)
k = sp.Symbol("k")
A = np.array([[0, 1], [-k, -0.3]], dtype=object)

for kv in [1.0, 2.0, 4.0]:
    sol = solve(Eq(u.d(), A @ u), ivp={u: [1, 0]}, tspan=(0, 10), params={k: kv})
```

### Differential-algebraic equations

A pendulum in Cartesian coordinates, with the string tension `lam` as an
algebraic variable:

```python
x, y, lam = var("x"), var("y"), var("lam")
eqs = [
    Eq(x.d(2), -lam * x),
    Eq(y.d(2), -lam * y - 9.81),
    Eq(x**2 + y**2, 1),
]
sol = solve(eqs, ivp={x: 0.6, x.d(): 0, y: -0.8, y.d(): 0}, tspan=(0, 10))
sol[lam]                      # tension
sol.constraint_residuals()    # remaining violation of x**2 + y**2 = 1
```

### Events

```python
from odecast import Event

y = var("y")
sol = solve(
    Eq(y.d(2), -9.81),
    ivp={y: 10, y.d(): 0},
    tspan=(0, 10),
    events=[Event(Eq(y, 0), terminal=True)],   # stop when y reaches 0
)
sol.t_events[0]   # [1.428]
```

### Equilibria and fitting

```python
from odecast import equilibria, fit

x, y = var("x"), var("y")
equilibria([Eq(x.d(), x - x * y), Eq(y.d(), x * y - y)])
# [Equilibrium(x=0, y=0, saddle), Equilibrium(x=1, y=1, center)]

c, k = sp.symbols("c k")
result = fit(Eq(y.d(2) + c * y.d() + k * y, 0), {y: (t_data, y_data)},
             guess={c: 1.0, k: 1.0}, ivp={y: 1, y.d(): 0})
result.params, result.stderr
```

### Output

`sol.plot()` plots the solution with Matplotlib, `sol.to_pandas()` returns
a DataFrame, and in Jupyter equations and symbolic solutions are displayed
as formulas.

### Solver options

Additional keyword arguments to `solve` are passed to
`scipy.integrate.solve_ivp` (or `solve_bvp` for boundary value problems):

```python
solve(eq, ivp=ic, tspan=(0, 10), method="LSODA", rtol=1e-9, t_eval=t_points)
```

## Limitations

odecast is at an early stage (alpha). In particular:

- Boundary value problems must be two-point problems: all conditions lie at
  the end points of `tspan`. Problems with unknown parameters, such as
  eigenvalue problems, are not supported.
- Functions must be applied to components of vector variables
  (`sp.sin(u[0])`), not to the vector as a whole.
- Matrix products require one constant operand: `A @ u` works, a product of
  two variables such as `M @ u` does not.
- The SymPy backend can solve what `sympy.dsolve` can solve: single
  equations and linear systems, but generally not nonlinear coupled systems.
- Differential-algebraic systems are reduced by differentiating constraints
  one at a time; structures that need the Pantelides algorithm are not
  supported. Boundary value problems, the SymPy backend and `equilibria()`
  support only algebraic equations that are linear in the algebraic
  variables and determine them directly (index 1).

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
