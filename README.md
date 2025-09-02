
# <img src="https://github.com/maroba/odecast/raw/main/docs/odecast-logo.png" alt="Odecast Logo" width="120" align="left" />

# Odecast

[![PyPI version](https://img.shields.io/pypi/v/odecast.svg)](https://pypi.org/project/odecast/)
[![License](https://img.shields.io/github/license/maroba/odecast)](https://github.com/maroba/odecast/blob/main/LICENSE)
[![Tests](https://github.com/maroba/odecast/actions/workflows/test.yml/badge.svg)](https://github.com/maroba/odecast/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python library for solving higher-order ordinary differential equations (ODEs) without manual reduction. Write equations in natural mathematical form (e.g. `y.d(2) + 0.3*y.d() + y = 0`) and let odecast handle order inference, validation, conversion to first-order systems, and solving via SymPy (symbolic) or SciPy (numeric).

## Features

- **Intuitive syntax**: Write ODEs as they appear in textbooks using `y.d(2)` for derivatives
- **Vector/Matrix variables**: Support for systems with `var("u", shape=2)` and component access `u[0]`, `u[1]`
- **Multiple backends**: Symbolic solutions via SymPy, numeric solutions via SciPy  
- **Automatic order inference**: No need to manually convert to first-order systems
- **Automatic equation expansion**: Vector equations automatically expand to component equations
- **Comprehensive validation**: Clear error messages for missing or inconsistent conditions
- **Flexible interface**: Support for IVP (initial value problems) and BVP (boundary value problems)

## Installation

```bash
pip install -e ".[math,dev]"
```

## Quick Start

```python
from odecast import t, var, Eq, solve

# Define a variable
y = var("y")

# Create an equation: y'' + 0.3*y' + y = 0 (damped harmonic oscillator)
eq = Eq(y.d(2) + 0.3*y.d() + y, 0)

# Solve with initial conditions y(0) = 1, y'(0) = 0
sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0.0, 10.0), backend="scipy")

# Access solution data
y_values = sol[y]        # Position over time
velocity = sol[y.d()]    # Velocity over time
times = sol.t            # Time points

# Evaluate at specific times
position_at_5s = sol.eval(y, 5.0)

# Vector variables for systems
u = var("u", shape=2)                    # 2D vector variable
eq_vector = Eq(u.d(2) + u, 0)          # Vector harmonic oscillator  
sol_vector = solve(eq_vector, 
                   ivp={u: [1.0, 0.0], u.d(): [0.0, 1.0]}, 
                   tspan=(0, 5), backend="scipy")
positions = sol_vector[u]               # 2×N array of [x, y] positions
```

## Vector/Matrix Variables

Odecast supports vector and matrix variables for systems of equations, making it easy to work with multi-dimensional problems.

### Creating Vector Variables

```python
from odecast import t, var, Eq, solve

# Create a 2D vector variable
u = var("u", shape=2)  # Creates u with components u[0], u[1]

# Create a 3D vector variable  
v = var("v", shape=3)  # Creates v with components v[0], v[1], v[2]

# Matrix variables (coming soon)
# A = var("A", shape=(2, 2))  # 2x2 matrix
```

### Component Access

```python
u = var("u", shape=2)

# Access individual components
u0 = u[0]  # First component
u1 = u[1]  # Second component

# Components behave like regular variables
eq1 = Eq(u[0].d(2) + u[0], 0)  # u₀'' + u₀ = 0
eq2 = Eq(u[1].d() + u[0], 0)   # u₁' + u₀ = 0
```

### Vector Operations

```python
u = var("u", shape=2)

# Vector derivatives
u_dot = u.d()      # First derivative: [u[0]', u[1]']
u_ddot = u.d(2)    # Second derivative: [u[0]'', u[1]'']

# Vector equations automatically expand to component equations
eq = Eq(u.d(2) + u, 0)  # Becomes: u[0]'' + u[0] = 0, u[1]'' + u[1] = 0

# Vector arithmetic
position = var("r", shape=2)
velocity = var("v", shape=2) 
eq1 = Eq(position.d() - velocity, 0)     # r' = v
eq2 = Eq(velocity.d() + 0.1*velocity, 0) # v' + 0.1v = 0 (damping)
```

### Vector Initial Conditions

```python
u = var("u", shape=2)
eq = Eq(u.d(2) + u, 0)  # 2D harmonic oscillator

# Vector-style initial conditions
ivp = {
    u: [1.0, 0.5],        # u(0) = [1.0, 0.5]
    u.d(): [0.0, -0.2]    # u'(0) = [0.0, -0.2]
}

sol = solve(eq, ivp=ivp, tspan=(0, 10), backend="scipy")

# Access vector solution (returns 2×N array)
u_trajectory = sol[u]  # Shape: (2, num_timepoints)

# Access individual components (returns 1D arrays)
u0_trajectory = sol[u[0]]
u1_trajectory = sol[u[1]]
```

### Mixed Vector/Scalar Systems

```python
# Coupled system with both scalar and vector variables
x = var("x")           # Scalar variable
u = var("u", shape=2)  # Vector variable

eqs = [
    Eq(x.d(2) + x - u[0], 0),    # x'' + x - u₀ = 0
    Eq(u[0].d() + u[1], x),      # u₀' + u₁ = x  
    Eq(u[1].d() + u[0], 0),      # u₁' + u₀ = 0
]

ivp = {
    x: 1.0,              # x(0) = 1
    x.d(): 0.0,          # x'(0) = 0
    u: [0.5, 0.0],       # u(0) = [0.5, 0.0]
    u.d(): [0.0, 0.0]    # u'(0) = [0.0, 0.0] (automatically filtered for first-order components)
}

sol = solve(eqs, ivp=ivp, tspan=(0, 5), backend="scipy")
```

## Examples

The `examples/` directory contains comprehensive examples:

- `01_ivp_damped_oscillator.py` - Numeric IVP solving with visualization
- `02_symbolic_simple.py` - Symbolic solutions using SymPy backend  
- `03_mixed_orders.py` - Coupled systems with mixed derivative orders
- `04_vector_harmonic_oscillator.py` - 2D harmonic oscillator using vector variables
- `05_vector_mixed_system.py` - Mixed vector/scalar systems
- `06_vector_simple.py` - Simple vector variable introduction

Run any example:
```bash
python examples/01_ivp_damped_oscillator.py
```

## Backends

### SciPy (Numeric)
```python
# Solve numerically over a time span
sol = solve(eq, ivp=conditions, tspan=(0, 10), backend="scipy")
```

### SymPy (Symbolic)  
```python
# Get exact symbolic solution
sol = solve(eq, backend="sympy")
expr = sol.as_expr(y)  # Returns SymPy expression
```

### Auto Backend Selection
```python
# Try symbolic first, fall back to numeric
sol = solve(eq, ivp=conditions, tspan=(0, 10), backend="auto")
```

## API Reference

### Variables and Equations
- `var(name, order=None)` - Create a scalar dependent variable
- `var(name, shape=n)` - Create a vector variable with n components  
- `var(name, shape=(m,n))` - Create a matrix variable (coming soon)
- `y.d(n)` - n-th derivative of variable y
- `u[i]` - Access i-th component of vector variable u
- `u.d(n)` - n-th derivative of vector variable (returns vector derivative)
- `Eq(lhs, rhs)` - Create an equation
- `t` - Independent variable (time)

### Solving
- `solve(equations, ivp=None, bvp=None, tspan=None, backend="auto")`

### Solution Objects
- `sol[y]` - Access solution values for variable y
- `sol[y.d()]` - Access derivative values
- `sol[u]` - Access vector solution (returns 2D array for vector variables)
- `sol[u[i]]` - Access i-th component solution (returns 1D array)
- `sol.eval(target, time_points)` - Evaluate at specific times
- `sol.as_first_order()` - Inspect first-order system representation

## Contributing

This project follows a test-driven development approach. All functionality is defined by comprehensive tests.

### Development Setup
```bash
git clone https://github.com/maroba/odecast.git
cd odecast
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest                    # Run all tests
pytest -v                # Verbose output
pytest tests/test_api*    # Run specific test files
```

### Code Quality
```bash
ruff check .              # Linting
black .                   # Formatting
```

### Contributing Guidelines
1. All new features must have comprehensive tests
2. Follow the existing API patterns and naming conventions
3. Add examples for significant new functionality  
4. Update documentation for user-facing changes
5. Ensure all tests pass before submitting PRs

## Status

Current implementation status:
- ✅ Core DSL (variables, equations, derivatives)
- ✅ Vector/Matrix variables with automatic equation expansion
- ✅ SciPy numeric backend (IVP)
- ✅ SymPy symbolic backend (decoupled systems)
- ✅ Automatic order inference and validation
- ✅ Comprehensive error handling
- ✅ Mixed-order coupled systems
- 🚧 BVP support (boundary value problems) - coming soon

## License

MIT License - see LICENSE file for details.
