# Odecast

A clean, ergonomic DSL for solving ordinary differential equations with symbolic and numeric backends.

## Features

- **Intuitive syntax**: Write ODEs as they appear in textbooks using `y.d(2)` for derivatives
- **Multiple backends**: Symbolic solutions via SymPy, numeric solutions via SciPy  
- **Automatic order inference**: No need to manually convert to first-order systems
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
```

## Examples

The `examples/` directory contains comprehensive examples:

- `01_ivp_damped_oscillator.py` - Numeric IVP solving with visualization
- `02_symbolic_simple.py` - Symbolic solutions using SymPy backend  
- `03_mixed_orders.py` - Coupled systems with mixed derivative orders

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
- `var(name, order=None)` - Create a dependent variable
- `y.d(n)` - n-th derivative of variable y
- `Eq(lhs, rhs)` - Create an equation
- `t` - Independent variable (time)

### Solving
- `solve(equations, ivp=None, bvp=None, tspan=None, backend="auto")`

### Solution Objects
- `sol[y]` - Access solution values for variable y
- `sol[y.d()]` - Access derivative values
- `sol.eval(target, time_points)` - Evaluate at specific times
- `sol.as_first_order()` - Inspect first-order system representation

## Contributing

This project follows a test-driven development approach. All functionality is defined by comprehensive tests.

### Development Setup
```bash
git clone https://github.com/your-username/odecast.git
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
- ✅ SciPy numeric backend (IVP)
- ✅ SymPy symbolic backend
- ✅ Automatic order inference and validation
- ✅ Comprehensive error handling
- ✅ Mixed-order coupled systems
- 🚧 BVP support (boundary value problems) - coming soon

## License

MIT License - see LICENSE file for details.
