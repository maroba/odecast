# Odecast

A DSL for ordinary differential equations with symbolic and numeric backends.

## Installation

```bash
pip install -e ".[math,dev]"
```

## Quick Start

```python
from odecast import t, var, Eq, solve

# Define a variable
y = var("y")

# Create an equation: y'' + 0.3*y' + y = 0
eq = Eq(y.d(2) + 0.3*y.d() + y, 0)

# Solve with initial conditions
sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0.0, 5.0), backend="scipy")
```

## Development

This project follows a test-driven development approach with "playbooks" that incrementally build functionality.

Run tests:
```bash
pytest
```

## Status

This is currently in development. The basic project structure is in place, but most functionality is not yet implemented.
