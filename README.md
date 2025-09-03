
# <img src="https://github.com/maroba/odecast/raw/main/docs/odecast-logo.png" alt="Odecast Logo" width="120" align="left" />

# Odecast
### Write ODEs Like Math, Not Code

[![PyPI version](https://img.shields.io/pypi/v/odecast.svg)](https://pypi.org/project/odecast/)
[![License](https://img.shields.io/github/license/maroba/odecast)](https://github.com/maroba/odecast/blob/main/LICENSE)
[![Tests](https://github.com/maroba/odecast/actions/workflows/test.yml/badge.svg)](https://github.com/maroba/odecast/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://pepy.tech/badge/odecast)](https://pepy.tech/project/odecast)

*Stop wrestling with manual ODE reduction. Write differential equations as they appear in your textbook.*

---


## 🚫 The Problem

**Traditional approach:** Converting high-order ODEs to first-order systems manually
```python
# The mathematical equation: y'' + 0.3*y' + y = 0
# Must be manually converted to a system:
def system(t, z):
    y, dy_dt = z
    d2y_dt2 = -0.3*dy_dt - y
    return [dy_dt, d2y_dt2]

# Then solve with scipy.integrate.solve_ivp
sol = solve_ivp(system, [0, 10], [1.0, 0.0])
# Which variable is which? What order? Error-prone! 😵‍💫
```

## ✅ The Odecast Way

**Write math as math:**
```python
from odecast import var, Eq, solve

y = var("y")
eq = Eq(y.d(2) + 0.3*y.d() + y, 0)  # Exactly as written in textbooks!
sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 10))

# Crystal clear, impossible to mess up! 🎯
```

## 🎯 Why Odecast?

> **"Finally, I can copy equations directly from papers into code!"** - *Frustrated PhD student everywhere*

| Traditional Approach | Odecast Approach |
|----------------------|------------------|
| 🔴 Manual reduction to first-order | ✅ Automatic order inference |
| 🔴 Error-prone variable ordering | ✅ Named variables with clear semantics |
| 🔴 Complex system setup | ✅ Write equations as they appear in papers |
| 🔴 Lost connection to original math | ✅ Maintains mathematical clarity |
| 🔴 Vector systems are nightmares | ✅ `u.d(2) + u = 0` for vector ODEs |

## 🚀 Quick Start

```bash
pip install odecast
```

**30-Second Example:** Damped harmonic oscillator

```python
from odecast import var, Eq, solve

# Write the equation exactly as it appears in your textbook
y = var("y")
equation = Eq(y.d(2) + 0.3*y.d() + y, 0)

# Solve it (automatically converts to first-order system)
solution = solve(
    equation, 
    ivp={y: 1.0, y.d(): 0.0},  # y(0)=1, y'(0)=0
    tspan=(0, 10)
)

# Get results
import matplotlib.pyplot as plt
plt.plot(solution.t, solution[y])
plt.show()  # Beautiful decay curve! 📈
```

## Power Features

### **Intuitive Syntax** 
```python
# Traditional nightmare:
def system(t, z): return [z[1], -0.3*z[1] - z[0]]

# Odecast elegance:
Eq(y.d(2) + 0.3*y.d() + y, 0)
```

### **Vector Systems Made Easy**
```python
# 2D harmonic oscillator in one line:
u = var("u", shape=2)
Eq(u.d(2) + u, 0)  # Automatically expands to u₀'' + u₀ = 0, u₁'' + u₁ = 0
```

### **Multiple Backends**
- **SciPy**: Lightning-fast numerics for engineering
- **SymPy**: Exact symbolic solutions for analysis  
- **Auto**: Tries numeric first, falls back to symbolic

### **Bulletproof Validation**
```python
# Clear error messages when you mess up:
y = var("y")
solve(Eq(y.d(2) + y, 0), ivp={y: 1.0})  # Missing y'(0)!
# ❌ ODEValidationError: Missing initial condition for y.d()
```

### **Real-World Examples**

| Domain | Equation | Odecast Code |
|--------|----------|--------------|
| **Physics** | Mass-spring: `mẍ + cẋ + kx = F(t)` | `Eq(m*x.d(2) + c*x.d() + k*x, F)` |
| **Biology** | Population: `ṅ = rn(1-n/K)` | `Eq(n.d() - r*n*(1-n/K), 0)` |
| **Engineering** | RLC Circuit: `Lq̈ + Rq̇ + q/C = V(t)` | `Eq(L*q.d(2) + R*q.d() + q/C, V)` |
| **Economics** | Growth: `K̇ = sY - δK` | `Eq(K.d() - s*Y + delta*K, 0)` |

### **Who Uses Odecast?**

- 🎓 **Researchers**: Copy equations directly from papers
- 👨‍🎓 **Students**: Focus on physics, not programming
- 🏭 **Engineers**: Rapid prototyping of dynamic systems  
- 📊 **Data Scientists**: Time-series modeling made easy

## Show Me The Code

**Example 1: Pendulum with damping**
```python
θ = var("θ")  # Angle
eq = Eq(θ.d(2) + 0.5*θ.d() + np.sin(θ), 0)  # Non-linear!
sol = solve(eq, ivp={θ: np.pi/4, θ.d(): 0}, tspan=(0, 20))
```

**Example 2: Lotka-Volterra (predator-prey)**
```python
x, y = var("x"), var("y")  # Rabbits, foxes
eqs = [
    Eq(x.d() - x*(1 - y), 0),      # Rabbit growth
    Eq(y.d() - y*(-1 + x), 0)      # Fox dynamics  
]
sol = solve(eqs, ivp={x: 1, y: 1}, tspan=(0, 15))
```

**Example 3: Vector oscillator (physics)**
```python
u = var("u", shape=2)  # 2D position
eq = Eq(u.d(2) + 0.1*u.d() + u, 0)  # Damped 2D harmonic oscillator
sol = solve(eq, ivp={u: [1, 0], u.d(): [0, 1]}, tspan=(0, 10))
plt.plot(sol[u[0]], sol[u[1]])  # Phase space plot! 🌀
```

## More Examples

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

---

*Built for scientists, engineers, and students who want to focus on the math, not the code.*

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
# Try numeric first, fall back to symbolic
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

## 📚 More Examples

Check out the `examples/` directory for complete working examples:
- 🔢 **Numeric solutions**: `01_ivp_damped_oscillator.py`  
- 🔣 **Symbolic solutions**: `02_symbolic_simple.py`
- 🏹 **Vector systems**: `04_vector_harmonic_oscillator.py`
- 🔗 **Coupled systems**: `05_vector_mixed_system.py`

```bash
# Run any example
python examples/01_ivp_damped_oscillator.py
```

## ⭐ Like what you see?

**Give us a star!** ⭐ It helps others discover Odecast.

**Share it** with colleagues who are tired of manual ODE reduction.

**Follow us** for updates on new features and improvements.

---

<div align="center">

**Stop fighting with first-order systems. Start writing math like math.**

[⭐ **Star on GitHub**](https://github.com/maroba/odecast) • [📦 **Install from PyPI**](https://pypi.org/project/odecast/) • [📖 **Read the Docs**](https://github.com/maroba/odecast/tree/main/examples)

*Made with ❤️ for scientists, engineers, and students worldwide.*

</div>s.io/github/license/maroba/odecast)](https://github.com/maroba/odecast/blob/main/LICENSE)
[![Tests](https://github.com/maroba/odecast/actions/workflows/test.yml/badge.svg)](https://github.com/maroba/odecast/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://pepy.tech/badge/odecast)](https://pepy.tech/project/odecast)

*Stop wrestling with manual ODE reduction. Write differential equations as they appear in your textbook.*

</div>

---


## 🤝 Contributing

Want to make Odecast even better? We'd love your help!

```bash
git clone https://github.com/maroba/odecast.git
cd odecast  
pip install -e ".[dev]"
pytest  # Run tests
```

**What we need:**
- 📖 More examples from your field
- 🐛 Bug reports and fixes  
- 🚀 Performance improvements
- 📝 Documentation improvements

