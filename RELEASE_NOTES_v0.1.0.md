# Odecast v0.1.0 - Initial Release

**🎉 First PyPI Release!**

We're excited to announce the first public release of Odecast - a Python library that lets you write differential equations exactly as they appear in textbooks, without manual reduction to first-order systems.

## 🚀 What is Odecast?

Stop wrestling with manual ODE reduction! Write differential equations in natural mathematical form and let Odecast handle the rest.

```python
# Instead of this mess:
def system(t, z):
    y, dy_dt = z
    d2y_dt2 = -0.3*dy_dt - y
    return [dy_dt, d2y_dt2]

# Write this:
y = var("y")
eq = Eq(y.d(2) + 0.3*y.d() + y, 0)
sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 10))
```

## ✨ Key Features

- **🎯 Intuitive Syntax**: Write `y.d(2)` for second derivatives, exactly like math notation
- **🔗 Vector Variables**: Handle multi-dimensional systems with `var("u", shape=2)`
- **⚡ Multiple Backends**: SciPy for numeric solutions, SymPy for symbolic solutions
- **🛡️ Automatic Validation**: Clear error messages for missing initial conditions
- **🔄 Automatic Reduction**: Converts high-order ODEs to first-order systems automatically
- **📊 Easy Results**: Access solutions with `sol[y]`, `sol[y.d()]`

## 🎨 What You Can Solve

- **Physics**: Mass-spring systems, pendulums, oscillators
- **Biology**: Population dynamics, predator-prey models  
- **Engineering**: RLC circuits, control systems
- **Economics**: Growth models, dynamic systems

## 📦 Installation

```bash
pip install odecast
```

## 🏆 Example Gallery

**Damped Harmonic Oscillator:**
```python
from odecast import var, Eq, solve

y = var("y")
eq = Eq(y.d(2) + 0.3*y.d() + y, 0)
sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 10))
```

**Lotka-Volterra (Predator-Prey):**
```python
x, y = var("x"), var("y")  # Rabbits, foxes
eqs = [
    Eq(x.d() - x*(1 - y), 0),
    Eq(y.d() - y*(-1 + x), 0)
]
sol = solve(eqs, ivp={x: 1, y: 1}, tspan=(0, 15))
```

**2D Vector Oscillator:**
```python
u = var("u", shape=2)
eq = Eq(u.d(2) + 0.1*u.d() + u, 0)
sol = solve(eq, ivp={u: [1, 0], u.d(): [0, 1]}, tspan=(0, 10))
```

## 🔧 Technical Details

- **Python Requirements**: 3.10+
- **Dependencies**: NumPy, SciPy, SymPy
- **Backends**: SciPy (numeric IVP), SymPy (symbolic), Auto (tries symbolic first)
- **License**: MIT

## 🎯 Who Should Use This?

- **🎓 Researchers**: Copy equations directly from papers into code
- **👨‍🎓 Students**: Focus on physics and math, not programming complexity
- **🏭 Engineers**: Rapid prototyping of dynamic systems
- **📊 Data Scientists**: Time-series modeling with differential equations

## 🔮 Coming Soon

- Boundary Value Problem (BVP) support
- Matrix variables for large systems  
- Performance optimizations
- More backend options

## 🤝 Contributing

Found a bug? Have a feature request? Want to contribute examples from your field?

- **GitHub**: https://github.com/maroba/odecast
- **Issues**: https://github.com/maroba/odecast/issues
- **Examples**: Check out the `examples/` directory

## 🙏 Acknowledgments

Special thanks to the scientific Python community and everyone who provided feedback during development.

---

**Ready to stop fighting with first-order systems?**

```bash
pip install odecast
```

