# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Boundary value problems: `solve(eq, bvp=[BC(...), ...], tspan=(a, b))`
  solves two-point problems with `scipy.integrate.solve_bvp`. Options
  `guess` and `mesh` control the initial guess and mesh.
- Powers (`y**2`), unary minus and functions of the unknowns. SymPy
  functions (`sp.sin(y)`) and NumPy ufuncs (`np.sin(y)`) are both accepted.
- `odecast.t` can be used directly in expressions and functions, e.g.
  `sp.sin(2*t)`; `t.symbol` is no longer needed.
- Matrix variables: `var("M", shape=(m, n))` with component access
  `M[i, j]`, element-wise equations, and solution arrays of shape
  `(m, n, len(t))`.
- The SymPy backend applies initial and boundary conditions and solves
  coupled linear systems. `SolutionExpr.as_expr` accepts derivatives and
  vector variables, and `SolutionExpr.eval` evaluates the solution
  numerically.

### Changed

- NumPy, SciPy and SymPy are now regular dependencies, so `pip install
  odecast` is sufficient. The `math` extra now only adds Matplotlib and is
  kept for backwards compatibility; prefer the new `examples` extra.
- `backend="auto"` no longer falls back to SymPy when the initial conditions
  are invalid; the validation error is raised instead.
- `backend="scipy"` (the default) now also handles boundary value problems;
  `"scipy_bvp"` is kept as an alias.
- Initial conditions for vector variables may also be NumPy arrays. Other
  types raise `TypeError` (previously `ValueError`).
- Component indices are validated: out-of-range indices and the wrong number
  of indices raise `IndexError`.
- Removed the unused placeholder class `odecast.solution.Solution`.
- The `dev` extra now includes Ruff (pinned to 0.17.1) and Black (pinned to
  26.10.1). The unused `.flake8` configuration was removed. The code
  base passes Ruff's default rule set; the only exceptions are configured in
  `pyproject.toml`.
- Rewrote README and documentation. Corrected several examples that did not
  run as written and documented current limitations.

### Fixed

- `odecast.__version__` reported 0.1.0 instead of 0.1.1. The version is now
  defined only in `src/odecast/__init__.py`.
- Vector initial conditions with too many components were silently accepted;
  a `ValueError` is now raised for any length mismatch.
- Solving an initial value problem no longer emits a SciPy warning about the
  unused `jac` argument.

## [0.1.1] - 2025-09-03

### Changed

- `backend="auto"` now tries SciPy first and falls back to SymPy. It
  previously tried SymPy first. Pass `backend="sympy"` explicitly to get the
  old behaviour for symbolic solutions.

### Fixed

- Plots in the documentation examples are now rendered on a fine time grid
  (using `max_step`) instead of the integrator's output points.

## [0.1.0] - 2025-09-02

Initial release.

- Equations of arbitrary order written with `var`, `Eq` and `.d(n)`;
  automatic reduction to a first-order system.
- Coupled systems of mixed order.
- Vector variables (`var("u", shape=n)`) with component-wise expansion of
  equations.
- Validation of initial conditions.
- Backends: SciPy (numerical IVP), SymPy (symbolic), and auto selection.
