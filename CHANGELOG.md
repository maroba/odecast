# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-10-22

### Added

- Event resets: `Event(..., reset={y.d(): -0.8 * y.d()})` changes the state
  at an event and continues the integration (impacts, switching, dosing).
- Sensitivities: `solve(..., sensitivities=[k])` computes derivatives of the
  solution with respect to parameters, including parameters in initial
  values; `sol.sensitivity(y, k, t)` returns them. `fit()` uses them for
  exact derivatives of the residuals.
- Boundary value problems with unknown parameters, e.g. eigenvalue problems:
  `solve(..., bvp=..., unknowns={lam: 3.8})`; results in `sol.unknowns`.
- `periodic_orbit()` computes limit cycles, their period, and Floquet
  multipliers.
- `linearize()` gives state-space models (A, B, C, D) and transfer
  functions around an operating point; `to_scipy()` converts them to
  `scipy.signal.StateSpace`.
- `phase_portrait()` draws direction field, nullclines, equilibria and
  trajectories of planar systems.
- `continuation()` follows branches of equilibria with pseudo-arclength
  continuation and detects folds, Hopf bifurcations and branch points.
- Delay differential equations with constant delays: `y.delay(tau)` and
  `y.d().delay(tau)`, solved with the method of steps; `history=` gives the
  values before the initial time.
- `solve_sde()` simulates stochastic differential equations (Euler-Maruyama,
  Milstein) for ensembles of paths, with mean, standard deviation, quantiles
  and plots.
- `odecast.pde`: time-dependent partial differential equations on grids in
  one to three dimensions with Dirichlet, Neumann and periodic boundaries,
  solved with the method of lines and finite differences from findiff (new
  `pde` extra).

### Changed

- `fit()` no longer aborts when the integration fails for trial parameters
  during the optimization; such steps are rejected.
- Documentation: new pages on analysis, delay and stochastic equations, and
  partial differential equations; equilibria and fitting moved to the
  analysis page.

## [0.3.0] - 2026-10-22

### Added

- Parameters: SymPy symbols in equations take values from
  `solve(..., params={k: 2.0})`, also in initial and boundary values. The
  reduced and compiled system is cached, so solving again with other
  parameter values does not recompile it.
- Events: `solve(..., events=[Event(Eq(y, 0), terminal=True)])` locates
  conditions during integration. Results are in `sol.t_events` and
  `sol.event_values(i, target)`.
- Matrix products `A @ u` with constant matrices (NumPy arrays, lists or
  SymPy matrices, possibly containing parameters) and vector or matrix
  variables.
- `equilibria()` finds the equilibria of autonomous systems and classifies
  their stability from the eigenvalues of the Jacobian.
- `fit()` estimates parameters of initial value problems from measurements
  by nonlinear least squares, with standard errors.
- Differential-algebraic equations. A variable without derivatives is
  algebraic and determined by the equations without highest derivatives.
  - Algebraic equations that are linear in the algebraic variables are
    eliminated symbolically (index 1).
  - Constraints without algebraic variables (higher index, e.g. a pendulum
    in Cartesian coordinates) are differentiated until the algebraic
    variables are determined. Initial values are checked for consistency,
    and drift is damped by Baumgarte stabilization (`baumgarte=` option).
    `sol.constraint_residuals()` shows the remaining violation.
  - Other algebraic equations are solved numerically with Newton's method at
    each evaluation; values for algebraic variables in `ivp` are starting
    guesses.
- `sol.plot()` and `sol.to_pandas()` for numerical solutions.
- LaTeX rendering of equations and symbolic solutions in Jupyter, and
  `odecast.latex()`.

### Changed

- `sol.eval` for initial value problems uses the solver's dense output
  instead of linear interpolation, so values between steps are as accurate
  as the solution. `max_step` is no longer needed for smooth plots.
- The implicit methods `"Radau"`, `"BDF"` and `"LSODA"` receive the exact
  Jacobian, computed symbolically.
- The `examples` and `dev` extras include pandas; `dev` also includes
  Matplotlib.
- A value for a variable that has no derivatives is no longer rejected as an
  extra initial condition; it is the starting guess for that algebraic
  variable.

### Fixed

- Solver options other than `method`, `rtol`, `atol`, `max_step`, `events`
  and `dense_output` were silently ignored; all options are now passed to
  `scipy.integrate.solve_ivp`, e.g. `t_eval` and `first_step`.
- Multiplying a SymPy expression by a vector variable (`k * u`) raised an
  error.
- A variable without derivatives led to an obscure `NameError` inside the
  compiled right-hand side.

## [0.2.0] - 2026-10-22

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
- Removed the unused placeholder class `odecast.solution.Solution` and
  placeholder functions that only raised `NotImplementedError`.
- Python 3.13 is tested and listed as supported.
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

[0.4.0]: https://github.com/maroba/odecast/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/maroba/odecast/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/maroba/odecast/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/maroba/odecast/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/maroba/odecast/releases/tag/v0.1.0
