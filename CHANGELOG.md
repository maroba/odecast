# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- NumPy, SciPy and SymPy are now regular dependencies, so `pip install
  odecast` is sufficient. The `math` extra now only adds Matplotlib and is
  kept for backwards compatibility; prefer the new `examples` extra.
- `backend="auto"` no longer falls back to SymPy when the initial conditions
  are invalid; the validation error is raised instead.
- A vector initial condition that is not a list or tuple now raises
  `TypeError` instead of `ValueError`.
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
