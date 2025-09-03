# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-09-03

### Changed

- **BREAKING CHANGE**: Auto backend now tries numeric (SciPy) first, then falls back to symbolic (SymPy)
  - Previously: Auto → SymPy first → SciPy fallback
  - Now: Auto → SciPy first → SymPy fallback
  - **Rationale**: Numeric solutions are generally faster and more reliable for typical IVP problems
  - **Impact**: Better performance for most users, minimal API impact

### Improved

- Enhanced documentation with smooth, high-quality plots in Sphinx examples
  - Fixed plot smoothness issues by using finer time grids and better scipy solver options
  - All mathematical examples now render with publication-quality smooth curves
- Updated user guide and README to reflect new auto backend behavior

### Fixed

- Updated failing test `test_solve_with_auto_backend_fallback_to_scipy` to match new behavior
- Resolved plot rendering issues in Sphinx documentation that caused jagged curves

### Technical Details

- Modified `solve()` function in `src/odecast/api.py` to prioritize SciPy backend
- Enhanced plot generation in documentation examples with `max_step` parameter for smoother output
- Updated documentation strings and comments to reflect new backend priority

### Migration Guide

If you were relying on the auto backend trying SymPy first:

- **No code changes needed** - the API remains the same
- **Behavior change**: Auto backend will now prefer numeric solutions when possible
- **Explicitly specify backend**: Use `backend="sympy"` if you need symbolic solutions specifically

## [0.1.0] - 2025-09-02

### Added

- Initial release of odecast
- Core DSL for writing ODEs like mathematical equations
- Support for higher-order ODEs without manual reduction
- Vector and matrix variable support
- Multiple backends: SciPy (numeric), SymPy (symbolic), Auto
- Comprehensive test suite
- Full documentation with examples
- PyPI package distribution

### Features

- **Intuitive Syntax**: Write `y.d(2) + 0.3*y.d() + y = 0` instead of manual first-order reduction
- **Vector Variables**: Support for `u = var("u", shape=2)` for multi-dimensional systems
- **Automatic Order Inference**: Automatically detects highest derivative orders
- **Multiple Backends**: Choose between numeric (SciPy) and symbolic (SymPy) solvers
- **Comprehensive Validation**: Clear error messages for common mistakes
- **Rich Examples**: Physics, engineering, biology, and economics examples

### Backends

- **SciPy Backend**: Fast numeric integration for IVP problems
- **SymPy Backend**: Exact symbolic solutions for linear ODEs
- **Auto Backend**: Intelligent backend selection (initially symbolic-first)

### Documentation

- Complete Sphinx documentation with mathematical examples
- API reference with detailed function signatures  
- User guide with step-by-step tutorials
- Real-world examples across multiple domains
