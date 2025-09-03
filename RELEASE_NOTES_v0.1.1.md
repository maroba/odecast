# Release Notes: Odecast v0.1.1

**Release Date**: September 3, 2025  
**Tag**: `v0.1.1`  
**Previous Version**: `v0.1.0`

## 🚀 What's New in v0.1.1

This release introduces a **significant performance improvement** by optimizing the auto backend behavior and enhances documentation quality with smooth mathematical plots.

### 🔄 Auto Backend Optimization (BREAKING CHANGE)

**The auto backend now prioritizes numeric solutions over symbolic ones:**

- **Before**: `Auto → SymPy (symbolic) → SciPy (numeric) fallback`
- **After**: `Auto → SciPy (numeric) → SymPy (symbolic) fallback`

**Why this change?**

- ⚡ **Faster**: Numeric solutions are typically 10-100x faster
- 🎯 **More reliable**: Better handling of complex nonlinear systems
- 📊 **User-focused**: Most users solve IVP problems requiring numeric integration
- 🔧 **Practical**: Aligns with typical engineering and scientific workflows

**Migration Guide:**

- ✅ **No code changes required** - the API remains identical
- ℹ️ **Behavior change**: Auto backend will prefer numeric when possible
- 🎯 **Explicit control**: Use `backend="sympy"` if you specifically need symbolic solutions

### 📈 Enhanced Documentation Quality

**Professional-grade plot rendering:**

- Fixed jagged plot issues in Sphinx documentation
- All mathematical examples now render with smooth, publication-quality curves
- Enhanced plot generation using optimized SciPy solver parameters
- Improved visual appeal for GitHub repository and documentation

### 🛠️ Technical Improvements

**Core API Updates:**

- Modified `solve()` function in `src/odecast/api.py` for better backend prioritization
- Enhanced error handling with informative messages when both backends fail
- Improved numeric integration settings for smoother plot output

**Documentation Updates:**

- Updated README.md to reflect new auto backend behavior
- Revised user guide documentation
- Added migration guidance for users

**Testing:**

- Fixed failing test `test_solve_with_auto_backend_fallback_to_scipy`
- Ensured backward compatibility while supporting new behavior
- All 104 tests passing with new logic

## 📋 Full Changelog

### Changed

- **BREAKING CHANGE**: Auto backend now tries numeric (SciPy) first, then falls back to symbolic (SymPy)

### Improved

- Enhanced documentation with smooth, high-quality plots in Sphinx examples
- Updated user guide and README to reflect new auto backend behavior

### Fixed

- Updated failing test to match new auto backend order
- Resolved plot rendering issues in Sphinx documentation

### Technical Details

- Modified `solve()` function to prioritize SciPy backend
- Enhanced plot generation with `max_step` parameter for smoother output
- Updated documentation strings and comments

## 🔧 API Compatibility

### ✅ Fully Backward Compatible API

- All existing function signatures unchanged
- All existing parameter names and types preserved
- Existing user code will continue to work without modifications

### ⚠️ Behavior Change Notice

- Auto backend selection priority has changed
- Results remain mathematically equivalent
- Performance characteristics improved

## 📦 Installation

### Update from v0.1.0

```bash
pip install --upgrade odecast
```

### Fresh installation

```bash
pip install odecast==0.1.1
```

### Verify installation

```python
import odecast
print(odecast.__version__)  # Should print "0.1.1"
```

## 🧪 Verification

Test the new auto backend behavior:

```python
from odecast import var, Eq, solve

# This will now use SciPy first (faster for IVP)
y = var("y")
eq = Eq(y.d(2) + 0.3*y.d() + y, 0)
sol = solve(eq, ivp={y: 1, y.d(): 0}, tspan=(0, 10), backend="auto")
print("Auto backend working correctly!")

# For symbolic solutions, be explicit:
sol_symbolic = solve(eq, backend="sympy")
print("Symbolic backend available when needed!")
```

## 🎯 Next Steps

This release establishes odecast as a mature, performance-oriented library for ODE solving. Future development will focus on:

- Enhanced vector system support
- Boundary value problem (BVP) backend implementation
- Additional solver method options
- Extended documentation with more domain-specific examples

## 🙏 Acknowledgments

Thanks to the community for feedback on performance priorities and documentation quality. This release directly addresses user needs for faster numeric solutions while maintaining the flexibility for symbolic computation when needed.

---

**Download**: [GitHub Release v0.1.1](https://github.com/maroba/odecast/releases/tag/v0.1.1)  
**Documentation**: [User Guide](https://github.com/maroba/odecast/tree/main/docs)  
**PyPI**: [odecast 0.1.1](https://pypi.org/project/odecast/0.1.1/)
