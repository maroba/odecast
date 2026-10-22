"""
Basic smoke test to ensure the package imports correctly
"""

import pytest


def test_package_imports():
    """Test that the main package imports without errors."""
    from odecast import BC, Eq, solve, t, var

    # Basic sanity checks
    assert t is not None
    assert callable(var)
    assert callable(Eq)
    assert callable(solve)
    assert BC is not None


def test_basic_variable_creation():
    """Test that variables can be created."""
    from odecast import var

    y = var("y")
    assert y.name == "y"

    # Test derivative creation
    yd = y.d()
    assert hasattr(yd, "variable")
    assert hasattr(yd, "order")


def test_basic_equation_creation():
    """Test that equations can be created."""
    from odecast import Eq, var

    y = var("y")
    eq = Eq(y.d(2) + y, 0)
    assert eq.lhs is not None
    assert eq.rhs == 0


def test_solve_backend_requirements():
    """Test backend selection and the arguments each backend requires."""
    from odecast import Eq, solve, var

    y = var("y")
    eq = Eq(y.d(2) + y, 0)

    # SymPy backend needs no conditions
    sol = solve(eq, backend="sympy")
    assert hasattr(sol, "as_expr")
    expr = sol.as_expr(y)
    # Should be a SymPy expression
    import sympy as sp

    assert isinstance(expr, sp.Expr)

    # BVP backend requires boundary conditions
    with pytest.raises(ValueError, match="bvp for a boundary value problem"):
        solve(eq, backend="scipy_bvp")

    # SciPy backend should require IVP and tspan
    with pytest.raises(ValueError, match="IVP conditions required"):
        solve(eq, backend="scipy")

    with pytest.raises(ValueError, match="tspan required"):
        solve(eq, ivp={y: 1.0, y.d(): 0.0}, backend="scipy")
