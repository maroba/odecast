"""
Basic smoke test to ensure the package imports correctly
"""

import pytest


def test_package_imports():
    """Test that the main package imports without errors."""
    from odecast import t, var, Eq, solve, BC

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
    from odecast import var, Eq

    y = var("y")
    eq = Eq(y.d(2) + y, 0)
    assert eq.lhs is not None
    assert eq.rhs == 0


def test_solve_placeholder():
    """Test that solve raises appropriate errors for unimplemented backends."""
    from odecast import var, Eq, solve

    y = var("y")
    eq = Eq(y.d(2) + y, 0)

    # SymPy backend should now work (Playbook 7 implemented)
    sol = solve(eq, backend="sympy")
    assert hasattr(sol, "as_expr")
    expr = sol.as_expr(y)
    # Should be a SymPy expression
    import sympy as sp

    assert isinstance(expr, sp.Expr)

    # BVP backend should raise NotImplementedError (Milestone 5)
    with pytest.raises(NotImplementedError, match="BVP backend"):
        solve(eq, backend="scipy_bvp")

    # SciPy backend should require IVP and tspan
    with pytest.raises(ValueError, match="IVP conditions required"):
        solve(eq, backend="scipy")

    with pytest.raises(ValueError, match="tspan required"):
        solve(eq, ivp={y: 1.0, y.d(): 0.0}, backend="scipy")
