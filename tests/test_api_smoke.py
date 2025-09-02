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
    """Test that solve raises NotImplementedError as expected."""
    from odecast import var, Eq, solve

    y = var("y")
    eq = Eq(y.d(2) + y, 0)

    with pytest.raises(NotImplementedError):
        solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 1), backend="scipy")
