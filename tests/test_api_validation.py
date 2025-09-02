import pytest
from odecast import t, var, Eq, solve
from odecast.errors import MissingInitialConditionError, OrderMismatchError


def test_missing_initial_condition():
    y = var("y")
    eq = Eq(y.d(2) + y, 0)
    with pytest.raises(MissingInitialConditionError):
        solve(eq, ivp={y: 1.0}, tspan=(0, 1), backend="scipy")


def test_order_mismatch():
    y = var("y", order=1)
    eq = Eq(y.d(2) + y, 0)
    with pytest.raises(OrderMismatchError):
        solve(eq, ivp={y: 0.0, y.d(): 0.0}, tspan=(0, 1), backend="scipy")
