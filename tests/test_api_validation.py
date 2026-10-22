import pytest

from odecast import Eq, solve, var
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


def test_auto_backend_reports_missing_initial_condition():
    y = var("y")
    eq = Eq(y.d(2) + y, 0)
    with pytest.raises(MissingInitialConditionError):
        solve(eq, ivp={y: 1.0}, tspan=(0, 1), backend="auto")


@pytest.mark.parametrize("value", [[1.0], [1.0, 0.0, 0.0]])
def test_vector_initial_condition_wrong_length(value):
    u = var("u", shape=2)
    eq = Eq(u.d(2) + u, 0)
    with pytest.raises(ValueError, match="has shape"):
        solve(eq, ivp={u: value, u.d(): [0.0, 0.0]}, tspan=(0, 1))
    with pytest.raises(ValueError, match="has shape"):
        solve(eq, ivp={u: [0.0, 0.0], u.d(): value}, tspan=(0, 1))
