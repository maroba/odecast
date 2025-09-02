import numpy as np
import pytest
from odecast import t, var, Eq, solve


def test_second_order_ivp_numeric():
    y = var("y")  # order inferred
    eq = Eq(y.d(2) + 0.3 * y.d() + y, 0)

    sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0.0, 5.0), backend="scipy")

    assert hasattr(sol, "t")
    assert isinstance(sol.t, np.ndarray)
    y_vals = sol[y]
    yprime_vals = sol[y.d()]
    assert y_vals.shape == sol.t.shape
    assert yprime_vals.shape == sol.t.shape


def test_as_first_order_contract():
    y = var("y")
    eq = Eq(y.d(2) + y, 0)
    sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0.0, 1.0), backend="scipy")
    f, jac, x0, t0, mapping = sol.as_first_order()
    assert callable(f)
    assert isinstance(x0, np.ndarray)
    assert isinstance(mapping, dict)
    assert mapping[y][0] == mapping[(y, 0)]
    assert mapping[y][1] == mapping[(y, 1)]
