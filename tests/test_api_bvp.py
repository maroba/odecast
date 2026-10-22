import numpy as np
import pytest
import sympy as sp

from odecast import BC, Eq, solve, var
from odecast.errors import MissingInitialConditionError, OverdeterminedConditionsError
from odecast.solution import SolutionBVP


def test_bvp_basic():
    y = var("y")
    eq = Eq(y.d(2) + y, 0)
    sol = solve(
        eq,
        bvp=[BC(y, t=0, value=0), BC(y, t=1, value=1)],
        tspan=(0, 1),
        backend="scipy_bvp",
    )
    assert sol[y].shape[0] > 2


def test_bvp_matches_exact_solution():
    # y'' + y = 0, y(0) = 0, y(1) = 1  =>  y = sin(t) / sin(1)
    y = var("y")
    sol = solve(
        Eq(y.d(2) + y, 0),
        bvp=[BC(y, t=0, value=0), BC(y, t=1, value=1)],
        tspan=(0, 1),
    )
    assert isinstance(sol, SolutionBVP)
    tt = np.linspace(0, 1, 7)
    np.testing.assert_allclose(sol.eval(y, tt), np.sin(tt) / np.sin(1), atol=1e-6)
    np.testing.assert_allclose(sol.eval(y.d(), tt), np.cos(tt) / np.sin(1), atol=1e-5)
    assert isinstance(sol.eval(y, 0.5), float)


def test_bvp_derivative_condition():
    # y'' = -1, y'(0) = 0, y(1) = 0  =>  y = (1 - t^2) / 2
    y = var("y")
    sol = solve(
        Eq(y.d(2), -1),
        bvp=[BC(y.d(), t=0, value=0), BC(y, t=1, value=0)],
        tspan=(0, 1),
    )
    assert sol.eval(y, 0.0) == pytest.approx(0.5)


def test_bvp_nonlinear_guess_selects_branch():
    # The Bratu problem y'' + exp(y) = 0, y(0) = y(1) = 0 has two solutions
    y = var("y")
    eq = Eq(y.d(2) + sp.exp(y), 0)
    bcs = [BC(y, t=0, value=0), BC(y, t=1, value=0)]

    lower = solve(eq, bvp=bcs, tspan=(0, 1))
    upper = solve(eq, bvp=bcs, tspan=(0, 1), guess={y: lambda t: 12 * t * (1 - t)})

    assert lower.eval(y, 0.5) == pytest.approx(0.1405, abs=1e-3)
    assert upper.eval(y, 0.5) == pytest.approx(4.0916, abs=1e-3)


def test_bvp_vector_variable():
    u = var("u", shape=2)
    sol = solve(
        Eq(u.d(2), 0),
        bvp=[BC(u, t=0, value=[0, 1]), BC(u, t=2, value=[2, 1])],
        tspan=(0, 2),
    )
    np.testing.assert_allclose(sol.eval(u, 1.0), [1.0, 1.0], atol=1e-8)


def test_bvp_auto_backend_uses_scipy():
    y = var("y")
    sol = solve(
        Eq(y.d(2) + y, 0),
        bvp=[BC(y, t=0, value=0), BC(y, t=1, value=1)],
        tspan=(0, 1),
        backend="auto",
    )
    assert isinstance(sol, SolutionBVP)


def test_bvp_too_few_conditions():
    y = var("y")
    with pytest.raises(MissingInitialConditionError):
        solve(Eq(y.d(2) + y, 0), bvp=[BC(y, t=0, value=0)], tspan=(0, 1))


def test_bvp_too_many_conditions():
    y = var("y")
    bcs = [BC(y, t=0, value=0), BC(y, t=1, value=1), BC(y.d(), t=1, value=0)]
    with pytest.raises(OverdeterminedConditionsError):
        solve(Eq(y.d(2) + y, 0), bvp=bcs, tspan=(0, 1))


def test_bvp_condition_on_highest_derivative():
    y = var("y")
    bcs = [BC(y, t=0, value=0), BC(y.d(2), t=1, value=1)]
    with pytest.raises(OverdeterminedConditionsError):
        solve(Eq(y.d(2) + y, 0), bvp=bcs, tspan=(0, 1))


def test_bvp_interior_point_rejected():
    y = var("y")
    bcs = [BC(y, t=0, value=0), BC(y, t=0.5, value=1)]
    with pytest.raises(ValueError, match="end point"):
        solve(Eq(y.d(2) + y, 0), bvp=bcs, tspan=(0, 1))


def test_ivp_and_bvp_together_rejected():
    y = var("y")
    with pytest.raises(ValueError, match="either ivp or bvp"):
        solve(Eq(y.d() + y, 0), ivp={y: 1}, bvp=[BC(y, t=0, value=1)], tspan=(0, 1))
