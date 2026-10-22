import numpy as np
import pytest
import sympy as sp

from odecast import Eq, solve, t, var


def _solve_scalar(eq, y, ic, tspan=(0, 1)):
    return solve(eq, ivp=ic, tspan=tspan, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("sin", [sp.sin, np.sin])
def test_sine_of_variable(sin):
    # Pendulum: compare against an independent SciPy integration
    from scipy.integrate import solve_ivp

    y = var("y")
    sol = _solve_scalar(Eq(y.d(2) + sin(y), 0), y, {y: 1.0, y.d(): 0.0})
    ref = solve_ivp(
        lambda _, z: [z[1], -np.sin(z[0])], (0, 1), [1.0, 0.0], rtol=1e-10, atol=1e-12
    )
    assert sol[y][-1] == pytest.approx(ref.y[0, -1], abs=1e-7)


def test_power_operator():
    # y' = -y**2, y(0) = 1  =>  y = 1 / (1 + t)
    y = var("y")
    sol = _solve_scalar(Eq(y.d() + y**2, 0), y, {y: 1.0})
    assert sol[y][-1] == pytest.approx(0.5, abs=1e-8)


def test_unary_minus():
    y = var("y")
    sol = _solve_scalar(Eq(-y.d() - y, 0), y, {y: 1.0})
    assert sol[y][-1] == pytest.approx(np.exp(-1), abs=1e-8)


def test_exp_of_variable():
    # y' = exp(-y), y(0) = 0  =>  y = log(1 + t)
    y = var("y")
    sol = _solve_scalar(Eq(y.d(), np.exp(-y)), y, {y: 0.0})
    assert sol[y][-1] == pytest.approx(np.log(2), abs=1e-8)


@pytest.mark.parametrize("forcing", [sp.sin(t), np.sin(t)])
def test_time_dependent_forcing_without_symbol(forcing):
    # y' = sin(t), y(0) = 0  =>  y = 1 - cos(t)
    y = var("y")
    sol = _solve_scalar(Eq(y.d(), forcing), y, {y: 0.0})
    assert sol[y][-1] == pytest.approx(1 - np.cos(1), abs=1e-8)


def test_arithmetic_with_t():
    y = var("y")
    sol = _solve_scalar(Eq(y.d(), 2 * t), y, {y: 0.0})
    assert sol[y][-1] == pytest.approx(1.0, abs=1e-8)


def test_numpy_scalar_coefficient():
    y = var("y")
    sol = _solve_scalar(Eq(y.d() + np.float64(2.0) * y, 0), y, {y: 1.0})
    assert sol[y][-1] == pytest.approx(np.exp(-2), abs=1e-8)


def test_variable_only_inside_function():
    # z occurs only inside SymPy functions
    x, z = var("x"), var("z")
    eqs = [Eq(x.d(), sp.cos(z)), Eq(sp.exp(z.d()) - 1, 0)]
    sol = solve(eqs, ivp={x: 0.0, z: 0.0}, tspan=(0, 1), rtol=1e-10, atol=1e-12)
    assert sol[x][-1] == pytest.approx(1.0, abs=1e-8)


def test_vector_power_and_negation():
    u = var("u", shape=2)
    sol = solve(
        Eq(-u.d() - u**2, 0), ivp={u: [1.0, 2.0]}, tspan=(0, 1), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(sol[u][:, -1], [0.5, 2 / 3], atol=1e-8)


def test_function_of_vector_variable_rejected():
    u = var("u", shape=2)
    with pytest.raises((TypeError, sp.SympifyError), match="components"):
        sp.sin(u)
