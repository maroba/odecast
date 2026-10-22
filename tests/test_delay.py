import numpy as np
import pytest
import sympy as sp

from odecast import BC, Eq, equilibria, latex, solve, var
from odecast.backends.sympy_backend import BackendError

TOL = {"rtol": 1e-10, "atol": 1e-12}


def exact(t):
    """y' = -y(t - 1), y = 1 for t <= 0, by the method of steps."""
    t = np.asarray(t, dtype=float)
    return np.where(
        t <= 1,
        1 - t,
        np.where(
            t <= 2,
            1 - t + (t - 1) ** 2 / 2,
            1 - t + (t - 1) ** 2 / 2 - (t - 2) ** 3 / 6,
        ),
    )


def test_matches_method_of_steps_solution():
    y = var("y")
    sol = solve(Eq(y.d(), -y.delay(1)), ivp={y: 1.0}, tspan=(0, 3), **TOL)
    tt = np.linspace(0, 3, 31)
    np.testing.assert_allclose(sol.eval(y, tt), exact(tt), atol=1e-10)


def test_parametric_delay_and_history_function():
    y = var("y")
    tau = sp.Symbol("tau")
    sol = solve(
        Eq(y.d(), -y.delay(tau)),
        ivp={y: 1.0},
        tspan=(0, 2),
        params={tau: 1},
        history={y: lambda s: 1.0},
        **TOL,
    )
    assert sol.eval(y, 2.0) == pytest.approx(float(exact(2.0)), abs=1e-10)


def test_history_affects_solution():
    # With history y(s) = 1 + s, y' = -y(t - 1) gives y(1) = 1 - (1/2) on [0, 1]
    y = var("y")
    sol = solve(
        Eq(y.d(), -y.delay(1)),
        ivp={y: 1.0},
        tspan=(0, 1),
        history={y: lambda s: 1 + s},
        **TOL,
    )
    # y' = -(1 + t - 1) = -t, so y(1) = 1 - 1/2
    assert sol.eval(y, 1.0) == pytest.approx(0.5, abs=1e-10)


def test_vector_variable():
    u = var("u", shape=2)
    sol = solve(Eq(u.d(), -u.delay(1)), ivp={u: [1, 2]}, tspan=(0, 3), **TOL)
    np.testing.assert_allclose(
        sol.eval(u, 3.0), np.array([1, 2]) * exact(3.0), atol=1e-10
    )


def test_delayed_logistic_oscillates():
    # Hutchinson's equation has a stable limit cycle for r * tau > pi / 2
    y = var("y")
    sol = solve(Eq(y.d(), 1.8 * y * (1 - y.delay(1))), ivp={y: 0.5}, tspan=(0, 60))
    late = sol.eval(y, np.linspace(40, 60, 400))
    assert late.max() - late.min() > 1.5


def test_delayed_derivative():
    # x'' + x = -k x'(t - tau): compare with a direct method-of-steps integration
    from scipy.integrate import solve_ivp

    x = var("x")
    sol = solve(
        Eq(x.d(2) + x, -0.5 * x.d().delay(0.5)),
        ivp={x: 1, x.d(): 0},
        tspan=(0, 1),
        **TOL,
    )
    first = solve_ivp(
        lambda t, z: [z[1], -z[0]], (0, 0.5), [1, 0], dense_output=True, **TOL
    )
    second = solve_ivp(
        lambda t, z: [z[1], -z[0] - 0.5 * first.sol(t - 0.5)[1]],
        (0.5, 1),
        first.y[:, -1],
        **TOL,
    )
    assert sol.eval(x, 1.0) == pytest.approx(second.y[0, -1], abs=1e-9)


def test_neutral_equations_rejected():
    x = var("x")
    with pytest.raises(ValueError, match="neutral"):
        solve(Eq(x.d(2) + x, -x.d(2).delay(0.5)), ivp={x: 1, x.d(): 0}, tspan=(0, 1))


def test_unsupported_uses():
    y = var("y")
    eq = Eq(y.d(), -y.delay(1))
    with pytest.raises(ValueError, match="delay"):
        equilibria(eq)
    with pytest.raises(BackendError, match="delay"):
        solve(eq, backend="sympy")
    with pytest.raises(ValueError, match="delay"):
        solve(
            Eq(y.d(2), -y.delay(1)),
            bvp=[BC(y, t=0, value=0), BC(y, t=1, value=1)],
            tspan=(0, 1),
        )
    with pytest.raises(ValueError, match="only used for delay"):
        solve(Eq(y.d(), -y), ivp={y: 1}, tspan=(0, 1), history={y: 1})


def test_latex():
    y = var("y")
    assert latex(Eq(y.d(), -y.delay(1))) == "y' = - y(t - 1)"
