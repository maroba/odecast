import numpy as np
import pytest
import sympy as sp

from odecast import BC, Eq, solve, var
from odecast.compile import compile_system

TOL = {"rtol": 1e-10, "atol": 1e-12}


def test_eval_uses_dense_output():
    # Between solver steps, eval is as accurate as the solver itself
    y = var("y")
    sol = solve(Eq(y.d(2) + y, 0), ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 10), **TOL)
    tt = np.linspace(0, 10, 1001)
    assert np.abs(sol.eval(y, tt) - np.cos(tt)).max() < 1e-8


def test_solver_options_are_passed_through():
    y = var("y")
    sol = solve(Eq(y.d() + y, 0), ivp={y: 1.0}, tspan=(0, 1), t_eval=[0, 0.5, 1])
    np.testing.assert_array_equal(sol.t, [0, 0.5, 1])

    sol = solve(Eq(y.d() + y, 0), ivp={y: 1.0}, tspan=(0, 1), first_step=1e-4)
    assert sol.t[1] == pytest.approx(1e-4)


@pytest.mark.parametrize("method", ["Radau", "BDF", "LSODA"])
def test_implicit_methods_get_jacobian(method):
    # Robertson's stiff chemical kinetics problem; a + b + c is conserved
    a, b, c = var("a"), var("b"), var("c")
    eqs = [
        Eq(a.d(), -0.04 * a + 1e4 * b * c),
        Eq(b.d(), 0.04 * a - 1e4 * b * c - 3e7 * b * b),
        Eq(c.d(), 3e7 * b * b),
    ]
    sol = solve(
        eqs,
        ivp={a: 1, b: 0, c: 0},
        tspan=(0, 1e5),
        method=method,
        rtol=1e-6,
        atol=1e-10,
    )
    _, jac, *_ = sol.as_first_order()
    assert jac is not None
    assert sol[a][-1] + sol[b][-1] + sol[c][-1] == pytest.approx(1.0)


def test_jacobian_is_exact():
    y = var("y")
    k = sp.Symbol("k")
    sol = solve(
        Eq(y.d(2) + k * y**3, 0),
        ivp={y: 1.0, y.d(): 0.0},
        tspan=(0, 1),
        params={k: 2.0},
        method="Radau",
    )
    _, jac, *_ = sol.as_first_order()
    np.testing.assert_allclose(jac(0.0, np.array([0.5, 0.0])), [[0, 1], [-1.5, 0]])


def test_explicit_methods_skip_jacobian():
    y = var("y")
    sol = solve(Eq(y.d() + y, 0), ivp={y: 1.0}, tspan=(0, 1))
    assert sol.as_first_order()[1] is None


class TestParameters:
    def test_parameter_values(self):
        y = var("y")
        k = sp.Symbol("k")
        sol = solve(
            Eq(y.d(2) + k * y, 0),
            ivp={y: 1, y.d(): 0},
            tspan=(0, 1),
            params={k: 4.0},
            **TOL,
        )
        assert sol.eval(y, 1.0) == pytest.approx(np.cos(2), abs=1e-8)

    def test_parameters_by_name(self):
        y = var("y")
        sol = solve(
            Eq(y.d() + sp.Symbol("k") * y, 0),
            ivp={y: 1},
            tspan=(0, 1),
            params={"k": 1.0},
            **TOL,
        )
        assert sol.eval(y, 1.0) == pytest.approx(np.exp(-1), abs=1e-8)

    def test_missing_parameter(self):
        y = var("y")
        with pytest.raises(ValueError, match="Missing values for parameters: k"):
            solve(Eq(y.d() + sp.Symbol("k") * y, 0), ivp={y: 1}, tspan=(0, 1))

    def test_parameter_in_initial_condition(self):
        y = var("y")
        y0 = sp.Symbol("y0")
        sol = solve(Eq(y.d(), 0), ivp={y: 2 * y0}, tspan=(0, 1), params={y0: 1.5})
        assert sol.eval(y, 1.0) == pytest.approx(3.0)

    def test_parameter_named_like_internal_state(self):
        y = var("y")
        x0 = sp.Symbol("x0")
        sol = solve(
            Eq(y.d(), -x0 * y), ivp={y: 1}, tspan=(0, 1), params={x0: 1.0}, **TOL
        )
        assert sol.eval(y, 1.0) == pytest.approx(np.exp(-1), abs=1e-8)

    def test_parameters_in_bvp(self):
        y = var("y")
        k = sp.Symbol("k")
        sol = solve(
            Eq(y.d(2) + k * y, 0),
            bvp=[BC(y, t=0, value=0), BC(y, t=1, value=1)],
            tspan=(0, 1),
            params={k: 1.0},
        )
        assert sol.eval(y, 0.5) == pytest.approx(np.sin(0.5) / np.sin(1), abs=1e-6)

    def test_parameters_symbolic_backend(self):
        y = var("y")
        k = sp.Symbol("k", positive=True)
        eq = Eq(y.d(2) + k * y, 0)
        expr = solve(eq, backend="sympy", ivp={y: 1, y.d(): 0}).as_expr(y)
        assert sp.simplify(expr - sp.cos(sp.sqrt(k) * sp.Symbol("t", real=True))) == 0
        expr = solve(eq, backend="sympy", ivp={y: 1, y.d(): 0}, params={k: 4}).as_expr(
            y
        )
        assert sp.simplify(expr - sp.cos(2 * sp.Symbol("t", real=True))) == 0

    def test_compilation_is_cached(self):
        y = var("y")
        k = sp.Symbol("k")
        eqs = [Eq(y.d(2) + k * y, 0)]
        orders = {y: 2}
        assert compile_system(eqs, orders) is compile_system(eqs, orders)
        # A new but identical variable reuses the compiled system
        y2 = var("y")
        assert compile_system([Eq(y2.d(2) + k * y2, 0)], {y2: 2}) is compile_system(
            eqs, orders
        )
