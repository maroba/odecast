"""
Tests for differential-algebraic equations
"""

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from odecast import BC, Eq, equilibria, fit, solve, var
from odecast.backends.sympy_backend import BackendError
from odecast.errors import NonSolvableFormError, OverdeterminedConditionsError

TOL = {"rtol": 1e-10, "atol": 1e-12}


@pytest.fixture
def kinetics():
    """A <=> B in fast equilibrium (B = K A), B -> C slowly; T = A + B."""
    T, C, A, B = var("T"), var("C"), var("A"), var("B")
    k, K = sp.symbols("k K", positive=True)
    eqs = [Eq(T.d(), -k * B), Eq(C.d(), k * B), Eq(B, K * A), Eq(A + B, T)]
    return eqs, (T, C, A, B), (k, K)


def pendulum():
    """Pendulum in Cartesian coordinates with string tension lam (index 3)."""
    x, y, lam = var("x"), var("y"), var("lam")
    eqs = [Eq(x.d(2), -lam * x), Eq(y.d(2), -lam * y - 9.81), Eq(x**2 + y**2, 1)]
    theta0 = np.pi / 3
    ic = {x: np.sin(theta0), x.d(): 0.0, y: -np.cos(theta0), y.d(): 0.0}
    return eqs, (x, y, lam), ic, theta0


class TestIndexOne:
    def test_symbolic_elimination(self, kinetics):
        eqs, (T, C, A, B), (k, K) = kinetics
        sol = solve(
            eqs, ivp={T: 1.0, C: 0.0}, tspan=(0, 5), params={k: 0.5, K: 3}, **TOL
        )
        rate = 0.5 * 3 / 4
        tt = np.linspace(0, 5, 11)
        np.testing.assert_allclose(sol.eval(T, tt), np.exp(-rate * tt), atol=1e-9)
        np.testing.assert_allclose(
            sol.eval(B, tt), 0.75 * np.exp(-rate * tt), atol=1e-9
        )
        assert sol[A].shape == sol.t.shape
        assert np.abs(sol.constraint_residuals()).max() < 1e-12

    def test_symbolic_backend(self, kinetics):
        eqs, (T, C, _A, B), (k, K) = kinetics
        sol = solve(eqs, backend="sympy", ivp={T: 1, C: 0})
        t = sp.Symbol("t", real=True)
        expected = K / (K + 1) * sp.exp(-K * k * t / (K + 1))
        assert sp.simplify(sol.as_expr(B) - expected) == 0

    def test_event_on_algebraic_variable(self, kinetics):
        eqs, (T, C, _A, B), (k, K) = kinetics
        sol = solve(
            eqs,
            ivp={T: 1, C: 0},
            tspan=(0, 10),
            params={k: 0.5, K: 3},
            events=Eq(B, 0.3),
            **TOL,
        )
        assert sol.t_events[0] == pytest.approx([np.log(2.5) / 0.375])

    def test_fit(self, kinetics):
        eqs, (T, C, _A, B), (k, K) = kinetics
        tt = np.linspace(0, 5, 30)
        result = fit(
            eqs,
            {B: (tt, 0.75 * np.exp(-0.375 * tt))},
            guess={k: 1.0},
            ivp={T: 1, C: 0},
            params={K: 3},
        )
        assert result.params[k] == pytest.approx(0.5, rel=1e-6)

    def test_boundary_value_problem(self):
        y, z = var("y"), var("z")
        sol = solve(
            [Eq(y.d(2), -z), Eq(z, y)],
            bvp=[BC(y, t=0, value=0), BC(y, t=1, value=1)],
            tspan=(0, 1),
        )
        assert sol.eval(z, 0.5) == pytest.approx(np.sin(0.5) / np.sin(1), abs=1e-6)

    def test_equilibria(self):
        x, z = var("x"), var("z")
        results = equilibria([Eq(x.d(), -z), Eq(z, x * (x - 1))])
        assert [(e.point[x], e.point[z], e.stability) for e in results] == [
            (0, 0, "unstable"),
            (1, 0, "stable"),
        ]


class TestHigherIndex:
    def test_pendulum_matches_polar_form(self):
        eqs, (x, y, lam), ic, theta0 = pendulum()
        th = var("th")
        ref = solve(
            Eq(th.d(2) + 9.81 * sp.sin(th), 0),
            ivp={th: theta0, th.d(): 0},
            tspan=(0, 10),
            **TOL,
        )
        sol = solve(eqs, ivp=ic, tspan=(0, 10), rtol=1e-8, atol=1e-10)

        tt = np.linspace(0, 10, 200)
        theta, omega = ref.eval(th, tt), ref.eval(th.d(), tt)
        np.testing.assert_allclose(sol.eval(x, tt), np.sin(theta), atol=1e-6)
        np.testing.assert_allclose(sol.eval(y, tt), -np.cos(theta), atol=1e-6)
        # String tension per unit mass and length: g cos(theta) + theta'^2
        np.testing.assert_allclose(
            sol.eval(lam, tt), 9.81 * np.cos(theta) + omega**2, atol=1e-4
        )
        assert np.abs(sol.constraint_residuals(tt)).max() < 1e-7

    def test_stabilization_limits_drift(self):
        eqs, (_x, _y, _lam), ic, _ = pendulum()
        tt = np.linspace(0, 10, 200)
        drift = np.abs(solve(eqs, ivp=ic, tspan=(0, 10)).constraint_residuals(tt)).max()
        unstabilized = solve(eqs, ivp=ic, tspan=(0, 10), baumgarte=0.0)
        assert drift < 1e-3
        assert np.abs(unstabilized.constraint_residuals(tt)).max() > 100 * drift

    def test_inconsistent_position(self):
        eqs, (x, y, _lam), ic, _ = pendulum()
        with pytest.raises(ValueError, match=r"x\*\*2 \+ y\*\*2 - 1 is 0.25"):
            solve(eqs, ivp={**ic, x: 1.0, y: -0.5}, tspan=(0, 1))

    def test_inconsistent_velocity(self):
        eqs, (x, _y, _lam), ic, _ = pendulum()
        with pytest.raises(ValueError, match="derivative of order 1"):
            solve(eqs, ivp={**ic, x.d(): 1.0}, tspan=(0, 1))

    def test_not_supported_by_sympy_backend_or_bvp(self):
        eqs, (x, y, _lam), _, _ = pendulum()
        with pytest.raises(BackendError, match="index 1"):
            solve(eqs, backend="sympy")
        bcs = [
            BC(x, t=0, value=0),
            BC(y, t=0, value=-1),
            BC(x, t=1, value=0),
            BC(y, t=1, value=-1),
        ]
        with pytest.raises(ValueError, match="index 1"):
            solve(eqs, bvp=bcs, tspan=(0, 1))
        with pytest.raises(ValueError, match="index 1"):
            equilibria(eqs)


class TestNumeric:
    def test_tank_with_colebrook_friction(self):
        # Tank draining through a pipe; the friction factor f follows the
        # implicit Colebrook equation, so v and f are computed numerically.
        g, D, length, eps, nu, k_loss = 9.81, 0.05, 20.0, 1e-4, 1e-6, 0.5
        area = np.pi * D**2 / 4
        h, v, f = var("h"), var("v"), var("f")
        eqs = [
            Eq(h.d(), -area * v),
            Eq(g * h, (1 + k_loss + f * length / D) * v**2 / 2),
            Eq(
                1 / sp.sqrt(f),
                -2
                * sp.log(eps / (3.7 * D) + 2.51 / (v * D / nu * sp.sqrt(f)))
                / sp.log(10),
            ),
        ]

        def velocity(hh, guess):
            def residual(z):
                vv, ff = z
                re = vv * D / nu
                return [
                    g * hh - (1 + k_loss + ff * length / D) * vv**2 / 2,
                    1 / np.sqrt(ff)
                    + 2 * np.log10(eps / (3.7 * D) + 2.51 / (re * np.sqrt(ff))),
                ]

            return fsolve(residual, guess, xtol=1e-13)

        last = {"z": [1.0, 0.02]}

        def rhs(_, y):
            last["z"] = velocity(y[0], last["z"])
            return [-area * last["z"][0]]

        ref = solve_ivp(rhs, (0, 500), [2.0], dense_output=True, **TOL)

        sol = solve(eqs, ivp={h: 2.0, v: 1.0, f: 0.02}, tspan=(0, 500), **TOL)
        tt = np.linspace(0, 500, 6)
        np.testing.assert_allclose(sol.eval(h, tt), ref.sol(tt)[0], atol=1e-8)
        np.testing.assert_allclose(sol.eval(v, 0.0), velocity(2.0, [1.0, 0.02])[0])
        assert np.abs(sol.constraint_residuals(tt)).max() < 1e-10

        radau = solve(
            eqs,
            ivp={h: 2.0, v: 1.0, f: 0.02},
            tspan=(0, 500),
            method="Radau",
            rtol=1e-8,
            atol=1e-10,
        )
        assert radau.eval(h, 500.0) == pytest.approx(ref.sol(500.0)[0], abs=1e-6)

    @pytest.mark.parametrize(("guess", "expected"), [(0.5, 0.25), (-0.5, 2.25)])
    def test_guess_selects_branch(self, guess, expected):
        # z**2 = y has two roots; the guess picks one. y' = -z.
        y, z = var("y"), var("z")
        sol = solve(
            [Eq(y.d(), -z), Eq(z**2, y)], ivp={y: 1.0, z: guess}, tspan=(0, 1), **TOL
        )
        assert sol.eval(y, 1.0) == pytest.approx(expected, abs=1e-8)

    def test_transcendental_equation(self):
        # Kepler-like equation z - 0.5 sin(z) = y with y = t
        y, z = var("y"), var("z")
        sol = solve(
            [Eq(y.d(), 1), Eq(z - 0.5 * sp.sin(z), y)], ivp={y: 0.0}, tspan=(0, 2)
        )
        zz = sol.eval(z, 2.0)
        assert zz - 0.5 * np.sin(zz) == pytest.approx(2.0, abs=1e-6)

    def test_unsolvable_algebraic_equation(self):
        y, z = var("y"), var("z")
        with pytest.raises(
            RuntimeError, match="Could not solve the algebraic equations"
        ):
            solve([Eq(y.d(), -z), Eq(z**2, y)], ivp={y: -1.0, z: 1.0}, tspan=(0, 1))


class TestStructure:
    def test_too_few_algebraic_equations(self):
        x, z = var("x"), var("z")
        with pytest.raises(NonSolvableFormError, match="need 1 algebraic equation"):
            solve(Eq(x.d(), z), ivp={x: 0.0}, tspan=(0, 1))

    def test_algebraic_variables_not_determined(self):
        x, z, w = var("x"), var("z"), var("w")
        eqs = [Eq(x.d(), z + w), Eq(z + w, 1), Eq(z + w, x)]
        with pytest.raises(NonSolvableFormError):
            solve(eqs, ivp={x: 1.0}, tspan=(0, 1))

    def test_boundary_condition_on_algebraic_variable(self):
        y, z = var("y"), var("z")
        with pytest.raises(OverdeterminedConditionsError, match="algebraic variable"):
            solve(
                [Eq(y.d(2), -z), Eq(z, y)],
                bvp=[BC(y, t=0, value=0), BC(z, t=1, value=1)],
                tspan=(0, 1),
            )
