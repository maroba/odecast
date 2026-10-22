import numpy as np
import pytest
import sympy as sp
from scipy.linalg import expm

from odecast import Eq, solve, var

TOL = {"rtol": 1e-10, "atol": 1e-12}
A = np.array([[0.0, 1.0], [-2.0, -0.3]])
X0 = np.array([1.0, 0.0])


@pytest.mark.parametrize(
    "matrix",
    [A, A.tolist(), sp.Matrix([[0, 1], [-2, sp.Rational(-3, 10)]])],
    ids=["ndarray", "list", "sympy"],
)
def test_linear_system(matrix):
    u = var("u", shape=2)
    sol = solve(Eq(u.d(), matrix @ u), ivp={u: X0}, tspan=(0, 2), **TOL)
    np.testing.assert_allclose(sol.eval(u, 2.0), expm(2 * A) @ X0, atol=1e-9)


def test_matrix_with_parameter():
    u = var("u", shape=2)
    k = sp.Symbol("k")
    B = np.array([[0, 1], [-k, -0.3]], dtype=object)
    sol = solve(Eq(u.d(), B @ u), ivp={u: X0}, tspan=(0, 2), params={k: 2.0}, **TOL)
    np.testing.assert_allclose(sol.eval(u, 2.0), expm(2 * A) @ X0, atol=1e-9)


def test_second_order_coupled_oscillators():
    # Symmetric mode of two coupled oscillators has frequency 1
    u = var("u", shape=2)
    K = np.array([[2, -1], [-1, 2]])
    sol = solve(
        Eq(u.d(2) + K @ u, 0), ivp={u: [1, 1], u.d(): [0, 0]}, tspan=(0, np.pi), **TOL
    )
    np.testing.assert_allclose(sol.eval(u, np.pi), [-1, -1], atol=1e-8)


def test_non_square_matrix():
    u, v = var("u", shape=2), var("v", shape=3)
    B = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]])
    sol = solve(
        [Eq(u.d(), B @ v), Eq(v.d(), -v)],
        ivp={u: [0, 0], v: [1, 1, 1]},
        tspan=(0, 1),
        **TOL,
    )
    np.testing.assert_allclose(
        sol.eval(u, 1.0), B @ np.ones(3) * (1 - np.exp(-1)), atol=1e-9
    )


def test_matrix_variable():
    M = var("M", shape=(2, 2))
    sol = solve(Eq(M.d(), A @ M), ivp={M: np.eye(2)}, tspan=(0, 1), **TOL)
    np.testing.assert_allclose(sol.eval(M, 1.0), expm(A), atol=1e-9)


def test_parameter_times_vector():
    u = var("u", shape=2)
    k = sp.Symbol("k")
    sol = solve(Eq(u.d(), -k * u), ivp={u: [1, 2]}, tspan=(0, 1), params={k: 1}, **TOL)
    np.testing.assert_allclose(
        sol.eval(u, 1.0), np.exp(-1) * np.array([1, 2]), atol=1e-9
    )


def test_scalar_variable_plus_vector():
    u, y = var("u", shape=2), var("y")
    sol = solve(
        [Eq(u.d(), y + u), Eq(y.d(), -y)], ivp={u: [0, 0], y: 1}, tspan=(0, 1), **TOL
    )
    np.testing.assert_allclose(sol.eval(u, 1.0), [np.sinh(1)] * 2, atol=1e-9)


def test_misaligned_shapes():
    u = var("u", shape=2)
    with pytest.raises(ValueError, match="not aligned"):
        np.ones((2, 3)) @ u


def test_result_shape_must_match_equation():
    u = var("u", shape=2)
    with pytest.raises(ValueError, match="same shape"):
        solve(Eq(u.d(), np.ones((3, 2)) @ u), ivp={u: X0}, tspan=(0, 1))
