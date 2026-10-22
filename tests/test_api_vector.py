from odecast import Eq, solve, var


def test_vector_variable_creation():
    u = var("u", shape=3)  # 3D vector u(t)
    v = var("v", shape=(2, 2))  # 2×2 matrix v(t)

    assert u.shape == (3,)
    assert v.shape == (2, 2)
    assert len(u) == 3


def test_vector_component_access():
    u = var("u", shape=3)

    # Component access should return scalar Variable objects
    u0 = u[0]
    u[1]

    assert isinstance(u0, type(var("test")))
    assert u0.name == "u[0]"

    # Derivatives of components
    eq = Eq(u[0].d(2) + u[1].d() + u[2], 0)
    assert eq  # Should construct properly


def test_vectorized_operations():
    u = var("u", shape=2)

    # Vector equation: u'' + u = 0 (componentwise)
    eq = Eq(u.d(2) + u, 0)

    # Should expand to component equations internally
    sol = solve(eq, backend="sympy")

    # Access components of solution
    u0_expr = sol.as_expr(u[0])
    u1_expr = sol.as_expr(u[1])

    assert u0_expr is not None
    assert u1_expr is not None


def test_vector_initial_conditions():
    u = var("u", shape=2)
    eq = Eq(u.d(2) + u, 0)

    # Vector-style IVP specification
    ivp = {u: [1.0, 0.5], u.d(): [0.0, 0.0]}  # u(0) = [1.0, 0.5]  # u'(0) = [0.0, 0.0]

    sol = solve(eq, ivp=ivp, tspan=(0, 1), backend="scipy")

    # Access solution arrays
    u_vals = sol[u]  # Should return 2×N array
    assert u_vals.shape[0] == 2

    # Component access
    u0_vals = sol[u[0]]
    sol[u[1]]
    assert len(u0_vals) == len(sol.t)


def test_mixed_vector_scalar_system():
    x = var("x")  # scalar
    u = var("u", shape=2)  # vector

    eqs = [
        Eq(x.d(2) + x - u[0], 0),  # scalar eq with vector component
        Eq(u[0].d() + u[1], x),  # vector component eq
        Eq(u[1].d() + u[0], 0),  # vector component eq
    ]

    ivp = {x: 1.0, x.d(): 0.0, u: [0.5, 0.0], u.d(): [0.0, 0.0]}

    sol = solve(eqs, ivp=ivp, tspan=(0, 1), backend="scipy")

    assert sol[x].shape == sol.t.shape
    assert sol[u].shape == (2, len(sol.t))


def test_matrix_variable_ivp():
    import numpy as np

    M = var("M", shape=(2, 2))
    A0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    sol = solve(Eq(M.d() + M, 0), ivp={M: A0}, tspan=(0, 1), rtol=1e-10, atol=1e-12)

    assert sol[M].shape == (2, 2, len(sol.t))
    np.testing.assert_allclose(sol.eval(M, 1.0), A0 * np.exp(-1), atol=1e-8)
    np.testing.assert_allclose(sol[M[1, 0]], sol[M][1, 0])


def test_matrix_initial_condition_shape_checked():
    import pytest

    M = var("M", shape=(2, 2))
    with pytest.raises(ValueError, match="has shape"):
        solve(Eq(M.d() + M, 0), ivp={M: [1, 2, 3, 4]}, tspan=(0, 1))


def test_matrix_index_validation():
    import pytest

    M = var("M", shape=(2, 3))
    assert M[1, 2] is M[1, 2]
    assert M[-1, -1] is M[1, 2]
    with pytest.raises(IndexError):
        M[0]
    with pytest.raises(IndexError):
        M[2, 0]


def test_mixed_shapes_in_one_equation_rejected():
    import pytest

    M = var("M", shape=(2, 2))
    u = var("u", shape=2)
    with pytest.raises(ValueError, match="same shape"):
        solve(Eq(M.d() + u, 0), ivp={M: [[1, 0], [0, 1]]}, tspan=(0, 1))
