import numpy as np
import pytest
import sympy as sp

from odecast import Eq, equilibria, linearize, var

s = sp.Symbol("s")


def test_mass_spring_damper_symbolic():
    x = var("x")
    F = sp.Symbol("F")
    m, c, k = sp.symbols("m c k", positive=True)
    lin = linearize(
        Eq(m * x.d(2) + c * x.d() + k * x, F), around={x: 0}, inputs=[F], outputs=[x]
    )
    assert lin.A == sp.Matrix([[0, 1], [-k / m, -c / m]])
    assert lin.B == sp.Matrix([[0], [1 / m]])
    assert sp.simplify(lin.transfer_function()[0] - 1 / (m * s**2 + c * s + k)) == 0
    assert lin.states == ["x", "x'"]


def test_numeric_transfer_function_and_scipy():
    x = var("x")
    F = sp.Symbol("F")
    lin = linearize(
        Eq(2 * x.d(2) + 0.5 * x.d() + 8 * x, F),
        around={x: 0},
        inputs=[F],
        outputs=[x, x.d()],
    )
    G = lin.transfer_function()
    assert sp.simplify(G[0] - 0.5 / (s**2 + 0.25 * s + 4)) == 0
    assert sp.simplify(G[1] - 0.5 * s / (s**2 + 0.25 * s + 4)) == 0

    ss = lin.to_scipy()
    np.testing.assert_allclose(ss.A, [[0, 1], [-4, -0.25]])
    np.testing.assert_allclose(ss.C, np.eye(2))


def test_pendulum_upright_is_unstable():
    theta = var("theta")
    F = sp.Symbol("F")
    g, L = sp.symbols("g L", positive=True)
    eq = Eq(theta.d(2) + g / L * sp.sin(theta), F)
    bottom = linearize(eq, around={theta: 0}, inputs=[F])
    top = linearize(eq, around={theta: sp.pi}, inputs=[F])
    assert bottom.A == sp.Matrix([[0, 1], [-g / L, 0]])
    assert set(top.eigenvalues()) == {sp.sqrt(g / L), -sp.sqrt(g / L)}


def test_operating_point_with_input():
    # x'' + x^3 = F has the equilibrium x = 1 for F = 1
    x = var("x")
    F = sp.Symbol("F")
    lin = linearize(Eq(x.d(2) + x**3, F), around={x: 1, F: 1}, inputs=[F])
    assert lin.A == sp.Matrix([[0, 1], [-3, 0]])


def test_around_equilibria():
    theta = var("theta")
    eq = Eq(theta.d(2) + 0.3 * theta.d() + sp.sin(theta), 0)

    def ordered(values):
        values = np.array([complex(v) for v in values])
        return values[np.lexsort((values.imag, np.round(values.real, 9)))]

    for e in equilibria(eq):
        np.testing.assert_allclose(
            ordered(linearize(eq, around=e).eigenvalues()),
            ordered(e.eigenvalues),
            atol=1e-9,
        )


def test_not_an_equilibrium():
    theta = var("theta")
    with pytest.raises(ValueError, match="not an equilibrium"):
        linearize(Eq(theta.d(2) + sp.sin(theta), 0), around={theta: 1.0})


def test_symbols_prevent_scipy_conversion():
    x = var("x")
    k = sp.Symbol("k")
    with pytest.raises(ValueError, match="give all parameters a value"):
        linearize(Eq(x.d(2) + k * x, 0), around={x: 0}).to_scipy()
