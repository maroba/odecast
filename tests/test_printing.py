import numpy as np
import sympy as sp

from odecast import Eq, latex, solve, t, var


def test_derivatives_as_primes():
    y = var("y")
    assert latex(Eq(y.d(2) + 0.3 * y.d() + y, sp.sin(t))) == (
        r"y + 0.3 y' + y'' = \sin{\left(t \right)}"
    )


def test_high_order_greek_and_components():
    theta = var("theta")
    assert r"\theta^{(4)}" in latex(theta.d(4) + theta)
    u, M = var("u", shape=2), var("M", shape=(2, 2))
    assert latex(u[0].d(2) + M[1, 0]) == r"M_{1,0} + u_{0}''"


def test_repr_latex():
    y = var("y")
    assert Eq(y.d() + y, 0)._repr_latex_() == r"$\displaystyle y + y' = 0$"
    assert y._repr_latex_() == r"$\displaystyle y$"


def test_vector_equation_rendered_per_component():
    u = var("u", shape=2)
    out = Eq(u.d(2) + np.array([[2, -1], [-1, 2]]) @ u, 0)._repr_latex_()
    assert r"\begin{aligned}" in out and out.count("&=") == 2


def test_symbolic_solution():
    y = var("y")
    sol = solve(Eq(y.d(2) + y, 0), backend="sympy", ivp={y: 1, y.d(): 0})
    assert sol._repr_latex_() == (
        r"$\displaystyle \begin{aligned}y(t) &= \cos{\left(t \right)}\end{aligned}$"
    )
