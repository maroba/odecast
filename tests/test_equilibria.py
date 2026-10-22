import numpy as np
import pytest
import sympy as sp

from odecast import Eq, equilibria, var


def test_lotka_volterra():
    x, y = var("x"), var("y")
    results = equilibria([Eq(x.d(), x - x * y), Eq(y.d(), x * y - y)])
    assert [(e.point[x], e.point[y], e.stability) for e in results] == [
        (0, 0, "saddle"),
        (1, 1, "center"),
    ]


def test_damped_pendulum():
    theta = var("theta")
    results = equilibria(Eq(theta.d(2) + 0.3 * theta.d() + sp.sin(theta), 0))
    found = sorted((float(e.point[theta]), e.stability) for e in results)
    assert found == [(0.0, "stable focus"), (pytest.approx(np.pi), "saddle")]


@pytest.mark.parametrize(
    ("damping", "stability"),
    [(3, "stable node"), (0.5, "stable focus"), (-0.5, "unstable focus")],
)
def test_linear_oscillator(damping, stability):
    y = var("y")
    (result,) = equilibria(Eq(y.d(2) + damping * y.d() + y, 0))
    assert result.stability == stability


def test_one_dimensional():
    x = var("x")
    results = equilibria(Eq(x.d(), x * (1 - x)))
    assert [(e.point[x], e.stability) for e in results] == [
        (0, "unstable"),
        (1, "stable"),
    ]


def test_parameters():
    y = var("y")
    k = sp.Symbol("k", positive=True)
    (symbolic,) = equilibria(Eq(y.d(2) + k * y, 0))
    assert symbolic.stability == "unknown"
    assert set(symbolic.eigenvalues) == {sp.I * sp.sqrt(k), -sp.I * sp.sqrt(k)}

    (numeric,) = equilibria(Eq(y.d(2) + 0.1 * y.d() + k * y, 0), params={k: 4})
    assert numeric.stability == "stable focus"


def test_complex_equilibria_dropped():
    x = var("x")
    assert equilibria(Eq(x.d(), x * x + 1)) == []


def test_non_autonomous_rejected():
    from odecast import t

    y = var("y")
    with pytest.raises(ValueError, match="autonomous"):
        equilibria(Eq(y.d(), sp.sin(t)))
