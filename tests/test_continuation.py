import numpy as np
import pytest
import sympy as sp

from odecast import Eq, continuation, var

r, mu = sp.symbols("r mu")


def test_saddle_node():
    # x' = r + x^2: equilibria x = -sqrt(-r) (stable) and +sqrt(-r) (unstable)
    x = var("x")
    branch = continuation(Eq(x.d(), r + x**2), r, {x: -1, r: -1}, (-1, 1))
    (fold,) = branch.bifurcations
    assert fold.kind == "fold"
    assert fold.param == pytest.approx(0, abs=1e-8)
    assert fold.point[x] == pytest.approx(0, abs=1e-6)
    # The branch turns back at the fold
    assert branch.param.max() == pytest.approx(0, abs=0.01)
    np.testing.assert_allclose(branch[x] ** 2, -branch.param, atol=1e-10)
    np.testing.assert_array_equal(branch.stable, branch[x] < 0)


def test_two_folds():
    x = var("x")
    branch = continuation(Eq(x.d(), r + x - x**3 / 3), r, {x: -2.5, r: -3}, (-3, 3))
    folds = sorted((b.param, b.point[x]) for b in branch.bifurcations)
    np.testing.assert_allclose(folds, [(-2 / 3, 1), (2 / 3, -1)], atol=1e-6)
    assert branch.param.min() >= -3 and branch.param.max() <= 3


@pytest.mark.parametrize(
    "rhs",
    [lambda x: r * x - x**2, lambda x: r * x - x**3],
    ids=["transcritical", "pitchfork"],
)
def test_branch_points(rhs):
    # x = 0 loses stability at r = 0, where another branch crosses
    x = var("x")
    branch = continuation(Eq(x.d(), rhs(x)), r, {x: 0, r: -1}, (-1, 1))
    (point,) = branch.bifurcations
    assert point.kind == "branch point"
    assert point.param == pytest.approx(0, abs=1e-8)
    assert branch.stable[0] and not branch.stable[-1]


def test_hopf():
    x, y = var("x"), var("y")
    eqs = [
        Eq(x.d(), mu * x - y - x * (x * x + y * y)),
        Eq(y.d(), x + mu * y - y * (x * x + y * y)),
    ]
    branch = continuation(eqs, mu, {x: 0, y: 0, mu: -1}, (-1, 1))
    (hopf,) = branch.bifurcations
    assert hopf.kind == "hopf"
    assert hopf.param == pytest.approx(0, abs=1e-8)
    np.testing.assert_allclose(sorted(hopf.eigenvalues.imag), [-1, 1], atol=1e-6)


def test_van_der_pol_hopf():
    v = var("v")
    branch = continuation(
        Eq(v.d(2) - mu * (1 - v**2) * v.d() + v, 0), mu, {v: 0, mu: -1}, (-1, 1)
    )
    assert [b.kind for b in branch.bifurcations] == ["hopf"]


def test_parameter_must_occur():
    x = var("x")
    with pytest.raises(ValueError, match="does not occur"):
        continuation(Eq(x.d(), -x), "r", {x: 0, "r": 0}, (0, 1))


def test_plot():
    plt = pytest.importorskip("matplotlib.pyplot")
    x = var("x")
    branch = continuation(Eq(x.d(), r + x - x**3 / 3), r, {x: -2.5, r: -3}, (-3, 3))
    _, ax = plt.subplots()
    branch.plot(x, ax=ax)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["fold"]
    assert ax.get_xlabel() == "r"
    plt.close("all")
