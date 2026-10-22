import numpy as np
import pytest

from odecast import Eq, solve, var


@pytest.fixture
def oscillator():
    u = var("u", shape=2)
    sol = solve(Eq(u.d(2) + u, 0), ivp={u: [1, 0], u.d(): [0, 1]}, tspan=(0, 4))
    return u, sol


def test_plot(oscillator):
    plt = pytest.importorskip("matplotlib.pyplot")
    u, sol = oscillator
    _, ax = plt.subplots()
    sol.plot(ax=ax)
    assert [line.get_label() for line in ax.get_lines()] == ["u[0]", "u[1]"]

    _, ax = plt.subplots()
    sol.plot(u.d(), ax=ax, n=50)
    assert [line.get_label() for line in ax.get_lines()] == ["u'[0]", "u'[1]"]
    assert len(ax.get_lines()[0].get_xdata()) == 50
    plt.close("all")


def test_to_pandas(oscillator):
    pytest.importorskip("pandas")
    u, sol = oscillator
    df = sol.to_pandas(t=[0.0, 1.0])
    assert list(df.columns) == ["u[0]", "u[1]"]
    assert df.index.name == "t"
    np.testing.assert_allclose(
        df.loc[1.0].to_numpy(), [np.cos(1), np.sin(1)], atol=1e-2
    )

    df = sol.to_pandas(u.d()[0])
    assert list(df.columns) == ["u[0]'"]
    np.testing.assert_array_equal(df.index, sol.t)
