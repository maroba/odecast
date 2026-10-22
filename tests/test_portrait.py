import numpy as np
import pytest
import sympy as sp

from odecast import Eq, phase_portrait, var

plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _legend(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


def test_lotka_volterra():
    x, y = var("x"), var("y")
    _, ax = plt.subplots()
    phase_portrait(
        [Eq(x.d(), x - x * y), Eq(y.d(), x * y - y)],
        xlim=(-0.5, 3),
        ylim=(-0.5, 3),
        trajectories=[{x: 0.5, y: 0.5}],
        ax=ax,
    )
    assert _legend(ax) == ["x' = 0", "y' = 0", "saddle", "center"]
    assert ax.get_xlabel() == "x" and ax.get_ylabel() == "y"
    # One trajectory plus one marker per equilibrium
    lines = ax.get_lines()
    assert len(lines) == 3
    trajectory = lines[0]
    assert trajectory.get_xdata().min() > 0.3 and trajectory.get_ydata().max() < 2.3


def test_second_order_equation_with_selected_axes():
    theta = var("theta")
    _, ax = plt.subplots()
    phase_portrait(
        Eq(theta.d(2) + 0.3 * theta.d() + sp.sin(theta), 0),
        x=theta.d(),
        y=theta,
        xlim=(-3, 3),
        ylim=(-4, 7),
        ax=ax,
    )
    assert ax.get_xlabel() == "theta'" and ax.get_ylabel() == "theta"
    assert "stable" in _legend(ax) and "saddle" in _legend(ax)
    markers = {(line.get_xdata()[0], line.get_ydata()[0]) for line in ax.get_lines()}
    assert (0.0, 0.0) in markers
    assert any(abs(px) < 1e-9 and abs(py - np.pi) < 1e-9 for px, py in markers)


def test_requires_two_states():
    y = var("y")
    with pytest.raises(ValueError, match="two states"):
        phase_portrait(Eq(y.d(3) + y, 0), xlim=(-1, 1), ylim=(-1, 1))
