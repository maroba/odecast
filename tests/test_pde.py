import numpy as np
import pytest
import sympy as sp

from odecast import Eq, t

pytest.importorskip("findiff")

from odecast.pde import (
    Dirichlet,
    Grid,
    Neumann,
    Periodic,
    field,
    solve_pde,
)

TOL = {"rtol": 1e-8, "atol": 1e-10}
X = np.linspace(0, 1, 101)
ZERO = (Dirichlet(0), Dirichlet(0))


@pytest.fixture
def u():
    return field("u", Grid(x=X))


@pytest.mark.parametrize(("acc", "error"), [(2, 1e-4), (4, 1e-8)])
def test_heat_equation(u, acc, error):
    D = 0.1
    sol = solve_pde(
        Eq(u.d(), D * u.dx(2)),
        ic={u: lambda x: np.sin(np.pi * x)},
        bc={u: {"x": ZERO}},
        tspan=(0, 1),
        acc=acc,
        **TOL,
    )
    exact = np.exp(-D * np.pi**2) * np.sin(np.pi * X)
    np.testing.assert_allclose(sol.eval(u, 1.0), exact, atol=error)
    assert sol[u].shape == (101, len(sol.t))


def test_neumann_boundaries(u):
    sol = solve_pde(
        Eq(u.d(), u.dx(2)),
        ic={u: lambda x: np.cos(np.pi * x)},
        bc={u: {"x": (Neumann(0), Neumann(0))}},
        tspan=(0, 0.1),
        acc=4,
        **TOL,
    )
    exact = np.exp(-(np.pi**2) * 0.1) * np.cos(np.pi * X)
    np.testing.assert_allclose(sol.eval(u, 0.1), exact, atol=1e-6)


def test_periodic_advection_diffusion():
    x = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    v = field("v", Grid(x=x))
    sol = solve_pde(
        Eq(v.d(), -v.dx() + 0.05 * v.dx(2)),
        ic={v: np.sin},
        bc={v: {"x": Periodic()}},
        tspan=(0, 1),
        acc=4,
        **TOL,
    )
    np.testing.assert_allclose(
        sol.eval(v, 1.0), np.exp(-0.05) * np.sin(x - 1), atol=1e-6
    )


def test_wave_equation(u):
    sol = solve_pde(
        Eq(u.d(2), u.dx(2)),
        ic={u: lambda x: np.sin(np.pi * x), u.d(): 0},
        bc={u: {"x": ZERO}},
        tspan=(0, 10),
        **TOL,
    )
    np.testing.assert_allclose(
        sol.eval(u, 10.0), np.cos(10 * np.pi) * np.sin(np.pi * X), atol=1e-5
    )


def test_high_order_stencils_warn_for_waves(u):
    with pytest.warns(UserWarning, match="unstable"):
        solve_pde(
            Eq(u.d(2), u.dx(2)),
            ic={u: 0, u.d(): 0},
            bc={u: {"x": ZERO}},
            tspan=(0, 0.01),
            acc=4,
        )


def test_time_dependent_boundary_values(u):
    # u = x^2 + 2t solves u_t = u_xx
    bc = {u: {"x": (Dirichlet(lambda t: 2 * t), Dirichlet(lambda t: 1 + 2 * t))}}
    sol = solve_pde(
        Eq(u.d(), u.dx(2)), ic={u: lambda x: x**2}, bc=bc, tspan=(0, 0.5), **TOL
    )
    np.testing.assert_allclose(sol.eval(u, 0.5), X**2 + 1, atol=1e-10)


def test_source_in_space_and_time(u):
    # u = exp(t) sin(pi x)
    grid = u.grid
    source = (1 + sp.pi**2) * sp.sin(sp.pi * grid.x) * sp.exp(t)
    sol = solve_pde(
        Eq(u.d(), u.dx(2) + source),
        ic={u: lambda x: np.sin(np.pi * x)},
        bc={u: {"x": ZERO}},
        tspan=(0, 1),
        acc=4,
        **TOL,
    )
    np.testing.assert_allclose(sol.eval(u, 1.0), np.e * np.sin(np.pi * X), atol=1e-6)


def test_two_dimensions():
    xs = np.linspace(0, 1, 41)
    grid = Grid(x=xs, y=xs)
    q = field("q", grid)
    sol = solve_pde(
        Eq(q.d(), q.laplace()),
        ic={q: lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)},
        bc={q: {"x": ZERO, "y": ZERO}},
        tspan=(0, 0.05),
        acc=4,
        **TOL,
    )
    X2, Y2 = grid.mesh
    exact = np.exp(-2 * np.pi**2 * 0.05) * np.sin(np.pi * X2) * np.sin(np.pi * Y2)
    np.testing.assert_allclose(sol.eval(q, 0.05), exact, atol=1e-6)


def test_coupled_fields_with_parameters():
    grid = Grid(x=X)
    a, b = field("a", grid), field("b", grid)
    k = sp.Symbol("k")
    sol = solve_pde(
        [Eq(a.d(), a.dx(2) - k * b), Eq(b.d(), b.dx(2) + k * a)],
        ic={a: lambda x: np.sin(np.pi * x), b: 0},
        bc={a: {"x": ZERO}, b: {"x": ZERO}},
        tspan=(0, 0.2),
        params={k: 1},
        acc=4,
        **TOL,
    )
    decay = np.exp(-(np.pi**2) * 0.2)
    np.testing.assert_allclose(
        sol.eval(a, 0.2), decay * np.cos(0.2) * np.sin(np.pi * X), atol=1e-6
    )
    np.testing.assert_allclose(
        sol.eval(b, 0.2), decay * np.sin(0.2) * np.sin(np.pi * X), atol=1e-6
    )


def test_fisher_kpp_front_speed():
    # Fronts of u_t = u_xx + u (1 - u) travel at speed close to 2
    x = np.linspace(0, 100, 501)
    u = field("u", Grid(x=x))
    sol = solve_pde(
        Eq(u.d(), u.dx(2) + u * (1 - u)),
        ic={u: lambda x: np.where(x < 5, 1.0, 0.0)},
        bc={u: {"x": (Neumann(0), Neumann(0))}},
        tspan=(0, 40),
    )

    def front(time):
        profile = sol.eval(u, time)
        return np.interp(0.5, profile[::-1], x[::-1])

    assert (front(40) - front(20)) / 20 == pytest.approx(2, abs=0.1)


def test_errors(u):
    with pytest.raises(ValueError, match="Missing boundary conditions"):
        solve_pde(Eq(u.d(), u.dx(2)), ic={u: 0}, bc={}, tspan=(0, 1))
    with pytest.raises(ValueError, match="second order in time"):
        solve_pde(Eq(u.d(2), u.dx(2)), ic={u: 0}, bc={u: {"x": ZERO}}, tspan=(0, 1))
    with pytest.raises(ValueError, match="Missing values for parameters: k"):
        solve_pde(
            Eq(u.d(), sp.Symbol("k") * u.dx(2)),
            ic={u: 0},
            bc={u: {"x": ZERO}},
            tspan=(0, 1),
        )


def test_plot(u):
    plt = pytest.importorskip("matplotlib.pyplot")
    sol = solve_pde(
        Eq(u.d(), u.dx(2)),
        ic={u: lambda x: np.sin(np.pi * x)},
        bc={u: {"x": ZERO}},
        tspan=(0, 0.1),
    )
    _, ax = plt.subplots()
    sol.plot(u, ax=ax)
    assert len(ax.get_lines()) == 5
    plt.close("all")
