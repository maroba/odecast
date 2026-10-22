import numpy as np
import pytest
import sympy as sp

from odecast import Eq, solve_sde, var

MU, SIGMA = 0.1, 0.3


def gbm(**kwargs):
    """Geometric Brownian motion dY = mu Y dt + sigma Y dW."""
    y = var("y")
    sol = solve_sde(
        Eq(y.d(), MU * y), noise={y: SIGMA * y}, ivp={y: 1.0}, tspan=(0, 1), **kwargs
    )
    exact = np.exp((MU - SIGMA**2 / 2) * sol.t + SIGMA * sol.W[0])
    return y, sol, exact


@pytest.mark.parametrize(("method", "order"), [("euler", 0.5), ("milstein", 1.0)])
def test_strong_convergence_order(method, order):
    errors = []
    for dt in (1e-2, 1e-3):
        y, sol, exact = gbm(dt=dt, paths=500, method=method, seed=2)
        errors.append(np.abs(sol[y][:, -1] - exact[:, -1]).mean())
    assert np.log10(errors[0] / errors[1]) == pytest.approx(order, abs=0.15)


def test_mean_and_variance():
    y, sol, _ = gbm(dt=1e-2, paths=20000, seed=1)
    mean = np.exp(MU)
    var_ = np.exp(2 * MU) * (np.exp(SIGMA**2) - 1)
    assert sol.mean(y)[-1] == pytest.approx(mean, abs=4 * np.sqrt(var_ / 20000))
    assert sol.std(y)[-1] ** 2 == pytest.approx(var_, rel=0.05)
    assert sol[y].shape == (20000, len(sol.t))


def test_ornstein_uhlenbeck_stationary_variance():
    y = var("y")
    theta, sigma = 2.0, 0.5
    sol = solve_sde(
        Eq(y.d(), -theta * y),
        noise={y: sigma},
        ivp={y: 0.0},
        tspan=(0, 5),
        dt=5e-3,
        paths=4000,
        seed=3,
    )
    assert sol.std(y)[-1] ** 2 == pytest.approx(sigma**2 / (2 * theta), rel=0.06)


def test_langevin_second_order():
    # x'' + gamma x' + omega^2 x = sigma xi: stationary Var(x) = sigma^2 / (2 gamma omega^2)
    x = var("x")
    gamma, omega, sigma = 0.5, 2.0, 0.4
    sol = solve_sde(
        Eq(x.d(2) + gamma * x.d() + omega**2 * x, 0),
        noise={x.d(): sigma},
        ivp={x: 0, x.d(): 0},
        tspan=(0, 30),
        dt=5e-3,
        paths=2000,
        seed=4,
    )
    assert sol[x][:, -1].var() == pytest.approx(
        sigma**2 / (2 * gamma * omega**2), rel=0.1
    )


def test_reproducible_with_seed():
    y = var("y")
    kwargs = {
        "noise": {y: 0.1},
        "ivp": {y: 0.0},
        "tspan": (0, 1),
        "dt": 0.01,
        "paths": 3,
    }
    a = solve_sde(Eq(y.d(), -y), seed=7, **kwargs)
    b = solve_sde(Eq(y.d(), -y), seed=7, **kwargs)
    np.testing.assert_array_equal(a[y], b[y])


def test_vector_variable_and_parameters():
    u = var("u", shape=2)
    k = sp.Symbol("k")
    sol = solve_sde(
        Eq(u.d(), -k * u),
        noise={u: [0.1, 0.2]},
        ivp={u: [1, 1]},
        tspan=(0, 1),
        dt=0.01,
        paths=3,
        params={k: 1},
        seed=0,
    )
    assert sol[u].shape == (2, 3, 101)
    assert sol.quantile(u[0], [0.1, 0.9]).shape == (2, 101)


def test_noise_must_enter_highest_derivative():
    x = var("x")
    with pytest.raises(ValueError, match="highest derivative"):
        solve_sde(
            Eq(x.d(2) + x, 0),
            noise={x: 0.4},
            ivp={x: 0, x.d(): 0},
            tspan=(0, 1),
            dt=0.1,
        )


def test_unknown_method():
    y = var("y")
    with pytest.raises(ValueError, match="Unknown method"):
        solve_sde(
            Eq(y.d(), -y), noise={y: 1}, ivp={y: 0}, tspan=(0, 1), dt=0.1, method="rk4"
        )


def test_plot():
    plt = pytest.importorskip("matplotlib.pyplot")
    y = var("y")
    sol = solve_sde(
        Eq(y.d(), -y),
        noise={y: 0.3},
        ivp={y: 1.0},
        tspan=(0, 1),
        dt=0.01,
        paths=50,
        seed=0,
    )
    _, ax = plt.subplots()
    sol.plot(y, paths=10, ax=ax)
    assert len(ax.get_lines()) == 11
    plt.close("all")
