import numpy as np
import pytest
import sympy as sp

from odecast import Eq, fit, solve, var


def test_fit_damped_oscillator_with_initial_value():
    y = var("y")
    c, k, y0 = sp.symbols("c k y0")
    eq = Eq(y.d(2) + c * y.d() + k * y, 0)
    true = {c: 0.4, k: 3.0, y0: 1.5}
    tt = np.linspace(0, 10, 80)
    clean = solve(
        eq, ivp={y: y0, y.d(): 0}, tspan=(0, 10), params=true, rtol=1e-10, atol=1e-12
    ).eval(y, tt)
    noisy = clean + np.random.default_rng(1).normal(0, 0.02, tt.size)

    result = fit(
        eq, {y: (tt, noisy)}, guess={c: 1.0, k: 2.0, y0: 1.0}, ivp={y: y0, y.d(): 0}
    )

    assert result.success
    for s, value in true.items():
        assert result.params[s] == pytest.approx(value, abs=4 * result.stderr[s])
    assert result.covariance.shape == (3, 3)
    assert result.solution.eval(y, 0.0) == pytest.approx(result.params[y0])


def test_fit_exact_data_recovers_parameters():
    y = var("y")
    k = sp.Symbol("k")
    tt = np.linspace(0, 2, 20)
    result = fit(
        Eq(y.d(), -k * y), {y: (tt, np.exp(-0.7 * tt))}, guess={k: 0.1}, ivp={y: 1.0}
    )
    assert result.params[k] == pytest.approx(0.7, rel=1e-6)


def test_fit_two_series_bounds_and_fixed_parameter():
    x, z = var("x"), var("z")
    a, b, c = sp.symbols("a b c")
    eqs = [Eq(x.d(), a * x - b * x * z), Eq(z.d(), -c * z + 0.075 * x * z)]
    ref = solve(
        eqs,
        ivp={x: 10, z: 5},
        tspan=(0, 15),
        params={a: 1.0, b: 0.1, c: 1.5},
        rtol=1e-10,
    )
    t1, t2 = np.linspace(0, 15, 40), np.linspace(0.3, 14.7, 25)
    data = {x: (t1, ref.eval(x, t1)), z: (t2, ref.eval(z, t2))}

    result = fit(
        eqs,
        data,
        guess={a: 0.7, b: 0.2},
        ivp={x: 10, z: 5},
        tspan=(0, 15),
        params={c: 1.5},
        bounds={a: (0, 5), b: (0, 1)},
        sigma={x: 0.1, z: 0.1},
    )

    assert result.params[a] == pytest.approx(1.0, rel=1e-4)
    assert result.params[b] == pytest.approx(0.1, rel=1e-4)


def test_jacobian_from_sensitivities_matches_finite_differences():

    x, z = var("x"), var("z")
    a, b, c, d = sp.symbols("a b c d")
    eqs = [Eq(x.d(), a * x - b * x * z), Eq(z.d(), -c * z + d * x * z)]
    true = {a: 1.0, b: 0.1, c: 1.5, d: 0.075}
    tt = np.linspace(0, 15, 60)
    ref = solve(eqs, ivp={x: 10, z: 5}, tspan=(0, 15), params=true, rtol=1e-10)
    data = {x: (tt, ref.eval(x, tt)), z: (tt, ref.eval(z, tt))}
    guess = {a: 0.9, b: 0.11, c: 1.4, d: 0.07}

    result = fit(eqs, data, guess=guess, ivp={x: 10, z: 5}, tspan=(0, 15))
    for s, value in true.items():
        assert result.params[s] == pytest.approx(value, rel=1e-6)
    # Exact derivatives need fewer evaluations than finite differences
    assert result.result.nfev < 15


def test_fit_survives_failing_trial_parameters():
    # From this start the optimizer tries parameters for which the
    # integration fails; such steps are rejected instead of aborting.
    x, z = var("x"), var("z")
    a, b, c, d = sp.symbols("a b c d")
    eqs = [Eq(x.d(), a * x - b * x * z), Eq(z.d(), -c * z + d * x * z)]
    ref = solve(
        eqs,
        ivp={x: 10, z: 5},
        tspan=(0, 15),
        params={a: 1.0, b: 0.1, c: 1.5, d: 0.075},
        rtol=1e-10,
    )
    tt = np.linspace(0, 15, 60)
    result = fit(
        eqs,
        {x: (tt, ref.eval(x, tt)), z: (tt, ref.eval(z, tt))},
        guess={a: 0.8, b: 0.12, c: 1.3, d: 0.06},
        ivp={x: 10, z: 5},
        tspan=(0, 15),
    )
    assert result.success
