import numpy as np
import pytest

from odecast import Eq, periodic_orbit, var

MU, OMEGA = 0.5, 2.0


def hopf(sign=1):
    """Hopf normal form: a limit cycle of radius sqrt(mu) and period 2 pi / omega."""
    x, y = var("x"), var("y")
    r2 = x * x + y * y
    eqs = [
        Eq(x.d(), sign * (MU * x - OMEGA * y - x * r2)),
        Eq(y.d(), sign * (OMEGA * x + MU * y - y * r2)),
    ]
    return eqs, x, y


def test_hopf_normal_form():
    eqs, x, y = hopf()
    orbit = periodic_orbit(eqs, ivp={x: 0.1, y: 0}, period=3.0, transient=20)
    assert orbit.period == pytest.approx(2 * np.pi / OMEGA, rel=1e-7)

    tt = np.linspace(0, orbit.period, 50)
    radius = np.hypot(orbit.solution.eval(x, tt), orbit.solution.eval(y, tt))
    np.testing.assert_allclose(radius, np.sqrt(MU), atol=1e-7)

    # Floquet multipliers: 1 and exp(-2 mu T)
    expected = np.exp(-2 * MU * orbit.period)
    np.testing.assert_allclose(
        np.sort(np.abs(orbit.multipliers)), [expected, 1.0], rtol=1e-6
    )
    assert orbit.stability == "stable"


def test_van_der_pol():
    v = var("v")
    orbit = periodic_orbit(
        Eq(v.d(2) - (1 - v**2) * v.d() + v, 0),
        ivp={v: 2, v.d(): 0},
        period=6.5,
        transient=20,
    )
    assert orbit.period == pytest.approx(6.6632868593, rel=1e-6)
    assert orbit.stability == "stable"
    # The solution is periodic
    assert orbit.solution.eval(v, 0.0) == pytest.approx(
        orbit.solution.eval(v, orbit.period), abs=1e-6
    )


def test_unstable_orbit():
    eqs, x, y = hopf(sign=-1)
    orbit = periodic_orbit(eqs, ivp={x: np.sqrt(MU), y: 0}, period=3.0)
    assert orbit.stability == "unstable"


def test_non_autonomous_rejected():
    from odecast import t

    y = var("y")
    with pytest.raises(ValueError, match="autonomous"):
        periodic_orbit(Eq(y.d(2) + y, np.cos(1) * t), ivp={y: 1, y.d(): 0}, period=6)
