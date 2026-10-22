import numpy as np
import pytest
import sympy as sp

from odecast import BC, Eq, Event, solve, var


def test_terminal_event():
    y = var("y")
    sol = solve(
        Eq(y.d(2), -9.81),
        ivp={y: 10.0, y.d(): 0.0},
        tspan=(0, 10),
        events=[Event(Eq(y, 0), terminal=True, direction=-1)],
    )
    t_hit = np.sqrt(20 / 9.81)
    assert sol.t_events[0] == pytest.approx([t_hit])
    assert sol.t[-1] == pytest.approx(t_hit)
    assert sol.event_values(0, y.d()) == pytest.approx([-9.81 * t_hit])


def test_plain_equation_as_event():
    y = var("y")
    sol = solve(
        Eq(y.d(2) + y, 0),
        ivp={y: 1, y.d(): 0},
        tspan=(0, 10),
        events=Eq(y, 0),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sol.t_events[0], [np.pi / 2, 3 * np.pi / 2, 5 * np.pi / 2]
    )


def test_event_direction_and_parameter():
    y = var("y")
    h = sp.Symbol("h")
    sol = solve(
        Eq(y.d(2) + y, 0),
        ivp={y: 1, y.d(): 0},
        tspan=(0, 10),
        params={h: 0.5},
        events=Event(y - h, direction=1),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(sol.t_events[0], [2 * np.pi - np.pi / 3])


def test_event_on_vector_component():
    u = var("u", shape=2)
    sol = solve(
        Eq(u.d(2) + u, 0),
        ivp={u: [1, 0], u.d(): [0, 1]},
        tspan=(0, 4),
        events=Eq(u[1], 0.5),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(sol.t_events[0], [np.pi / 6, 5 * np.pi / 6])


def test_callable_event_passed_through():
    y = var("y")

    def crossing(t, x):
        return x[0]

    sol = solve(Eq(y.d() + 1, 0), ivp={y: 1}, tspan=(0, 2), events=crossing)
    assert sol.t_events[0] == pytest.approx([1.0])


def test_event_on_highest_derivative_rejected():
    y = var("y")
    with pytest.raises(ValueError, match="below their highest order"):
        solve(
            Eq(y.d(2) + y, 0), ivp={y: 1, y.d(): 0}, tspan=(0, 1), events=Eq(y.d(2), 0)
        )


def test_events_not_supported_for_bvp():
    y = var("y")
    with pytest.raises(ValueError, match="initial value problems"):
        solve(
            Eq(y.d(2) + y, 0),
            bvp=[BC(y, t=0, value=0), BC(y, t=1, value=1)],
            tspan=(0, 1),
            events=Eq(y, 0.5),
        )
