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


class TestResets:
    G, E = 9.81, 0.8

    def _ball(self, **kwargs):
        y = var("y")
        event = Event(Eq(y, 0), reset={y.d(): -self.E * y.d()}, **kwargs)
        return y, Eq(y.d(2), -self.G), event

    def test_bouncing_ball_matches_exact_bounce_times(self):
        y, eq, bounce = self._ball(direction=-1)
        sol = solve(
            eq,
            ivp={y: 1.0, y.d(): 0.0},
            tspan=(0, 3),
            events=bounce,
            rtol=1e-10,
            atol=1e-12,
        )
        t1 = np.sqrt(2 / self.G)
        v1 = self.G * t1
        exact = [
            t1 + sum(2 * self.E**k * v1 / self.G for k in range(1, n + 1))
            for n in range(5)
        ]
        np.testing.assert_allclose(sol.t_events[0][:5], exact, rtol=1e-9)
        # Velocity before each bounce, and the jump in the interpolated solution
        assert sol.event_values(0, y.d())[0] == pytest.approx(-v1)
        assert sol.eval(y.d(), exact[0] + 1e-9) == pytest.approx(self.E * v1, rel=1e-6)
        assert sol.eval(y, np.linspace(0, 3, 3001)).min() > -1e-8

    def test_works_without_direction(self):
        y, eq, bounce = self._ball()
        sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 2), events=bounce)
        assert len(sol.t_events[0]) == 3

    def test_t_eval_across_resets(self):
        y, eq, bounce = self._ball(direction=-1)
        kwargs = {
            "ivp": {y: 1.0, y.d(): 0.0},
            "tspan": (0, 3),
            "events": bounce,
            "rtol": 1e-10,
            "atol": 1e-12,
        }
        sol = solve(eq, t_eval=np.linspace(0, 3, 7), **kwargs)
        ref = solve(eq, **kwargs)
        np.testing.assert_allclose(sol.t, np.linspace(0, 3, 7))
        np.testing.assert_allclose(sol[y], ref.eval(y, sol.t), atol=1e-12)

    def test_doses_at_fixed_times(self):
        from odecast import t

        c = var("c")
        doses = Event(Eq(sp.sin(sp.pi * t), 0), reset={c: c + 1})
        sol = solve(
            Eq(c.d(), -0.5 * c),
            ivp={c: 1.0},
            tspan=(0, 3.5),
            events=doses,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(sol.t_events[0], [1, 2, 3])
        expected = sum(np.exp(-0.5 * (3.5 - k)) for k in range(4))
        assert sol.eval(c, 3.5) == pytest.approx(expected, rel=1e-8)

    def test_terminal_event_stops_after_resets(self):
        y, eq, bounce = self._ball(direction=-1)
        stop = Event(Eq(y, 0.5), direction=1, terminal=True)
        sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 3), events=[bounce, stop])
        assert len(sol.t_events[0]) == 1
        assert sol.t[-1] == pytest.approx(sol.t_events[1][0])

    def test_accumulating_events_are_stopped(self):
        y, eq, bounce = self._ball(direction=-1)
        with pytest.raises(RuntimeError, match="accumulate"):
            solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 10), events=bounce)

    def test_only_states_can_be_reset(self):
        y = var("y")
        bad = Event(Eq(y, 0), reset={y.d(2): 0})
        with pytest.raises(ValueError, match="Cannot reset"):
            solve(Eq(y.d(2), -9.81), ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 1), events=bad)
