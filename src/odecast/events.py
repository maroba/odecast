"""
Events: conditions that are located during integration
"""

from collections.abc import Callable

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef

from .equation import Eq
from .symbols import as_sympy, substitute_params


class Event:
    """
    A condition whose occurrences are located during integration.

    An event occurs when ``lhs - rhs`` of the condition changes sign. Its
    times are available as ``sol.t_events`` and the values of the solution
    there through ``sol.event_values``.

    Args:
        condition: An Eq, e.g. ``Eq(y, 0)``, or an expression that is zero
            at the event
        terminal: Stop the integration at the first occurrence
        direction: Only count zero crossings from negative to positive (1),
            from positive to negative (-1), or both (0)
        reset: New values for states at the event, e.g.
            ``{y.d(): -0.8 * y.d()}``. Right-hand sides are evaluated with
            the values just before the event; integration then continues
            from the new state. Only variables and derivatives below the
            highest order can be reset.

    Examples:
        >>> y = var("y")
        >>> hits_ground = Event(Eq(y, 0), terminal=True, direction=-1)
        >>> sol = solve(Eq(y.d(2), -9.81), ivp={y: 10, y.d(): 0},
        ...             tspan=(0, 10), events=[hits_ground])
        >>> sol.t_events[0]   # [1.428]

        A bouncing ball that loses 20% of its speed at each bounce:

        >>> bounce = Event(Eq(y, 0), direction=-1, reset={y.d(): -0.8 * y.d()})
    """

    def __init__(
        self,
        condition,
        *,
        terminal: bool = False,
        direction: int = 0,
        reset: dict | None = None,
    ):
        self.condition = condition
        self.terminal = terminal
        self.direction = direction
        self.reset = reset

    def __repr__(self):
        reset = f", reset={self.reset!r}" if self.reset else ""
        return (
            f"Event({self.condition!r}, terminal={self.terminal}, "
            f"direction={self.direction}{reset})"
        )

    def expr(self) -> sp.Expr:
        """The expression whose zeros are the event."""
        c = self.condition
        if isinstance(c, Eq):
            return as_sympy(c.lhs) - as_sympy(c.rhs)
        return as_sympy(c)


def compile_events(
    events, mapping, state_syms, t_symbol, params: dict, algebraic: dict | None = None
) -> list:
    """
    Convert events to functions g(t, x) for scipy.integrate.solve_ivp.

    Plain callables are passed through unchanged. Eq objects and expressions
    are wrapped in a non-terminal Event. ``algebraic`` maps algebraic
    variables z(t) of a DAE to their expressions in the states.
    """
    from .reduce import _substitute_derivatives_with_states

    if events is None:
        return None
    if not isinstance(events, (list, tuple)):
        events = [events]

    compiled = []
    for event in events:
        if callable(event) and not isinstance(event, Event):
            compiled.append(event)
            continue
        if not isinstance(event, Event):
            event = Event(event)

        def to_states(value, what):
            expr = as_sympy(value).xreplace(algebraic or {})
            expr = _substitute_derivatives_with_states(expr, mapping, state_syms)
            expr = substitute_params(expr, params)
            if expr.atoms(AppliedUndef, sp.Derivative):
                raise ValueError(
                    f"{what} may only involve variables, derivatives below their "
                    "highest order, and algebraic variables that can be "
                    "eliminated symbolically"
                )
            unknown = expr.free_symbols - {t_symbol} - set(state_syms)
            if unknown:
                names = ", ".join(sorted(s.name for s in unknown))
                raise ValueError(f"Missing values for parameters in event: {names}")
            return expr

        expr = to_states(event.expr(), f"Event {event.condition!r}")
        function = _event_function(expr, state_syms, t_symbol, event)
        if event.reset:
            function.reset = _reset_function(
                event, mapping, state_syms, t_symbol, to_states
            )
            function.terminal = True  # odecast restarts the integration
        compiled.append(function)
    return compiled


def _event_function(expr, state_syms, t_symbol, event: Event) -> Callable:
    g = sp.lambdify([t_symbol, *state_syms], expr, modules="numpy")

    def event_function(t, x):
        return float(g(t, *x))

    event_function.terminal = event.terminal
    event_function.direction = event.direction
    event_function.reset = None
    event_function.stops = event.terminal
    return event_function


def _reset_function(event, mapping, state_syms, t_symbol, to_states) -> Callable:
    """x_new = reset(t, x) for an event with a reset."""
    from .validate import normalize_ivp

    indices, exprs = [], []
    for (v, level), value in normalize_ivp(event.reset).items():
        if (v, level) not in mapping:
            what = v.name if level == 0 else f"{v.name}^({level})"
            raise ValueError(
                f"Cannot reset {what}: only variables and derivatives below the "
                "highest order are states"
            )
        indices.append(mapping[(v, level)])
        exprs.append(to_states(value, f"Reset value for {v.name}"))
    f = sp.lambdify([t_symbol, *state_syms], exprs, modules="numpy")

    def reset(t, x):
        new = np.array(x, dtype=float)
        new[indices] = np.array(f(t, *x), dtype=float)
        return new

    return reset
