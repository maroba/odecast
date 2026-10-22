"""
Events: conditions that are located during integration
"""

from collections.abc import Callable

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

    Examples:
        >>> y = var("y")
        >>> hits_ground = Event(Eq(y, 0), terminal=True, direction=-1)
        >>> sol = solve(Eq(y.d(2), -9.81), ivp={y: 10, y.d(): 0},
        ...             tspan=(0, 10), events=[hits_ground])
        >>> sol.t_events[0]   # [1.428]
    """

    def __init__(self, condition, *, terminal: bool = False, direction: int = 0):
        self.condition = condition
        self.terminal = terminal
        self.direction = direction

    def __repr__(self):
        return (
            f"Event({self.condition!r}, terminal={self.terminal}, "
            f"direction={self.direction})"
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

        expr = event.expr().xreplace(algebraic or {})
        expr = _substitute_derivatives_with_states(expr, mapping, state_syms)
        expr = substitute_params(expr, params)
        if expr.atoms(AppliedUndef, sp.Derivative):
            raise ValueError(
                f"Event {event.condition!r} may only involve variables, "
                "derivatives below their highest order, and algebraic variables "
                "that can be eliminated symbolically"
            )
        unknown = expr.free_symbols - {t_symbol} - set(state_syms)
        if unknown:
            names = ", ".join(sorted(s.name for s in unknown))
            raise ValueError(f"Missing values for parameters in event: {names}")

        compiled.append(_event_function(expr, state_syms, t_symbol, event))
    return compiled


def _event_function(expr, state_syms, t_symbol, event: Event) -> Callable:
    g = sp.lambdify([t_symbol, *state_syms], expr, modules="numpy")

    def event_function(t, x):
        return float(g(t, *x))

    event_function.terminal = event.terminal
    event_function.direction = event.direction
    return event_function
