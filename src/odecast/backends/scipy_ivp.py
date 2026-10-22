"""
SciPy-based backend for solving initial value problems
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.integrate import OdeSolution, solve_ivp

from ..solution import SolutionIVP
from ..symbols import Derivative, Variable

# Limit on restarts after event resets, to stop accumulating events
MAX_RESETS = 10000


class ScipyIVPBackend:
    """
    Backend for solving IVPs using SciPy's solve_ivp.
    """

    def solve(
        self,
        f_compiled: Callable,
        jac_compiled: Callable | None,
        x0: np.ndarray,
        t0: float,
        tspan: tuple[float, float],
        mapping: dict[Variable | tuple[Variable, int], list | int],
        options: dict[str, Any] | None = None,
        after_reset: Callable | None = None,
    ) -> SolutionIVP:
        """
        Solve the IVP using SciPy's solve_ivp.

        Args:
            f_compiled: Compiled RHS function f(t, x) -> dx/dt
            jac_compiled: Compiled Jacobian function (optional)
            x0: Initial state vector
            t0: Initial time
            tspan: Tuple of (t_start, t_end)
            mapping: State mapping from build_state_map
            options: Keyword arguments for scipy.integrate.solve_ivp (method,
                rtol, atol, max_step, t_eval, first_step, events, ...)
            after_reset: Called as after_reset(t, x) after an event has reset
                the state, e.g. to check the new state

        Events with a ``reset`` attribute end the integration; the state is
        reset and the integration restarts until the end of tspan.

        Returns:
            SolutionIVP object with results
        """
        options = dict(options or {})
        options.setdefault("rtol", 1e-3)
        options.setdefault("atol", 1e-6)
        # Dense output gives eval() an interpolant of the solver's own order
        options["dense_output"] = True
        if jac_compiled is not None:
            options.setdefault("jac", jac_compiled)
        events = options.get("events")
        n_events = (
            len(events) if isinstance(events, (list, tuple)) else (1 if events else 0)
        )
        event_list = list(events) if isinstance(events, (list, tuple)) else [events]

        t_end = tspan[1]
        t_start, x_start = tspan[0], np.asarray(x0, dtype=float)
        times, states, segments = [], [], []
        t_events = [[] for _ in range(n_events)]
        y_events = [[] for _ in range(n_events)]

        t_eval = options.pop("t_eval", None)
        fired = []
        for _ in range(MAX_RESETS + 1):
            if t_eval is not None:
                t_eval = np.asarray(t_eval, dtype=float)
                options["t_eval"] = t_eval[(t_eval >= t_start) & (t_eval <= t_end)]
            if n_events:
                options["events"] = [
                    _ignore_zero_at_start(
                        e,
                        f_compiled,
                        t_start,
                        x_start,
                        t_end,
                        any(e is f for f in fired),
                    )
                    for e in event_list
                ]
            sol = solve_ivp(
                fun=f_compiled, t_span=(t_start, t_end), y0=x_start, **options
            )
            if not sol.success:
                reason = getattr(f_compiled, "failure", None)
                detail = f" Last problem: {reason}" if reason else ""
                raise RuntimeError(f"SciPy solve_ivp failed: {sol.message}{detail}")

            # Consecutive segments share the time of the reset
            seg_t = np.asarray(sol.t, dtype=float)
            seg_y = np.asarray(sol.y, dtype=float).reshape(len(x_start), -1)
            last = next((ts[-1] for ts in reversed(times) if len(ts)), None)
            skip = 1 if len(seg_t) and seg_t[0] == last else 0
            times.append(seg_t[skip:])
            states.append(seg_y[:, skip:])
            segments.append(sol.sol)
            for i in range(n_events):
                t_events[i].extend(sol.t_events[i])
                y_events[i].extend(sol.y_events[i])

            if sol.status != 1:  # reached the end of tspan
                break

            # A terminal event occurred: stop, or apply resets and restart.
            # The integration stopped at the latest event.
            t_hit = max(te[-1] for te in sol.t_events if len(te))
            fired, x_hit = [], None
            for e, te, ye in zip(event_list, sol.t_events, sol.y_events, strict=True):
                if len(te) and te[-1] == t_hit:
                    fired.append(e)
                    x_hit = ye[-1]
            resets = [e.reset for e in fired if getattr(e, "reset", None) is not None]
            if not resets or any(getattr(e, "stops", True) for e in fired):
                break
            if t_hit == t_start:
                raise RuntimeError(
                    f"Event at t={t_hit:g} occurred again immediately after a reset. "
                    "Either the reset state triggers it again (set the event's "
                    "direction), or the events accumulate, e.g. a bouncing ball "
                    "coming to rest (shorten tspan or add a terminal event)."
                )
            x_start = x_hit
            for reset in resets:
                x_start = reset(t_hit, x_start)
            if after_reset is not None:
                after_reset(t_hit, x_start)
            t_start = t_hit
        else:
            raise RuntimeError(
                f"More than {MAX_RESETS} resets; the events may accumulate "
                "(e.g. a bouncing ball coming to rest). Shorten tspan or add a "
                "terminal event."
            )

        if len(segments) == 1:
            interpolant = segments[0]
        else:
            interpolant = OdeSolution(
                np.concatenate([segments[0].ts] + [s.ts[1:] for s in segments[1:]]),
                [i for s in segments for i in s.interpolants],
            )

        return SolutionIVP(
            t=np.concatenate(times),
            y=np.concatenate(states, axis=1),
            mapping=mapping,
            t0=t0,
            f_compiled=f_compiled,
            jac_compiled=jac_compiled,
            x0=x0,
            interpolant=interpolant,
            t_events=[np.array(te) for te in t_events] if n_events else sol.t_events,
            y_events=(
                [np.array(ye).reshape(-1, len(x0)) for ye in y_events]
                if n_events
                else sol.y_events
            ),
        )


def _ignore_zero_at_start(event, f, t_start, x_start, t_end, just_fired=False):
    """
    Do not let an event fire at the start of a segment.

    SciPy counts a change from zero to nonzero as a crossing. An event that is
    zero at the start (e.g. Eq(t, 1) right after its own reset at t=1) would
    therefore fire again immediately. For events that are exactly zero at the
    start, or that have just fired (their value is then zero up to rounding),
    the value at the start is replaced by a tiny number with the sign the
    event has just afterwards.
    """
    if not just_fired and event(t_start, x_start) != 0:
        return event
    h = 1e-8 * max(1.0, abs(t_start)) * np.sign(t_end - t_start)
    x_next = np.asarray(x_start, dtype=float) + h * np.asarray(f(t_start, x_start))
    sign = np.sign(event(t_start + h, x_next)) or 1.0

    def shifted(t, x):
        return sign * 1e-300 if t == t_start else event(t, x)

    for attribute in ("terminal", "direction", "reset", "stops"):
        if hasattr(event, attribute):
            setattr(shifted, attribute, getattr(event, attribute))
    return shifted


def convert_ivp_to_state_vector(
    ivp_dict: dict[Variable | Derivative, float],
    mapping: dict[Variable | tuple[Variable, int], list | int],
) -> np.ndarray:
    """
    Convert IVP dictionary to initial state vector.

    Args:
        ivp_dict: Dictionary mapping variables/derivatives to initial values
        mapping: State mapping from build_state_map

    Returns:
        Initial state vector x0
    """
    # Import here to avoid circular import
    from ..validate import normalize_ivp

    # Normalize the IVP dictionary to handle vector variables
    normalized_ivp = normalize_ivp(ivp_dict)

    # Determine the size of the state vector
    max_index = 0
    for key, value in mapping.items():
        if isinstance(key, tuple):  # (var, level) keys
            max_index = max(max_index, value)
        elif isinstance(key, Variable) and isinstance(value, list) and value:
            max_index = max(max_index, max(value))

    n_states = max_index + 1
    x0 = np.zeros(n_states)

    # Fill in the initial conditions using normalized format
    for (var, level), value in normalized_ivp.items():
        if (var, level) in mapping:
            state_index = mapping[(var, level)]
            x0[state_index] = value
        # else: silently ignore conditions that aren't needed
        # This happens when vector derivatives are provided but some
        # components don't need that level of derivative

    return x0
