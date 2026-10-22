"""
SciPy-based backend for solving initial value problems
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from ..solution import SolutionIVP
from ..symbols import Derivative, Variable


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

        sol = solve_ivp(fun=f_compiled, t_span=tspan, y0=x0, **options)

        if not sol.success:
            raise RuntimeError(f"SciPy solve_ivp failed: {sol.message}")

        return SolutionIVP(
            t=sol.t,
            y=sol.y,
            mapping=mapping,
            t0=t0,
            f_compiled=f_compiled,
            jac_compiled=jac_compiled,
            x0=x0,
            interpolant=sol.sol,
            t_events=sol.t_events,
            y_events=sol.y_events,
        )


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
