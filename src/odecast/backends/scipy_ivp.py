"""
SciPy-based backend for solving initial value problems
"""

from typing import Dict, Tuple, Callable, Optional, Any, Union
import numpy as np
from scipy.integrate import solve_ivp

from ..symbols import Variable, Derivative
from ..solution import SolutionIVP


class ScipyIVPBackend:
    """
    Backend for solving IVPs using SciPy's solve_ivp.
    """

    def solve(
        self,
        f_compiled: Callable,
        jac_compiled: Optional[Callable],
        x0: np.ndarray,
        t0: float,
        tspan: Tuple[float, float],
        mapping: Dict[Union[Variable, Tuple[Variable, int]], Union[list, int]],
        options: Optional[Dict[str, Any]] = None,
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
            options: Additional solver options

        Returns:
            SolutionIVP object with results
        """
        options = options or {}

        # Extract solver options
        method = options.get("method", "RK45")
        rtol = options.get("rtol", 1e-3)
        atol = options.get("atol", 1e-6)
        events = options.get("events", None)
        dense_output = options.get("dense_output", False)
        max_step = options.get("max_step", np.inf)

        # Solve using scipy.integrate.solve_ivp
        sol = solve_ivp(
            fun=f_compiled,
            t_span=tspan,
            y0=x0,
            method=method,
            rtol=rtol,
            atol=atol,
            events=events,
            dense_output=dense_output,
            max_step=max_step,
            jac=jac_compiled,
        )

        if not sol.success:
            raise RuntimeError(f"SciPy solve_ivp failed: {sol.message}")

        # Create and return SolutionIVP object
        return SolutionIVP(
            t=sol.t,
            y=sol.y,
            mapping=mapping,
            t0=t0,
            f_compiled=f_compiled,
            jac_compiled=jac_compiled,
            x0=x0,
        )


def convert_ivp_to_state_vector(
    ivp_dict: Dict[Union[Variable, Derivative], float],
    mapping: Dict[Union[Variable, Tuple[Variable, int]], Union[list, int]],
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
        elif isinstance(key, Variable):  # var keys with lists
            if isinstance(value, list) and value:
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


def solve_ivp_scipy(equation, initial_conditions, tspan, **kwargs):
    """
    Solve an IVP using SciPy's solve_ivp.

    This is a placeholder that will be removed once the ScipyIVPBackend is integrated.
    """
    raise NotImplementedError(
        "SciPy IVP backend will be implemented in later playbooks"
    )
