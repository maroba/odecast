"""
Solution objects for storing and accessing ODE solutions
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from .symbols import Derivative, Variable


class SolutionIVP:
    """
    Container for IVP solution data with convenient access methods.
    """

    def __init__(
        self,
        t: np.ndarray,
        y: np.ndarray,
        mapping: Any,  # StateMapping from reduce.py
        t0: float,
        f_compiled: Callable | None = None,
        jac_compiled: Callable | None = None,
        x0: np.ndarray | None = None,
        interpolant: Callable | None = None,
    ):
        """
        Initialize IVP solution.

        Args:
            t: Time points array
            y: Solution array with shape (n_states, n_timepoints)
            mapping: State mapping from reduce.build_state_map
            t0: Initial time
            f_compiled: Compiled RHS function f(t, x)
            jac_compiled: Compiled Jacobian function (optional)
            x0: Initial state vector
            interpolant: Optional callable returning the state vector at
                given times; used by ``eval`` instead of linear interpolation
        """
        self.t = t
        self.y = y
        self.mapping = mapping
        self.t0 = t0
        self._f_compiled = f_compiled
        self._jac_compiled = jac_compiled
        self._x0 = x0
        self._interpolant = interpolant

    def __getitem__(self, target: Variable | Derivative) -> np.ndarray:
        """
        Access solution values for a specific variable or derivative.

        Args:
            target: Variable, Derivative, or vector/matrix variable or derivative

        Returns:
            Array of solution values at all time points. For a vector or
            matrix target the array has shape ``(*shape, len(t))``.
        """
        return self._select(target, self.y)

    def _select(self, target, states: np.ndarray) -> np.ndarray:
        """Pick the rows of a state array that belong to ``target``."""
        from .symbols import VectorDerivative

        if isinstance(target, (Derivative, VectorDerivative)):
            variable, level = target.variable, target.order
        elif isinstance(target, Variable):
            variable, level = target, 0
        else:
            raise TypeError(
                f"Target must be Variable, Derivative, or VectorDerivative, got {type(target)}"
            )

        if variable.shape is None:
            return states[self._state_index(variable, level)]
        rows = [states[self._state_index(c, level)] for c in variable.components()]
        return np.stack(rows).reshape(*variable.shape, *states.shape[1:])

    def _state_index(self, variable: Variable, level: int) -> int:
        if (variable, level) not in self.mapping:
            what = (
                variable.name if level == 0 else f"Derivative {variable.name}^({level})"
            )
            raise KeyError(f"{what} not found in solution")
        return self.mapping[(variable, level)]

    def eval(self, target, tpoints: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate solution at specific time points using interpolation.

        Args:
            target: Variable, Derivative, or VectorDerivative to evaluate
            tpoints: Time point(s) to evaluate at

        Returns:
            Interpolated solution value(s). For vector and matrix targets the
            component axes come first.
        """
        if self._interpolant is not None:
            values = self._select(target, self._interpolant(np.atleast_1d(tpoints)))
            return values[..., 0][()] if np.isscalar(tpoints) else values

        y_values = self[target]
        if y_values.ndim == 1:
            return np.interp(tpoints, self.t, y_values)

        # Vector/matrix result - interpolate each component
        flat = y_values.reshape(-1, y_values.shape[-1])
        result = np.array([np.interp(tpoints, self.t, row) for row in flat])
        return result.reshape(*y_values.shape[:-1], *np.shape(tpoints))

    def as_first_order(
        self,
    ) -> tuple[Callable, Callable | None, np.ndarray, float, dict]:
        """
        Return first-order system representation.

        Returns:
            Tuple of (f, jac, x0, t0, mapping) where:
            - f: Compiled RHS function f(t, x)
            - jac: Compiled Jacobian function (or None)
            - x0: Initial state vector
            - t0: Initial time
            - mapping: State mapping dictionary
        """
        if self._f_compiled is None:
            raise ValueError("No compiled RHS function available")
        if self._x0 is None:
            raise ValueError("No initial state vector available")

        return (self._f_compiled, self._jac_compiled, self._x0, self.t0, self.mapping)


class SolutionBVP(SolutionIVP):
    """
    Solution of a boundary value problem.

    Supports the same access methods as SolutionIVP. ``eval`` uses the
    cubic interpolant returned by ``scipy.integrate.solve_bvp``.
    """

    def __init__(
        self,
        t: np.ndarray,
        y: np.ndarray,
        mapping: Any,
        interpolant: Callable,
        f_compiled: Callable | None = None,
    ):
        super().__init__(
            t=t,
            y=y,
            mapping=mapping,
            t0=t[0],
            f_compiled=f_compiled,
            x0=y[:, 0],
            interpolant=interpolant,
        )
