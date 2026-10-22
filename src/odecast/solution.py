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
        t_events: list[np.ndarray] | None = None,
        y_events: list[np.ndarray] | None = None,
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
            t_events: Times at which each event occurred
            y_events: State vectors at those times
        """
        self.t = t
        self.y = y
        self.mapping = mapping
        self.t0 = t0
        self._f_compiled = f_compiled
        self._jac_compiled = jac_compiled
        self._x0 = x0
        self._interpolant = interpolant
        self.t_events = t_events
        self._y_events = y_events

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

    def event_values(self, index: int, target) -> np.ndarray:
        """
        Values of a variable or derivative at the occurrences of an event.

        Args:
            index: Position of the event in the ``events`` list passed to solve
            target: Variable, Derivative, or vector/matrix variable or derivative

        Returns:
            Array with one value per occurrence (last axis)
        """
        if self._y_events is None:
            raise ValueError("No events were passed to solve")
        return self._select(target, np.asarray(self._y_events[index]).T)

    def variables(self) -> list[Variable]:
        """The scalar variables (and vector components) of the system."""
        return [v for (v, level) in self.mapping if level == 0]

    def _columns(self, targets, tpoints) -> list[tuple[str, np.ndarray]]:
        """Labels and values for plotting and tabulating."""
        columns = []
        for target in targets or self.variables():
            values = self.eval(target, tpoints)
            if np.ndim(values) == 1:
                columns.append((repr(target), values))
                continue
            for index in np.ndindex(*values.shape[:-1]):
                label = f"{target!r}[{','.join(map(str, index))}]"
                columns.append((label, values[index]))
        return columns

    def plot(self, *targets, ax=None, n: int = 500, **kwargs):
        """
        Plot variables or derivatives against t using Matplotlib.

        Args:
            *targets: Variables or derivatives to plot; all variables if omitted
            ax: Matplotlib axes to draw on; the current axes if omitted
            n: Number of evaluation points
            **kwargs: Passed to ``ax.plot``

        Returns:
            The Matplotlib axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                'plot() requires Matplotlib: pip install "odecast[examples]"'
            ) from e

        ax = ax or plt.gca()
        tpoints = np.linspace(self.t[0], self.t[-1], n)
        for label, values in self._columns(targets, tpoints):
            ax.plot(tpoints, values, label=label, **kwargs)
        ax.set_xlabel("t")
        ax.legend()
        return ax

    def to_pandas(self, *targets, t: np.ndarray | None = None):
        """
        Tabulate the solution as a pandas DataFrame indexed by t.

        Args:
            *targets: Variables or derivatives; all variables if omitted
            t: Time points; the solver's time points if omitted

        Returns:
            pandas.DataFrame with one column per scalar component
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                'to_pandas() requires pandas: pip install "odecast[examples]"'
            ) from e

        tpoints = self.t if t is None else np.asarray(t, dtype=float)
        columns = self._columns(targets, tpoints)
        return pd.DataFrame(dict(columns), index=pd.Index(tpoints, name="t"))

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
