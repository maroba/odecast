"""
Solution objects for storing and accessing ODE solutions
"""

import numpy as np
from typing import Dict, Any, Callable, Optional, Union, Tuple
from .symbols import Variable, Derivative


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
        f_compiled: Optional[Callable] = None,
        jac_compiled: Optional[Callable] = None,
        x0: Optional[np.ndarray] = None,
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
        """
        self.t = t
        self.y = y
        self.mapping = mapping
        self.t0 = t0
        self._f_compiled = f_compiled
        self._jac_compiled = jac_compiled
        self._x0 = x0

    def __getitem__(self, target: Union[Variable, Derivative]) -> np.ndarray:
        """
        Access solution values for a specific variable or derivative.

        Args:
            target: Variable or Derivative to get values for

        Returns:
            Array of solution values at all time points
        """
        if isinstance(target, Variable):
            # Check if it's a vector variable
            if target.shape is not None:
                # Vector variable - return 2D array with component values
                n_components = (
                    target.shape[0] if isinstance(target.shape, tuple) else target.shape
                )
                component_arrays = []

                for i in range(n_components):
                    component_var = target[i]
                    if (component_var, 0) in self.mapping:
                        state_index = self.mapping[(component_var, 0)]
                        component_arrays.append(self.y[state_index, :])
                    else:
                        raise KeyError(
                            f"Component {component_var.name} not found in solution"
                        )

                return np.vstack(component_arrays)
            else:
                # Scalar variable - variable itself is the 0th derivative
                if (target, 0) in self.mapping:
                    state_index = self.mapping[(target, 0)]
                    return self.y[state_index, :]
                else:
                    raise KeyError(f"Variable {target.name} not found in solution")

        elif isinstance(target, Derivative):
            # Derivative of specified order
            if (target.variable, target.order) in self.mapping:
                state_index = self.mapping[(target.variable, target.order)]
                return self.y[state_index, :]
            else:
                raise KeyError(
                    f"Derivative {target.variable.name}^({target.order}) not found in solution"
                )
        else:
            # Check if it's a VectorDerivative
            from .symbols import VectorDerivative

            if isinstance(target, VectorDerivative):
                # Vector derivative - return 2D array with component derivatives
                vector_var = target.variable
                n_components = (
                    vector_var.shape[0]
                    if isinstance(vector_var.shape, tuple)
                    else vector_var.shape
                )
                component_arrays = []

                for i in range(n_components):
                    component_var = vector_var[i]
                    if (component_var, target.order) in self.mapping:
                        state_index = self.mapping[(component_var, target.order)]
                        component_arrays.append(self.y[state_index, :])
                    else:
                        raise KeyError(
                            f"Derivative {component_var.name}^({target.order}) not found in solution"
                        )

                return np.vstack(component_arrays)
            else:
                raise TypeError(
                    f"Target must be Variable, Derivative, or VectorDerivative, got {type(target)}"
                )

    def eval(
        self, target, tpoints: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Evaluate solution at specific time points using interpolation.

        Args:
            target: Variable, Derivative, or VectorDerivative to evaluate
            tpoints: Time point(s) to evaluate at

        Returns:
            Interpolated solution value(s)
        """
        # Get the full solution array for this target
        y_values = self[target]

        # Handle vector case (2D array)
        if y_values.ndim == 2:
            # Vector result - interpolate each component
            result = np.zeros((y_values.shape[0], np.atleast_1d(tpoints).shape[0]))
            for i in range(y_values.shape[0]):
                result[i, :] = np.interp(tpoints, self.t, y_values[i, :])

            # Return scalar if single time point requested
            if np.isscalar(tpoints):
                result = result.squeeze(axis=1)

            return result
        else:
            # Scalar result - standard interpolation
            return np.interp(tpoints, self.t, y_values)

    def as_first_order(
        self,
    ) -> Tuple[Callable, Optional[Callable], np.ndarray, float, Dict]:
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


class Solution:
    """
    Base container for ODE solution data and methods.

    This is a placeholder for future symbolic solutions.
    """

    def __init__(self, t_values=None, y_values=None):
        self.t = t_values if t_values is not None else np.array([])
        self._y_values = y_values if y_values is not None else {}

    def __getitem__(self, variable):
        """Access solution values for a specific variable."""
        raise NotImplementedError(
            "Solution indexing will be implemented in later playbooks"
        )

    def as_first_order(self):
        """Return first-order system representation."""
        raise NotImplementedError(
            "as_first_order will be implemented in later playbooks"
        )

    def as_expr(self, variable):
        """Return symbolic expression for a variable."""
        raise NotImplementedError("as_expr will be implemented in later playbooks")
