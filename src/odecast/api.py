"""
Main API functions for odecast
"""

from typing import Dict, Any, Optional, Union, List
from .symbols import t, Variable
from .equation import Eq


def var(name: str, order: Optional[int] = None) -> Variable:
    """
    Factory function for creating variables.

    Args:
        name: Name of the variable
        order: Maximum order of derivatives expected (for validation)

    Returns:
        Variable object that supports .d(n) for derivatives
    """
    return Variable(name, order)


def solve(equation, *, ivp=None, bvp=None, tspan=None, backend=None, **kwargs):
    """
    Solve an ordinary differential equation.

    Args:
        equation: An Eq object representing the ODE
        ivp: Dictionary of initial conditions for IVP
        bvp: List of boundary conditions for BVP
        tspan: Tuple of (t_start, t_end) for the solution domain
        backend: Backend to use ('scipy', 'sympy', 'scipy_bvp')
        **kwargs: Additional backend-specific options

    Returns:
        Solution object with methods to access results

    Raises:
        NotImplementedError: This is a placeholder implementation
    """
    raise NotImplementedError("solve() will be implemented in later playbooks")


class BC:
    """
    Boundary condition for BVP problems.

    This is a placeholder for boundary condition specification.
    """

    def __init__(self, variable, *, t=None, value=None):
        """
        Create a boundary condition.

        Args:
            variable: The variable to constrain
            t: The time/position where condition applies
            value: The value at that position
        """
        self.variable = variable
        self.t = t
        self.value = value

    def __repr__(self):
        return f"BC({self.variable}, t={self.t}, value={self.value})"
