"""
Equation objects for representing ODEs
"""

from typing import Any
import sympy as sp
from .symbols import as_sympy


class Eq:
    """
    Represents an ordinary differential equation.

    The Eq class is used to create equations of the form lhs = rhs, where the
    left and right sides can contain variables, derivatives, and expressions.
    This mirrors SymPy's Eq class but works with odecast's Variable and Derivative objects.

    Examples:
        Simple harmonic oscillator:
        >>> y = var("y")
        >>> eq = Eq(y.d(2) + y, 0)

        Damped oscillator with forcing:
        >>> eq = Eq(y.d(2) + 0.3*y.d() + y, sp.sin(t.symbol))

        First-order ODE:
        >>> eq = Eq(y.d() - 2*y, 0)

        Coupled system:
        >>> y, z = var("y"), var("z")
        >>> eq1 = Eq(y.d() - z, 0)
        >>> eq2 = Eq(z.d() + y, 0)
    """

    def __init__(self, lhs, rhs=0):
        """
        Create an equation lhs = rhs.

        Args:
            lhs: Left-hand side expression (can contain variables, derivatives, constants)
            rhs: Right-hand side expression (default 0)
        """
        self.lhs = lhs
        self.rhs = rhs

    def sympy(self) -> sp.Eq:
        """
        Convert this equation to a SymPy equation.

        Returns:
            SymPy Eq object
        """
        return sp.Eq(as_sympy(self.lhs), as_sympy(self.rhs))

    def __repr__(self):
        if self.rhs == 0:
            return f"Eq({self.lhs}, 0)"
        return f"Eq({self.lhs}, {self.rhs})"

    def __eq__(self, other):
        """Check equality of equations."""
        if not isinstance(other, Eq):
            return False
        return self.lhs == other.lhs and self.rhs == other.rhs
