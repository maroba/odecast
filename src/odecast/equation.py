"""
Equation objects for representing ODEs
"""

from typing import Any
import sympy as sp
from .symbols import as_sympy


class Eq:
    """
    Represents an ordinary differential equation.

    This is typically created as Eq(lhs, rhs) where lhs and rhs are
    expressions involving variables and their derivatives.
    """

    def __init__(self, lhs, rhs=0):
        """
        Create an equation lhs = rhs.

        Args:
            lhs: Left-hand side expression
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
