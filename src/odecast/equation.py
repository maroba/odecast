"""
Equation objects for representing ODEs
"""

import numpy as np
import sympy as sp

from .symbols import Variable, VectorDerivative, as_sympy, component_of


def expand_vector_equations(equations: list["Eq"]) -> list["Eq"]:
    """
    Expand vector equations to component equations.

    Args:
        equations: List of equations that may contain vector operations

    Returns:
        List of expanded component equations
    """
    expanded = []

    for eq in equations:
        if _has_vector_operations(eq):
            # Expand this vector equation to component equations
            component_eqs = _expand_vector_equation(eq)
            expanded.extend(component_eqs)
        else:
            # Regular scalar equation
            expanded.append(eq)

    return expanded


def _has_vector_operations(eq: "Eq") -> bool:
    """Check if equation contains vector operations."""
    return bool(_find_vector_variables(eq))


def _expand_vector_equation(eq: "Eq") -> list["Eq"]:
    """
    Expand a single vector equation to component equations.

    For example: Eq(u.d(2) + u, 0) where u is shape (2,)
    Expands to: [Eq(u[0].d(2) + u[0], 0), Eq(u[1].d(2) + u[1], 0)]

    Matrix-valued variables are expanded in row-major order.
    """
    vector_vars = _find_vector_variables(eq)
    shapes = {v.shape for v in vector_vars}
    if len(shapes) > 1:
        names = ", ".join(f"{v.name}{v.shape}" for v in vector_vars)
        raise ValueError(
            f"Vector variables in one equation must have the same shape, got {names}"
        )
    (shape,) = shapes

    component_eqs = []
    for index in np.ndindex(*shape):
        index = index[0] if len(index) == 1 else index
        component_eqs.append(
            Eq(component_of(eq.lhs, index), component_of(eq.rhs, index))
        )
    return component_eqs


def _find_vector_variables(eq: "Eq") -> list:
    """Find all vector variables in the equation, in order of appearance."""
    vector_vars = []

    def collect_vectors(expr):
        if isinstance(expr, Variable) and expr.shape is not None:
            if expr not in vector_vars:
                vector_vars.append(expr)
        elif isinstance(expr, VectorDerivative):
            collect_vectors(expr.variable)
        elif hasattr(expr, "left") and hasattr(expr, "right"):
            collect_vectors(expr.left)
            collect_vectors(expr.right)

    collect_vectors(eq.lhs)
    collect_vectors(eq.rhs)
    return vector_vars


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
        >>> eq = Eq(y.d(2) + 0.3*y.d() + y, sp.sin(t))

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
        if isinstance(self.rhs, (int, float)) and self.rhs == 0:
            return f"Eq({self.lhs}, 0)"
        return f"Eq({self.lhs}, {self.rhs})"

    def __eq__(self, other):
        """Check equality of equations."""
        if not isinstance(other, Eq):
            return False
        return self.lhs == other.lhs and self.rhs == other.rhs
