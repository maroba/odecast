"""
Equation objects for representing ODEs
"""

from typing import Any, List, Union
import sympy as sp
from .symbols import as_sympy, VectorExpression, VectorDerivative, Variable


def expand_vector_equations(equations: List["Eq"]) -> List["Eq"]:
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
    return _contains_vector_ops(eq.lhs) or _contains_vector_ops(eq.rhs)


def _contains_vector_ops(expr) -> bool:
    """Recursively check if expression contains vector operations."""
    if isinstance(expr, (VectorExpression, VectorDerivative)):
        return True
    elif hasattr(expr, "left") and hasattr(expr, "right"):
        # Expression with left/right operands
        return _contains_vector_ops(expr.left) or _contains_vector_ops(expr.right)
    elif isinstance(expr, Variable) and expr.shape is not None:
        return True
    return False


def _expand_vector_equation(eq: "Eq") -> List["Eq"]:
    """
    Expand a single vector equation to component equations.

    For example: Eq(u.d(2) + u, 0) where u is shape (2,)
    Expands to: [Eq(u[0].d(2) + u[0], 0), Eq(u[1].d(2) + u[1], 0)]
    """
    # Find vector variables in the equation to determine shape
    vector_vars = _find_vector_variables(eq)

    if not vector_vars:
        return [eq]  # No vector variables found

    # For now, assume all vector variables have the same shape
    # This is a simplification for Playbook 11
    shape = list(vector_vars)[0].shape
    component_eqs = []

    for i in range(shape[0]):  # Handle 1D vectors for now
        # Expand LHS and RHS for component i
        lhs_component = _expand_expression_component(eq.lhs, i)
        rhs_component = _expand_expression_component(eq.rhs, i)

        # Create component equation
        component_eq = Eq(lhs_component, rhs_component)
        component_eqs.append(component_eq)

    return component_eqs


def _find_vector_variables(eq: "Eq") -> set:
    """Find all vector variables in the equation."""
    vector_vars = set()

    def collect_vectors(expr):
        if isinstance(expr, Variable) and expr.shape is not None:
            vector_vars.add(expr)
        elif isinstance(expr, VectorDerivative):
            vector_vars.add(expr.variable)
        elif hasattr(expr, "left") and hasattr(expr, "right"):
            collect_vectors(expr.left)
            collect_vectors(expr.right)

    collect_vectors(eq.lhs)
    collect_vectors(eq.rhs)
    return vector_vars


def _expand_expression_component(expr, index: int):
    """
    Expand an expression to its component for the given index.

    For example: u.d(2) + u with index=0 becomes u[0].d(2) + u[0]
    """
    if isinstance(expr, Variable) and expr.shape is not None:
        # Vector variable -> component variable
        return expr[index]
    elif isinstance(expr, VectorDerivative):
        # Vector derivative -> component derivative
        return expr[index]
    elif isinstance(expr, VectorExpression):
        # Vector expression -> expand operands and combine
        left_comp = _expand_expression_component(expr.left, index)
        right_comp = _expand_expression_component(expr.right, index)

        # Create scalar expression
        from .symbols import Expression

        return Expression(expr.operator, left_comp, right_comp)
    elif hasattr(expr, "left") and hasattr(expr, "right"):
        # Regular expression with vector operands
        left_comp = _expand_expression_component(expr.left, index)
        right_comp = _expand_expression_component(expr.right, index)

        # Preserve the expression type
        from .symbols import Expression

        return Expression(expr.operator, left_comp, right_comp)
    else:
        # Scalar or constant - same for all components
        return expr


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
