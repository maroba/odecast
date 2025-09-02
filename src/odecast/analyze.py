"""
Analysis functions for ODEs - order inference, validation, etc.
"""

from typing import Set, List, Dict
import sympy as sp
from .symbols import Variable, Derivative, as_sympy
from .equation import Eq
from .errors import OrderMismatchError


def collect_variables(eqs: List[Eq]) -> Set[Variable]:
    """
    Collect all variables used in the given equations.

    Args:
        eqs: List of equations to analyze

    Returns:
        Set of Variable objects found in the equations
    """
    variables = set()

    for eq in eqs:
        # Convert equation to SymPy and walk the expression tree
        sympy_eq = eq.sympy()

        # Get all functions in the equation (these correspond to our variables)
        functions = sympy_eq.atoms(sp.Function)

        # Map back to our Variable objects by walking the original expressions
        _collect_variables_from_expr(eq.lhs, variables)
        _collect_variables_from_expr(eq.rhs, variables)

    return variables


def _collect_variables_from_expr(expr, variables: Set[Variable]):
    """Helper function to recursively collect variables from an expression."""
    from .symbols import Expression  # Avoid circular import

    if isinstance(expr, Variable):
        variables.add(expr)
    elif isinstance(expr, Derivative):
        variables.add(expr.variable)
    elif isinstance(expr, Expression):
        _collect_variables_from_expr(expr.left, variables)
        _collect_variables_from_expr(expr.right, variables)
    # For other types (numbers, SymPy expressions), do nothing


def infer_orders(eqs: List[Eq]) -> Dict[Variable, int]:
    """
    Infer the maximum derivative order for each variable from the equations.

    Args:
        eqs: List of equations to analyze

    Returns:
        Dictionary mapping each variable to its maximum derivative order
    """
    orders = {}

    for eq in eqs:
        # Analyze both sides of the equation
        _infer_orders_from_expr(eq.lhs, orders)
        _infer_orders_from_expr(eq.rhs, orders)

    return orders


def _infer_orders_from_expr(expr, orders: Dict[Variable, int]):
    """Helper function to recursively infer orders from an expression."""
    from .symbols import Expression  # Avoid circular import

    if isinstance(expr, Variable):
        # Variable itself has order 0
        current_order = orders.get(expr, 0)
        orders[expr] = max(current_order, 0)
    elif isinstance(expr, Derivative):
        # Derivative has the specified order
        current_order = orders.get(expr.variable, 0)
        orders[expr.variable] = max(current_order, expr.order)
    elif isinstance(expr, Expression):
        _infer_orders_from_expr(expr.left, orders)
        _infer_orders_from_expr(expr.right, orders)
    # For other types (numbers, SymPy expressions), do nothing


def resolve_orders(
    variables: List[Variable], inferred: Dict[Variable, int]
) -> Dict[Variable, int]:
    """
    Resolve final orders by checking intent_order against inferred orders.

    Args:
        variables: List of variables to resolve orders for
        inferred: Dictionary of inferred orders from equations

    Returns:
        Dictionary mapping each variable to its resolved order

    Raises:
        OrderMismatchError: If intent_order is set but is less than inferred order
    """
    resolved = {}

    for var in variables:
        inferred_order = inferred.get(var, 0)

        if var.intent_order is not None:
            if inferred_order > var.intent_order:
                raise OrderMismatchError(
                    f"Variable '{var.name}' was declared with order {var.intent_order}, "
                    f"but equations use derivatives up to order {inferred_order}"
                )
            # Use the maximum of intent_order and inferred_order
            resolved[var] = max(var.intent_order, inferred_order)
        else:
            # Use inferred order
            resolved[var] = inferred_order

    return resolved


def analyze_equation(equation):
    """
    Analyze an equation to determine properties like order, variables involved, etc.

    This is a placeholder for future implementation.
    """
    raise NotImplementedError("analyze_equation will be implemented in later playbooks")
