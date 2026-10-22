"""
Analysis functions for ODEs - order inference, validation, etc.
"""

import sympy as sp
from sympy.core.function import AppliedUndef

from .equation import Eq
from .errors import OrderMismatchError
from .symbols import Derivative, Expression, Variable, as_sympy, lookup_variable, t


def collect_variables(eqs: list[Eq]) -> set[Variable]:
    """
    Collect all variables used in the given equations.

    Args:
        eqs: List of equations to analyze

    Returns:
        Set of Variable objects found in the equations
    """
    return set(infer_orders(eqs))


def infer_orders(eqs: list[Eq]) -> dict[Variable, int]:
    """
    Infer the maximum derivative order for each variable from the equations.

    The equations are inspected in their SymPy form, so variables that only
    occur inside functions such as ``sp.sin(y)`` are found as well.

    Args:
        eqs: List of equations to analyze

    Returns:
        Dictionary mapping each variable to its maximum derivative order,
        in order of first appearance
    """
    known: dict[str, Variable] = {}
    for eq in eqs:
        _collect_variables_from_expr(eq.lhs, known)
        _collect_variables_from_expr(eq.rhs, known)

    found: dict[str, int] = {}
    for eq in eqs:
        expr = as_sympy(eq.lhs) - as_sympy(eq.rhs)
        for func in expr.atoms(AppliedUndef):
            if func.args == (t.symbol,):
                found.setdefault(func.func.__name__, 0)
        for deriv in expr.atoms(sp.Derivative):
            func = deriv.expr
            if isinstance(func, AppliedUndef) and func.args == (t.symbol,):
                name = func.func.__name__
                found[name] = max(found.get(name, 0), deriv.derivative_count)

    # Keep the order in which variables appear in the equations
    names = [n for n in known if n in found] + [n for n in found if n not in known]
    orders = {}
    for name in names:
        variable = known.get(name) or lookup_variable(name)
        if variable is None:
            raise ValueError(f"Unknown function '{name}' in equations")
        orders[variable] = found[name]
    return orders


def _collect_variables_from_expr(expr, variables: dict[str, Variable]):
    """Collect Variable objects from an expression tree, keyed by name."""
    if isinstance(expr, Variable):
        variables.setdefault(expr.name, expr)
    elif isinstance(expr, Derivative):
        variables.setdefault(expr.variable.name, expr.variable)
    elif isinstance(expr, Expression):
        _collect_variables_from_expr(expr.left, variables)
        _collect_variables_from_expr(expr.right, variables)
    # SymPy subexpressions are handled through the registry


def resolve_orders(
    variables: list[Variable], inferred: dict[Variable, int]
) -> dict[Variable, int]:
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
