"""
LaTeX rendering of equations and solutions
"""

import re

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.printing.latex import LatexPrinter

from .symbols import t

_COMPONENT = re.compile(r"^(.+)\[([0-9, ]+)\]$")


def _name(func_name: str) -> str:
    """LaTeX for a variable name: Greek letters, components as subscripts."""
    match = _COMPONENT.match(func_name)
    if match:
        base, index = match.groups()
        return f"{sp.latex(sp.Symbol(base))}_{{{index.replace(' ', '')}}}"
    return sp.latex(sp.Symbol(func_name))


def _is_unknown(expr) -> bool:
    return isinstance(expr, AppliedUndef) and expr.args == (t.symbol,)


class _Printer(LatexPrinter):
    """Prints y(t) as y and its derivatives as y', y'', y''', y^{(4)}."""

    def _print_Function(self, expr, exp=None):
        if _is_unknown(expr):
            name = _name(expr.func.__name__)
            return name if exp is None else f"{name}^{{{exp}}}"
        return super()._print_Function(expr, exp)

    def _print_Derivative(self, expr):
        if _is_unknown(expr.expr) and set(expr.variables) == {t.symbol}:
            name = _name(expr.expr.func.__name__)
            n = expr.derivative_count
            return name + "'" * n if n <= 3 else f"{name}^{{({n})}}"
        return super()._print_Derivative(expr)


def latex(expr) -> str:
    """
    LaTeX code for an equation or expression.

    Args:
        expr: Eq, odecast expression, or SymPy expression

    Returns:
        LaTeX string without surrounding $ signs
    """
    from .equation import Eq
    from .symbols import as_sympy

    if isinstance(expr, Eq):
        return f"{latex(expr.lhs)} = {latex(expr.rhs)}"
    return _Printer().doprint(as_sympy(expr))


def latex_solution(solutions: dict) -> str:
    """LaTeX code for a dict mapping variables to solution expressions."""
    lines = [
        rf"{_name(v.name)}(t) &= {_Printer().doprint(expr)}"
        for v, expr in solutions.items()
    ]
    return r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}"
