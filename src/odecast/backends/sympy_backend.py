"""
SymPy-based backend for symbolic ODE solving
"""

from typing import Dict, List, Union
import sympy as sp
from ..symbols import Variable, Derivative
from ..equation import Eq
from ..errors import BackendError


class SolutionExpr:
    """
    Symbolic solution object that wraps SymPy expressions.
    """

    def __init__(self, solutions: Dict[Variable, sp.Expr], t_symbol: sp.Symbol):
        """
        Initialize symbolic solution.

        Args:
            solutions: Dictionary mapping variables to their SymPy expressions
            t_symbol: The SymPy symbol representing the independent variable
        """
        self.solutions = solutions
        self.t_symbol = t_symbol

    def as_expr(self, var: Variable) -> sp.Expr:
        """
        Get the SymPy expression for a variable.

        Args:
            var: Variable to get expression for

        Returns:
            SymPy expression in terms of t

        Raises:
            KeyError: If variable not found in solution
        """
        if var not in self.solutions:
            raise KeyError(f"Variable {var.name} not found in symbolic solution")
        return self.solutions[var]


class SymPyBackend:
    """
    Backend that uses SymPy for symbolic ODE solving.
    """

    def solve(
        self, equations: List[Eq], t_symbol: sp.Symbol, **options
    ) -> SolutionExpr:
        """
        Solve ODEs symbolically using SymPy.

        Args:
            equations: List of equations to solve
            t_symbol: SymPy symbol for independent variable
            **options: Additional options (currently unused)

        Returns:
            SolutionExpr object with symbolic solutions

        Raises:
            BackendError: If SymPy cannot solve the system
        """
        try:
            # For now, only handle single equations
            if len(equations) != 1:
                raise BackendError(
                    f"SymPy backend currently only supports single equations, "
                    f"got {len(equations)} equations"
                )

            eq = equations[0]

            # Convert equation to SymPy form
            sympy_eq = eq.sympy()

            # Extract the variables from the equation
            from ..analyze import collect_variables

            variables = collect_variables([eq])

            if len(variables) != 1:
                raise BackendError(
                    f"SymPy backend currently only supports single-variable equations, "
                    f"got variables: {[v.name for v in variables]}"
                )

            var = list(variables)[0]

            # Create SymPy function for the variable
            y_func = sp.Function(var.name)(t_symbol)

            # Use SymPy's dsolve to solve the ODE
            solution_expr = sp.dsolve(sympy_eq, y_func)

            # Extract the right-hand side if it's an equation
            if isinstance(solution_expr, sp.Eq):
                solution_expr = solution_expr.rhs

            # Store the solution
            solutions = {var: solution_expr}

            return SolutionExpr(solutions, t_symbol)

        except Exception as e:
            raise BackendError(f"SymPy failed to solve the equation: {e}") from e


def solve_symbolic_sympy(equation, **kwargs):
    """
    Solve an ODE symbolically using SymPy.

    This is a placeholder for future implementation.
    """
    raise NotImplementedError("SymPy backend will be implemented in later playbooks")
