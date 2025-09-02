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
            # Extract all variables from all equations
            from ..analyze import collect_variables

            variables = collect_variables(equations)

            # Handle single equation case
            if len(equations) == 1:
                eq = equations[0]
                sympy_eq = eq.sympy()

                if len(variables) != 1:
                    raise BackendError(
                        f"SymPy backend currently only supports single-variable equations, "
                        f"got variables: {[v.name for v in variables]}"
                    )

                var = list(variables)[0]
                y_func = sp.Function(var.name)(t_symbol)
                solution_expr = sp.dsolve(sympy_eq, y_func)

                if isinstance(solution_expr, sp.Eq):
                    solution_expr = solution_expr.rhs

                solutions = {var: solution_expr}

            # Handle multiple equations - try to solve each independently if they're decoupled
            else:
                solutions = {}

                # Group equations by variables to see if they're decoupled
                var_to_eqs = {}
                for eq in equations:
                    eq_vars = collect_variables([eq])
                    if len(eq_vars) == 1:
                        var = list(eq_vars)[0]
                        if var not in var_to_eqs:
                            var_to_eqs[var] = []
                        var_to_eqs[var].append(eq)
                    else:
                        # Coupled system - not supported yet
                        raise BackendError(
                            f"SymPy backend currently only supports decoupled systems, "
                            f"but equation {eq} involves variables: {[v.name for v in eq_vars]}"
                        )

                # Solve each variable's equation(s) independently
                for var, var_eqs in var_to_eqs.items():
                    if len(var_eqs) != 1:
                        raise BackendError(
                            f"Variable {var.name} appears in {len(var_eqs)} equations, "
                            f"but SymPy backend expects exactly one equation per variable"
                        )

                    eq = var_eqs[0]
                    sympy_eq = eq.sympy()
                    y_func = sp.Function(var.name)(t_symbol)
                    solution_expr = sp.dsolve(sympy_eq, y_func)

                    if isinstance(solution_expr, sp.Eq):
                        solution_expr = solution_expr.rhs

                    solutions[var] = solution_expr

            return SolutionExpr(solutions, t_symbol)

        except Exception as e:
            raise BackendError(f"SymPy failed to solve the equation: {e}") from e


def solve_symbolic_sympy(equation, **kwargs):
    """
    Solve an ODE symbolically using SymPy.

    This is a placeholder for future implementation.
    """
    raise NotImplementedError("SymPy backend will be implemented in later playbooks")
