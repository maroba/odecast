"""
SymPy-based backend for symbolic ODE solving
"""

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef

from ..analyze import collect_variables, infer_orders
from ..equation import Eq
from ..errors import BackendError
from ..symbols import Derivative, Variable, VectorDerivative


class SolutionExpr:
    """
    Symbolic solution object that wraps SymPy expressions.

    Without initial or boundary conditions the expressions contain
    integration constants ``C1, C2, ...``.
    """

    def __init__(self, solutions: dict[Variable, sp.Expr], t_symbol: sp.Symbol):
        """
        Initialize symbolic solution.

        Args:
            solutions: Dictionary mapping variables to their SymPy expressions
            t_symbol: The SymPy symbol representing the independent variable
        """
        self.solutions = solutions
        self.t_symbol = t_symbol

    def as_expr(self, target) -> sp.Expr:
        """
        Get the SymPy expression for a variable or one of its derivatives.

        Args:
            target: Variable, Derivative, or vector/matrix variable or derivative

        Returns:
            SymPy expression in terms of t; a SymPy Array for vector and
            matrix targets

        Raises:
            KeyError: If variable not found in solution
        """
        if isinstance(target, Derivative):
            return sp.diff(self.as_expr(target.variable), self.t_symbol, target.order)
        if isinstance(target, VectorDerivative) or (
            isinstance(target, Variable) and target.shape is not None
        ):
            variable = target if isinstance(target, Variable) else target.variable
            level = 0 if isinstance(target, Variable) else target.order
            exprs = [
                sp.diff(self.as_expr(c), self.t_symbol, level)
                for c in variable.components()
            ]
            return sp.Array(exprs, variable.shape)
        if target not in self.solutions:
            raise KeyError(f"Variable {target.name} not found in symbolic solution")
        return self.solutions[target]

    def eval(self, target, tpoints: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the solution numerically.

        Args:
            target: Variable or derivative, as for ``as_expr``
            tpoints: Time point(s)

        Returns:
            Value(s) at ``tpoints``; for vector targets the component axis
            comes first

        Raises:
            ValueError: If the solution still contains integration constants
        """
        expr = self.as_expr(target)
        free = sp.Array([expr]).free_symbols - {self.t_symbol}
        if free:
            names = ", ".join(sorted(str(s) for s in free))
            raise ValueError(
                f"The solution contains undetermined constants ({names}). "
                "Pass initial or boundary conditions to determine them."
            )
        tpoints_arr = np.asarray(tpoints, dtype=float)
        if isinstance(expr, sp.Array):
            f = sp.lambdify(self.t_symbol, list(expr.reshape(len(expr))), "numpy")
            values = np.array(
                [np.broadcast_to(v, tpoints_arr.shape) for v in f(tpoints_arr)],
                dtype=float,
            )
            return values.reshape(*expr.shape, *tpoints_arr.shape)
        f = sp.lambdify(self.t_symbol, expr, "numpy")
        values = np.broadcast_to(f(tpoints_arr), tpoints_arr.shape).astype(float)
        return float(values) if values.ndim == 0 else values


class SymPyBackend:
    """
    Backend that uses SymPy for symbolic ODE solving.
    """

    def solve(
        self,
        equations: list[Eq],
        t_symbol: sp.Symbol,
        conditions: list[tuple[Variable, int, float, object]] | None = None,
        **options,
    ) -> SolutionExpr:
        """
        Solve ODEs symbolically using SymPy.

        Independent equations, each involving a single variable, are solved
        one at a time. Coupled systems are passed to ``sympy.dsolve`` as a
        whole, which supports linear systems.

        Args:
            equations: List of equations to solve
            t_symbol: SymPy symbol for independent variable
            conditions: Optional (variable, derivative level, t, value)
                tuples, used to determine the integration constants
            **options: Additional options (currently unused)

        Returns:
            SolutionExpr object with symbolic solutions

        Raises:
            BackendError: If SymPy cannot solve the system
        """
        conditions = conditions or []
        variables = list(infer_orders(equations))
        funcs = {v: sp.Function(v.name)(t_symbol) for v in variables}

        if len(equations) != len(variables):
            raise BackendError(
                f"SymPy backend needs one equation per variable, got "
                f"{len(equations)} equation(s) for variables "
                f"{[v.name for v in variables]}"
            )

        def ics_for(vs):
            ics = {}
            for v, level, t_value, value in conditions:
                if v in vs:
                    key = funcs[v].diff(t_symbol, level) if level else funcs[v]
                    ics[key.subs(t_symbol, t_value)] = value
            return ics

        def dsolve(eqs, vs):
            try:
                result = sp.dsolve(
                    [e.sympy() for e in eqs] if len(eqs) > 1 else eqs[0].sympy(),
                    [funcs[v] for v in vs] if len(vs) > 1 else funcs[vs[0]],
                    ics=ics_for(vs) or None,
                )
            except Exception as e:
                raise BackendError(f"SymPy failed to solve the equation: {e}") from e
            result = result if isinstance(result, list) else [result]
            by_name = {}
            for sol in result:
                if not isinstance(sol.lhs, AppliedUndef):
                    continue  # implicit solution
                name = sol.lhs.func.__name__
                if name in by_name:
                    raise BackendError(
                        f"SymPy found several solutions for {name}: "
                        f"{[r.rhs for r in result]}"
                    )
                by_name[name] = sol.rhs
            missing = [v.name for v in vs if v.name not in by_name]
            if missing:
                raise BackendError(
                    f"SymPy failed to solve the equation: no explicit solution "
                    f"for {', '.join(missing)}"
                )
            return {v: by_name[v.name] for v in vs}

        groups = [(eq, list(collect_variables([eq]))) for eq in equations]
        decoupled = all(len(vs) == 1 for _, vs in groups) and len(
            {vs[0] for _, vs in groups}
        ) == len(groups)

        solutions = {}
        if decoupled:
            for eq, vs in groups:
                solutions.update(dsolve([eq], vs))
        else:
            solutions.update(dsolve(equations, variables))

        return SolutionExpr(solutions, t_symbol)
