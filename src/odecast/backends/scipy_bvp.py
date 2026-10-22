"""
SciPy-based backend for solving boundary value problems
"""

from collections.abc import Callable

import numpy as np
import sympy as sp
from scipy.integrate import solve_bvp

from ..solution import SolutionBVP
from ..symbols import Variable
from ..validate import normalize_ivp


class ScipyBVPBackend:
    """
    Backend for solving two-point boundary value problems using SciPy's
    solve_bvp.
    """

    def solve(
        self,
        f_sym_vec: sp.Matrix,
        t_symbol: sp.Symbol,
        state_syms: list[sp.Symbol],
        conditions: list[tuple[Variable, int, float, object]],
        tspan: tuple[float, float],
        mapping: dict,
        guess: dict | None = None,
        mesh: int | np.ndarray = 50,
        **options,
    ) -> SolutionBVP:
        """
        Solve a boundary value problem.

        Args:
            f_sym_vec: Right-hand side of the first-order system
            t_symbol: SymPy symbol for the independent variable
            state_syms: State symbols used in ``f_sym_vec``
            conditions: Validated (variable, level, t, value) boundary conditions
            tspan: Interval (a, b); all conditions lie at a or b
            mapping: State mapping from build_state_map
            guess: Initial guess, keyed like an IVP dictionary. Values may be
                numbers, arrays matching the mesh, or callables of t.
                Unspecified states start at zero.
            mesh: Number of initial mesh nodes, or the mesh itself
            **options: Passed to scipy.integrate.solve_bvp (tol, max_nodes,
                bc_tol, verbose)

        Returns:
            SolutionBVP object with results

        Raises:
            RuntimeError: If solve_bvp does not converge
        """
        a, b = tspan
        x = np.linspace(a, b, mesh) if np.isscalar(mesh) else np.asarray(mesh, float)

        fun = _vectorized_rhs(f_sym_vec, t_symbol, state_syms)

        residuals = []
        for v, level, t_value, value in conditions:
            at_start = bool(np.isclose(t_value, a))
            residuals.append((mapping[(v, level)], at_start, float(value)))

        def bc(ya, yb):
            return np.array(
                [
                    (ya if at_start else yb)[i] - value
                    for i, at_start, value in residuals
                ]
            )

        y0 = np.zeros((len(state_syms), x.size))
        for (v, level), value in normalize_ivp(guess or {}).items():
            if (v, level) not in mapping:
                raise ValueError(
                    f"Guess for {v.name}^({level}) does not match any state"
                )
            y0[mapping[(v, level)]] = value(x) if callable(value) else value

        sol = solve_bvp(fun, bc, x, y0, **options)
        if not sol.success:
            raise RuntimeError(f"SciPy solve_bvp failed: {sol.message}")

        return SolutionBVP(
            t=sol.x,
            y=sol.y,
            mapping=mapping,
            interpolant=sol.sol,
            f_compiled=lambda t, x: fun(np.atleast_1d(t), np.asarray(x)[:, None])[:, 0],
        )


def _vectorized_rhs(
    f_sym_vec: sp.Matrix, t_symbol: sp.Symbol, state_syms: list[sp.Symbol]
) -> Callable:
    """Compile f(t, x) for arrays of mesh points, as solve_bvp requires."""
    f = sp.lambdify([t_symbol, *state_syms], list(f_sym_vec), modules="numpy")

    def fun(t, y):
        values = f(t, *y)
        return np.array([np.broadcast_to(v, t.shape) for v in values], dtype=float)

    return fun
