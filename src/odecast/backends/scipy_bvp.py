"""
SciPy-based backend for solving boundary value problems
"""

from collections.abc import Callable

import numpy as np
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
        fun: Callable,
        n_states: int,
        conditions: list[tuple[Variable, int, float, object]],
        tspan: tuple[float, float],
        mapping: dict,
        guess: dict | None = None,
        mesh: int | np.ndarray = 50,
        p0: list[float] | None = None,
        **options,
    ) -> SolutionBVP:
        """
        Solve a boundary value problem.

        Args:
            fun: Right-hand side f(t, X) of the first-order system,
                vectorized over mesh points
            n_states: Number of first-order states
            conditions: Validated (variable, level, t, value) boundary conditions
            tspan: Interval (a, b); all conditions lie at a or b
            mapping: State mapping from build_state_map
            guess: Initial guess, keyed like an IVP dictionary. Values may be
                numbers, arrays matching the mesh, or callables of t.
                Unspecified states start at zero.
            mesh: Number of initial mesh nodes, or the mesh itself
            p0: Starting values of unknown parameters; ``fun`` then takes
                them as a third argument
            **options: Passed to scipy.integrate.solve_bvp (tol, max_nodes,
                bc_tol, verbose)

        Returns:
            SolutionBVP object with results

        Raises:
            RuntimeError: If solve_bvp does not converge
        """
        a, b = tspan
        x = np.linspace(a, b, mesh) if np.isscalar(mesh) else np.asarray(mesh, float)

        residuals = []
        for v, level, t_value, value in conditions:
            at_start = bool(np.isclose(t_value, a))
            residuals.append((mapping[(v, level)], at_start, float(value)))

        def bc(ya, yb, *p):
            return np.array(
                [
                    (ya if at_start else yb)[i] - value
                    for i, at_start, value in residuals
                ]
            )

        y0 = np.zeros((n_states, x.size))
        for (v, level), value in normalize_ivp(guess or {}).items():
            if (v, level) not in mapping:
                raise ValueError(
                    f"Guess for {v.name}^({level}) does not match any state"
                )
            y0[mapping[(v, level)]] = value(x) if callable(value) else value

        if p0 is not None:
            options["p"] = np.asarray(p0, dtype=float)
        sol = solve_bvp(fun, bc, x, y0, **options)
        if not sol.success:
            raise RuntimeError(f"SciPy solve_bvp failed: {sol.message}")

        solution = SolutionBVP(
            t=sol.x,
            y=sol.y,
            mapping=mapping,
            interpolant=sol.sol,
            f_compiled=(
                (
                    lambda t, x: fun(np.atleast_1d(t), np.asarray(x)[:, None], sol.p)[
                        :, 0
                    ]
                )
                if p0 is not None
                else (lambda t, x: fun(np.atleast_1d(t), np.asarray(x)[:, None])[:, 0])
            ),
        )
        solution.p = sol.p
        return solution
