"""
Differential-algebraic equations

A variable that occurs without derivatives is algebraic: it is determined by
the equations that contain no highest derivative (the algebraic equations).
The system is reduced to ordinary differential equations for the remaining
variables:

1. Algebraic equations that do not contain an algebraic variable are hidden
   constraints on the states. They are differentiated with respect to t,
   using the differential equations, until an algebraic variable appears
   (index reduction). The initial values must satisfy the constraint and
   its derivatives up to that point; drift away from the constraint is
   damped with Baumgarte stabilization, i.e. the k-th derivative c^(k) = 0
   is replaced by (d/dt + alpha)^k c = 0.
2. If the resulting equations are linear in the algebraic variables, these
   are eliminated symbolically.
3. Otherwise they are computed numerically with Newton's method whenever the
   right-hand side is evaluated.
"""

import math
from collections.abc import Callable

import numpy as np
import sympy as sp
from scipy.optimize import root

from .compile import CompiledSystem
from .errors import NonSolvableFormError
from .reduce import _substitute_derivatives_with_states, build_state_map
from .symbols import as_sympy

# Rate of the Baumgarte stabilization; bound like a parameter
BAUMGARTE = sp.Symbol("_odecast_baumgarte")

MAX_DIFFERENTIATIONS = 3


def compile_dae(eqs: list, orders: dict, t_symbol: sp.Symbol) -> CompiledSystem:
    """
    Reduce a differential-algebraic system and compile it.

    Args:
        eqs: Scalar equations
        orders: Variable orders; algebraic variables have order 0
        t_symbol: SymPy symbol of the independent variable

    Returns:
        CompiledSystem, or NumericDAE if the algebraic variables must be
        computed numerically

    Raises:
        NonSolvableFormError: If the system cannot be brought into a form in
            which all highest derivatives and algebraic variables are determined
    """
    mapping = build_state_map(orders)
    states = [
        sp.Symbol(f"{CompiledSystem.STATE_PREFIX}{i}") for i in range(len(mapping))
    ]
    differential = [v for v, order in orders.items() if order > 0]
    algebraic = [v for v, order in orders.items() if order == 0]

    d_syms = {v: sp.Symbol(f"_odecast_d{i}") for i, v in enumerate(differential)}
    z_syms = {v: sp.Symbol(f"_odecast_z{i}") for i, v in enumerate(algebraic)}
    replace = {v.d(orders[v]).sympy(): d_syms[v] for v in differential}
    replace.update({v.sympy(): z_syms[v] for v in algebraic})

    residuals = []
    for eq in eqs:
        r = (as_sympy(eq.lhs) - as_sympy(eq.rhs)).xreplace(replace)
        residuals.append(_substitute_derivatives_with_states(r, mapping, states))

    d_set = set(d_syms.values())
    differential_eqs = [r for r in residuals if r.free_symbols & d_set]
    constraints = [r for r in residuals if not r.free_symbols & d_set]
    names = ", ".join(v.name for v in algebraic)
    if len(constraints) != len(algebraic):
        raise NonSolvableFormError(
            f"The algebraic variables ({names}) need {len(algebraic)} algebraic "
            f"equation(s) without highest derivatives, got {len(constraints)}"
        )

    solutions = sp.solve(differential_eqs, list(d_syms.values()), dict=True)
    if not solutions or any(d not in solutions[0] for d in d_set):
        raise NonSolvableFormError(
            "Cannot solve the differential equations for the highest derivatives"
        )
    highest = {v: solutions[0][d_syms[v]] for v in differential}

    # Time derivatives of the states
    xdot = [None] * len(states)
    for (v, level), i in mapping.items():
        if level < orders[v] - 1:
            xdot[i] = states[mapping[(v, level + 1)]]
        else:
            xdot[i] = highest[v]

    def total_derivative(c):
        return sp.diff(c, t_symbol) + sum(
            sp.diff(c, x) * dx for x, dx in zip(states, xdot, strict=True)
        )

    z_set = set(z_syms.values())
    invariants = []
    reduced = []
    for c0 in constraints:
        chain = [c0]
        while not chain[-1].free_symbols & z_set:
            if len(chain) > MAX_DIFFERENTIATIONS:
                raise NonSolvableFormError(
                    f"The algebraic equation {_label(c0, mapping, states)} = 0 does not "
                    f"determine any of the algebraic variables ({names}), even after "
                    f"{MAX_DIFFERENTIATIONS} differentiations"
                )
            chain.append(total_derivative(chain[-1]))
        k = len(chain) - 1
        for j in range(k):
            what = "" if j == 0 else f"its derivative of order {j}"
            label = f"{_label(c0, mapping, states)}" + (f" ({what})" if what else "")
            invariants.append((chain[j], label))
        reduced.append(
            sum(math.comb(k, j) * BAUMGARTE ** (k - j) * chain[j] for j in range(k + 1))
        )

    z_list = list(z_syms.values())
    jacobian_z = sp.Matrix(reduced).jacobian(z_list)
    if sp.simplify(jacobian_z.det()) == 0:
        raise NonSolvableFormError(
            f"The algebraic equations do not determine the algebraic variables "
            f"({names}). This structure requires an index reduction that is not "
            "supported."
        )

    z_by_name = {v.name: z for v, z in z_syms.items()}
    constraints_original = [c for c in constraints]
    exclude = list(z_list)

    if all(_is_linear(r, z_list) for r in reduced):
        A, b = sp.linear_eq_to_matrix(reduced, z_list)
        z_exprs = list(A.LUsolve(b))
        z_subs = dict(zip(z_list, z_exprs, strict=True))
        f = sp.Matrix([dx.xreplace(z_subs) for dx in xdot])
        system = CompiledSystem(
            f,
            t_symbol,
            extra_exprs=[*z_exprs, *(c for c, _ in invariants), *constraints_original],
            exclude=exclude,
        )
        system.algebraic_exprs = {
            v.name: expr for v, expr in zip(algebraic, z_exprs, strict=True)
        }
    else:
        system = NumericDAE(
            xdot, reduced, z_list, t_symbol, extra_exprs=[c for c, _ in invariants]
        )

    system.z_syms = z_by_name
    system.invariants = invariants
    system.constraints = constraints_original
    return system


def default_baumgarte(system, params, t0, x0, z_guess) -> float:
    """
    Baumgarte rate for a system with hidden constraints.

    Constraint violations should decay faster than the dynamics evolve, so
    the rate is five times the largest absolute eigenvalue of the Jacobian
    at the initial point (computed without stabilization).
    """
    if x0 is None:
        return 0.0
    unstabilized = {**(params or {}), BAUMGARTE: 0.0}
    try:
        jac = system.jac(unstabilized, z_guess)(t0, np.asarray(x0, dtype=float))
        rate = np.abs(np.linalg.eigvals(jac)).max()
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return 0.0
    return 5.0 * float(rate) if np.isfinite(rate) else 0.0


def _is_linear(expr, symbols) -> bool:
    try:
        return sp.Poly(expr, *symbols).total_degree() <= 1
    except sp.PolynomialError:
        return False


def _label(expr, mapping, states) -> str:
    """Readable form of a constraint in terms of the variables."""
    names = {}
    for (v, level), i in mapping.items():
        names[states[i]] = sp.Symbol(v.name + "'" * level)
    return str(expr.xreplace(names))


class NumericDAE(CompiledSystem):
    """
    Differential-algebraic system whose algebraic variables are computed
    numerically.

    Whenever the right-hand side is evaluated, the algebraic equations
    g(t, x, z) = 0 are solved for z with Newton's method, starting from the
    previous solution. The Jacobian for implicit solvers follows from the
    implicit function theorem: dz/dx = -g_z^{-1} g_x.
    """

    def __init__(self, xdot, reduced, z_syms, t_symbol, extra_exprs=()):
        self.t_symbol = t_symbol
        self.f_sym = sp.Matrix(xdot)
        self.state_syms = [
            sp.Symbol(f"{self.STATE_PREFIX}{i}") for i in range(len(xdot))
        ]
        self._z_list = list(z_syms)
        self._g_sym = sp.Matrix(reduced)

        free = self.f_sym.free_symbols | self._g_sym.free_symbols
        for expr in extra_exprs:
            free |= expr.free_symbols
        free -= {t_symbol, *self.state_syms, *self._z_list}
        self.param_syms = tuple(sorted(free, key=lambda s: s.name))
        self._args = [t_symbol, *self.state_syms, *self.param_syms]

        args = [t_symbol, *self.state_syms, *self._z_list, *self.param_syms]
        self._dae_args = args
        self._F = sp.lambdify(args, list(self.f_sym), modules="numpy", cse=True)
        self._G = sp.lambdify(args, list(self._g_sym), modules="numpy", cse=True)
        self._Gz = sp.lambdify(
            args, self._g_sym.jacobian(self._z_list).tolist(), modules="numpy", cse=True
        )
        self._jacobians = None

        self.z_syms = {}
        self.algebraic_exprs = {}
        self.invariants = []
        self.constraints = []

    @property
    def jac_sym(self):
        raise ValueError(
            "The algebraic variables of this system are computed numerically; "
            "a symbolic Jacobian is not available"
        )

    def _initial_z(self, z_guess: dict | None) -> np.ndarray:
        guess = z_guess or {}
        return np.array([float(guess.get(name, 0.0)) for name in self.z_syms])

    def _solver(self, p: list[float]) -> Callable:
        """z(t, x, z0) solving the algebraic equations."""
        G, Gz, m = self._G, self._Gz, len(self._z_list)

        def residual(t, x, z):
            return np.asarray(G(t, *x, *z, *p), dtype=float)

        def jacobian(t, x, z):
            return np.asarray(Gz(t, *x, *z, *p), dtype=float).reshape(m, m)

        def solve_z(t, x, z0):
            with np.errstate(all="ignore"):
                return newton(t, x, z0)

        def newton(t, x, z0):
            z = np.array(z0, dtype=float)
            for _ in range(30):
                try:
                    step = np.linalg.solve(jacobian(t, x, z), -residual(t, x, z))
                except np.linalg.LinAlgError:
                    break
                if not np.all(np.isfinite(step)):
                    break
                z = z + step
                if np.linalg.norm(step) <= 1e-12 * (1.0 + np.linalg.norm(z)):
                    return z
            result = root(
                lambda zz: residual(t, x, zz), z0, jac=lambda zz: jacobian(t, x, zz)
            )
            if result.success:
                return result.x
            names = ", ".join(self.z_syms)
            message = " ".join(str(result.message).split())
            raise RuntimeError(
                f"Could not solve the algebraic equations for {names} at t={t:g}: "
                f"{message} The equations may have no solution there, or a better "
                "starting guess (in ivp) is needed."
            )

        return solve_z

    def rhs(self, params: dict | None = None, z_guess: dict | None = None) -> Callable:
        p = self.param_values(params)
        solve_z, F = self._solver(p), self._F
        last = {"z": self._initial_z(z_guess)}

        def rhs_function(t, x):
            # A failure at a trial point makes the solver reject the step and
            # retry with a smaller one; the message is kept for the case that
            # the integration fails altogether.
            try:
                z = solve_z(t, x, last["z"])
            except RuntimeError as e:
                rhs_function.failure = str(e)
                return np.full(len(x), np.nan)
            last["z"] = z
            return np.array(F(t, *x, *z, *p), dtype=float)

        rhs_function.failure = None
        return rhs_function

    def rhs_vectorized(
        self, params: dict | None = None, z_guess: dict | None = None
    ) -> Callable:
        rhs = self.rhs(params, z_guess)

        def rhs_function(t, x):
            t = np.atleast_1d(np.asarray(t, dtype=float))
            return np.column_stack([rhs(ti, x[:, i]) for i, ti in enumerate(t)])

        return rhs_function

    def jac(self, params: dict | None = None, z_guess: dict | None = None) -> Callable:
        p = self.param_values(params)
        if self._jacobians is None:
            x, z = self.state_syms, self._z_list
            self._jacobians = [
                sp.lambdify(self._dae_args, m.tolist(), modules="numpy", cse=True)
                for m in (
                    self.f_sym.jacobian(x),
                    self.f_sym.jacobian(z),
                    self._g_sym.jacobian(x),
                )
            ]
        Fx, Fz, Gx = self._jacobians
        solve_z, Gz = self._solver(p), self._Gz
        n, m = len(self.state_syms), len(self._z_list)
        last = {"z": self._initial_z(z_guess)}

        def jac_function(t, x):
            z = last["z"] = solve_z(t, x, last["z"])
            args = (t, *x, *z, *p)
            dz_dx = -np.linalg.solve(
                np.asarray(Gz(*args), dtype=float).reshape(m, m),
                np.asarray(Gx(*args), dtype=float).reshape(m, n),
            )
            return (
                np.asarray(Fx(*args), dtype=float).reshape(n, n)
                + np.asarray(Fz(*args), dtype=float).reshape(n, m) @ dz_dx
            )

        return jac_function

    def sensitivity_rhs(self, params, names):
        raise ValueError(
            "Sensitivities are not available when algebraic variables are "
            "computed numerically"
        )

    def algebraic_functions(
        self, params: dict | None = None, z_guess: dict | None = None
    ) -> dict[str, Callable]:
        p = self.param_values(params)
        solve_z = self._solver(p)
        z0 = self._initial_z(z_guess)

        def values(t, x):
            t = np.atleast_1d(np.asarray(t, dtype=float))
            x = np.asarray(x, dtype=float).reshape(len(self.state_syms), -1)
            out, z = np.empty((len(z0), t.size)), z0
            for i, ti in enumerate(t):
                z = out[:, i] = solve_z(ti, x[:, i], z)
            return out

        return {
            name: (lambda t, x, i=i: values(t, x)[i])
            for i, name in enumerate(self.z_syms)
        }
