"""
Compilation of symbolic expressions to numerical functions
"""

from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import sympy as sp


def lambdify_rhs(
    f_sym_vec: sp.Matrix,
    t_symbol: sp.Symbol,
    state_syms: list[sp.Symbol],
    param_names: list[str] | None = None,
) -> Callable:
    """
    Compile symbolic RHS vector to a numerical function.

    Args:
        f_sym_vec: SymPy Matrix representing the RHS vector f(t, x)
        t_symbol: SymPy symbol for time
        state_syms: List of SymPy symbols for state variables [x0, x1, ...]
        param_names: Optional list of parameter names to include in the function

    Returns:
        Callable function f(t, x) -> np.ndarray
    """
    # Convert Matrix to a list of expressions
    f_exprs = [f_sym_vec[i] for i in range(f_sym_vec.shape[0])]

    # Build argument list: t, then state variables, then parameters
    args = [t_symbol] + state_syms
    if param_names:
        param_syms = [sp.Symbol(name) for name in param_names]
        args.extend(param_syms)

    # Use SymPy's lambdify with numpy backend
    f_lambdified = sp.lambdify(args, f_exprs, modules="numpy")

    def rhs_function(t, x, *params):
        """
        RHS function compatible with scipy.integrate.solve_ivp.

        Args:
            t: Time value (scalar)
            x: State vector (numpy array)
            *params: Optional parameter values

        Returns:
            numpy array representing dx/dt
        """
        # Ensure x is a numpy array
        x = np.asarray(x)

        # Build argument list for the lambdified function
        args_vals = [t] + [x[i] for i in range(len(state_syms))]
        if params:
            args_vals.extend(params)

        # Call the lambdified function
        result = f_lambdified(*args_vals)

        # Ensure result is a numpy array
        if np.isscalar(result):
            return np.array([result])
        else:
            return np.array(result)

    return rhs_function


def lambdify_jac(
    jac_sym: sp.Matrix | None,
    t_symbol: sp.Symbol,
    state_syms: list[sp.Symbol],
    param_names: list[str] | None = None,
) -> Callable | None:
    """
    Compile symbolic Jacobian matrix to a numerical function.

    Args:
        jac_sym: SymPy Matrix representing the Jacobian df/dx (can be None)
        t_symbol: SymPy symbol for time
        state_syms: List of SymPy symbols for state variables
        param_names: Optional list of parameter names

    Returns:
        Callable function jac(t, x) -> np.ndarray, or None if jac_sym is None
    """
    if jac_sym is None:
        return None

    # Convert Matrix to nested list of expressions
    jac_exprs = [
        [jac_sym[i, j] for j in range(jac_sym.shape[1])]
        for i in range(jac_sym.shape[0])
    ]

    # Build argument list
    args = [t_symbol] + state_syms
    if param_names:
        param_syms = [sp.Symbol(name) for name in param_names]
        args.extend(param_syms)

    # Use SymPy's lambdify
    jac_lambdified = sp.lambdify(args, jac_exprs, modules="numpy")

    def jac_function(t, x, *params):
        """
        Jacobian function compatible with scipy.integrate.solve_ivp.

        Args:
            t: Time value (scalar)
            x: State vector (numpy array)
            *params: Optional parameter values

        Returns:
            numpy array representing df/dx
        """
        # Ensure x is a numpy array
        x = np.asarray(x)

        # Build argument list
        args_vals = [t] + [x[i] for i in range(len(state_syms))]
        if params:
            args_vals.extend(params)

        # Call the lambdified function
        result = jac_lambdified(*args_vals)

        # Ensure result is a 2D numpy array
        return np.array(result)

    return jac_function


def _symbol_named(symbols, name: str) -> sp.Symbol:
    """The symbol with the given name, or a new one (whose derivatives vanish)."""
    return next((s for s in symbols if s.name == name), sp.Symbol(name))


class CompiledSystem:
    """
    First-order system compiled to numerical functions.

    Free symbols other than t and the states are parameters; their values
    are bound when the numerical functions are requested. The Jacobian is
    computed on first use.

    For differential-algebraic systems (see :mod:`odecast.dae`), the
    algebraic variables have been eliminated from ``f_sym``;
    ``algebraic_exprs`` gives them in terms of t and the states,
    ``invariants`` lists hidden constraints that the initial values must
    satisfy, and ``constraints`` the original algebraic equations.
    """

    STATE_PREFIX = "_odecast_x"

    def __init__(
        self,
        f_sym_vec: sp.Matrix,
        t_symbol: sp.Symbol,
        extra_exprs=(),
        exclude=(),
    ):
        self.f_sym = f_sym_vec
        self.t_symbol = t_symbol
        self.state_syms = [
            sp.Symbol(f"{self.STATE_PREFIX}{i}") for i in range(f_sym_vec.shape[0])
        ]
        free = set(f_sym_vec.free_symbols)
        for expr in extra_exprs:
            free |= expr.free_symbols
        free -= {t_symbol, *self.state_syms, *exclude}
        self.param_syms = tuple(sorted(free, key=lambda s: s.name))
        self._args = [t_symbol, *self.state_syms, *self.param_syms]
        self._f_compiled = None
        self._jac_sym = None
        self._jac = None

        self.z_syms: dict[str, sp.Symbol] = {}
        self.algebraic_exprs: dict[str, sp.Expr] = {}
        self.invariants: list[tuple[sp.Expr, str]] = []
        self.constraints: list[sp.Expr] = []

    def _f(self) -> Callable:
        """f(t, *x, *p), compiled on first use."""
        if self._f_compiled is None:
            self._f_compiled = sp.lambdify(
                self._args, list(self.f_sym), modules="numpy", cse=True
            )
        return self._f_compiled

    @property
    def is_dae(self) -> bool:
        return bool(self.z_syms)

    @property
    def jac_sym(self) -> sp.Matrix:
        """Symbolic Jacobian df/dx."""
        if self._jac_sym is None:
            self._jac_sym = self.f_sym.jacobian(self.state_syms)
        return self._jac_sym

    def param_values(self, params: dict | None) -> list[float]:
        """Order parameter values like ``param_syms``, checking completeness."""
        by_name = {getattr(k, "name", k): v for k, v in (params or {}).items()}
        missing = [s.name for s in self.param_syms if s.name not in by_name]
        if missing:
            raise ValueError(
                f"Missing values for parameters: {', '.join(missing)}. "
                "Pass them with params={...}."
            )
        return [float(by_name[s.name]) for s in self.param_syms]

    def rhs(self, params: dict | None = None, z_guess: dict | None = None) -> Callable:
        """f(t, x) for scipy.integrate.solve_ivp."""
        p = self.param_values(params)
        f = self._f()

        def rhs_function(t, x):
            return np.array(f(t, *x, *p), dtype=float)

        return rhs_function

    def rhs_vectorized(
        self, params: dict | None = None, z_guess: dict | None = None
    ) -> Callable:
        """f(t, X) for arrays of time points, as scipy.integrate.solve_bvp requires."""
        p = self.param_values(params)
        f = self._f()

        def rhs_function(t, x):
            t = np.asarray(t)
            return np.array(
                [np.broadcast_to(v, t.shape) for v in f(t, *x, *p)], dtype=float
            )

        return rhs_function

    def jac(self, params: dict | None = None, z_guess: dict | None = None) -> Callable:
        """Jacobian J(t, x) for implicit solvers."""
        p = self.param_values(params)
        if self._jac is None:
            self._jac = sp.lambdify(
                self._args, self.jac_sym.tolist(), modules="numpy", cse=True
            )
        jac = self._jac

        def jac_function(t, x):
            return np.array(jac(t, *x, *p), dtype=float)

        return jac_function

    def sensitivity_rhs(self, params: dict | None, names: list[str]) -> Callable:
        """
        Right-hand side of the system augmented by its sensitivities.

        The state is (x, S) with S = dx/dp for the parameters ``names``,
        stored column by column after x. S obeys S' = J S + df/dp.
        """
        if not hasattr(self, "_sensitivity_cache"):
            self._sensitivity_cache = {}
        key = tuple(names)
        if key not in self._sensitivity_cache:
            symbols = [_symbol_named(self.param_syms, name) for name in names]
            fp = (
                self.f_sym.jacobian(symbols)
                if symbols
                else sp.zeros(self.f_sym.shape[0], 0)
            )
            self._sensitivity_cache[key] = (
                sp.lambdify(
                    self._args, self.jac_sym.tolist(), modules="numpy", cse=True
                ),
                sp.lambdify(self._args, fp.tolist(), modules="numpy", cse=True),
            )
        jac, fp = self._sensitivity_cache[key]
        p = self.param_values(params)
        f = self._f()
        n, m = len(self.state_syms), len(names)

        def rhs_function(t, y):
            x = y[:n]
            S = y[n:].reshape(m, n).T
            dS = np.asarray(jac(t, *x, *p), dtype=float).reshape(n, n) @ S + np.asarray(
                fp(t, *x, *p), dtype=float
            ).reshape(n, m)
            return np.concatenate([np.asarray(f(t, *x, *p), dtype=float), dS.T.ravel()])

        return rhs_function

    def _vectorized(self, exprs) -> Callable:
        """Compile expressions in (t, states, params) for arrays of time points."""
        f = sp.lambdify(self._args, list(exprs), modules="numpy", cse=True)

        def evaluate(t, x, p):
            t = np.asarray(t, dtype=float)
            shape = np.broadcast_shapes(t.shape, np.shape(x[0]))
            return np.array(
                [np.broadcast_to(v, shape) for v in f(t, *x, *p)], dtype=float
            )

        return evaluate

    def algebraic_functions(
        self, params: dict | None = None, z_guess: dict | None = None
    ) -> dict[str, Callable]:
        """Functions z(t, X) for the algebraic variables, keyed by name."""
        if not self.algebraic_exprs:
            return {}
        p = self.param_values(params)
        names = list(self.algebraic_exprs)
        evaluate = self._vectorized(self.algebraic_exprs.values())

        def make(i):
            return lambda t, x: evaluate(t, x, p)[i]

        return {name: make(i) for i, name in enumerate(names)}

    def check_consistency(self, t0: float, x0: np.ndarray, params: dict | None) -> None:
        """
        Check that initial values satisfy the hidden constraints.

        Raises:
            ValueError: If a constraint or one of its derivatives does not
                vanish at the initial point
        """
        if not self.invariants:
            return
        p = self.param_values(params)
        values = self._vectorized([expr for expr, _ in self.invariants])(t0, x0, p)
        scale = 1.0 + float(np.abs(x0).max(initial=0.0))
        for (_, label), value in zip(self.invariants, values, strict=True):
            if abs(value) > 1e-8 * scale:
                raise ValueError(
                    f"Initial values are inconsistent with the algebraic equations: "
                    f"{label} is {float(value):.3g} at t={t0}, but must be 0."
                )

    def constraint_function(
        self, params: dict | None = None, z_guess: dict | None = None
    ) -> Callable | None:
        """Residuals g(t, X) of the original algebraic equations."""
        if not self.constraints:
            return None
        p = self.param_values(params)
        z_names = list(self.z_syms)
        args = [
            self.t_symbol,
            *self.state_syms,
            *self.z_syms.values(),
            *self.param_syms,
        ]
        g = sp.lambdify(args, self.constraints, modules="numpy", cse=True)
        algebraic = self.algebraic_functions(params, z_guess)

        def residuals(t, x):
            t = np.asarray(t, dtype=float)
            z = [algebraic[name](t, x) for name in z_names]
            shape = np.broadcast_shapes(t.shape, np.shape(x[0]))
            return np.array(
                [np.broadcast_to(v, shape) for v in g(t, *x, *z, *p)], dtype=float
            )

        return residuals


_cache: "OrderedDict[tuple, CompiledSystem]" = OrderedDict()
_CACHE_SIZE = 64


def compile_system(eqs: list, orders: dict) -> CompiledSystem:
    """
    Reduce equations to a first-order system and compile it.

    Results are cached by the structure of the equations, so solving the
    same system repeatedly (e.g. for different parameter values) compiles
    it only once.

    Args:
        eqs: Scalar equations (vector equations already expanded)
        orders: Variable orders as returned by resolve_orders

    Returns:
        CompiledSystem whose states are ordered like build_state_map(orders)
    """
    from .reduce import build_state_map, isolate_highest_derivatives, make_rhs
    from .symbols import as_sympy, t

    key = (
        tuple(as_sympy(eq.lhs) - as_sympy(eq.rhs) for eq in eqs),
        tuple((v.name, order) for v, order in orders.items()),
    )
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    if any(order == 0 for order in orders.values()):
        from .dae import compile_dae

        system = compile_dae(eqs, orders, t.symbol)
        _cache[key] = system
        if len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
        return system

    mapping = build_state_map(orders)
    highest_rules = isolate_highest_derivatives(eqs, orders)
    f_sym_vec, _ = make_rhs(
        t.symbol, mapping, highest_rules, state_prefix=CompiledSystem.STATE_PREFIX
    )
    system = CompiledSystem(f_sym_vec, t.symbol)

    _cache[key] = system
    if len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return system
