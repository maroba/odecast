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


class CompiledSystem:
    """
    First-order system compiled to numerical functions.

    Free symbols other than t and the states are parameters; their values
    are bound when the numerical functions are requested. The Jacobian is
    computed on first use.
    """

    STATE_PREFIX = "_odecast_x"

    def __init__(self, f_sym_vec: sp.Matrix, t_symbol: sp.Symbol):
        self.f_sym = f_sym_vec
        self.t_symbol = t_symbol
        self.state_syms = [
            sp.Symbol(f"{self.STATE_PREFIX}{i}") for i in range(f_sym_vec.shape[0])
        ]
        free = f_sym_vec.free_symbols - {t_symbol} - set(self.state_syms)
        self.param_syms = tuple(sorted(free, key=lambda s: s.name))
        self._args = [t_symbol, *self.state_syms, *self.param_syms]
        self._f = sp.lambdify(self._args, list(f_sym_vec), modules="numpy", cse=True)
        self._jac_sym = None
        self._jac = None

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

    def rhs(self, params: dict | None = None) -> Callable:
        """f(t, x) for scipy.integrate.solve_ivp."""
        p = self.param_values(params)
        f = self._f

        def rhs_function(t, x):
            return np.array(f(t, *x, *p), dtype=float)

        return rhs_function

    def rhs_vectorized(self, params: dict | None = None) -> Callable:
        """f(t, X) for arrays of time points, as scipy.integrate.solve_bvp requires."""
        p = self.param_values(params)
        f = self._f

        def rhs_function(t, x):
            t = np.asarray(t)
            return np.array(
                [np.broadcast_to(v, t.shape) for v in f(t, *x, *p)], dtype=float
            )

        return rhs_function

    def jac(self, params: dict | None = None) -> Callable:
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
