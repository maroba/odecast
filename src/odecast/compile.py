"""
Compilation of symbolic expressions to numerical functions
"""

from typing import List, Callable, Optional
import sympy as sp
import numpy as np


def lambdify_rhs(
    f_sym_vec: sp.Matrix,
    t_symbol: sp.Symbol,
    state_syms: List[sp.Symbol],
    param_names: Optional[List[str]] = None,
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
    jac_sym: Optional[sp.Matrix],
    t_symbol: sp.Symbol,
    state_syms: List[sp.Symbol],
    param_names: Optional[List[str]] = None,
) -> Optional[Callable]:
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


def compile_to_function(expression):
    """
    Compile a symbolic expression to a numerical function.

    This is a placeholder that will be removed once the above functions are integrated.
    """
    raise NotImplementedError(
        "compile_to_function will be implemented in later playbooks"
    )
