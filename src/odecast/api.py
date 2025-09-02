"""
Main API functions for odecast
"""

from typing import Dict, Any, Optional, Union, List
from .symbols import t, Variable
from .equation import Eq


def var(name: str, order: Optional[int] = None) -> Variable:
    """
    Factory function for creating variables.

    Args:
        name: Name of the variable
        order: Maximum order of derivatives expected (for validation)

    Returns:
        Variable object that supports .d(n) for derivatives
    """
    return Variable(name, order)


def solve(equation, *, ivp=None, bvp=None, tspan=None, backend=None, **kwargs):
    """
    Solve an ordinary differential equation.

    Args:
        equation: An Eq object representing the ODE
        ivp: Dictionary of initial conditions for IVP
        bvp: List of boundary conditions for BVP
        tspan: Tuple of (t_start, t_end) for the solution domain
        backend: Backend to use ('scipy', 'sympy', 'scipy_bvp')
        **kwargs: Additional backend-specific options

    Returns:
        Solution object with methods to access results

    Raises:
        NotImplementedError: Symbolic and BVP backends not yet implemented
    """
    from .analyze import collect_variables, infer_orders, resolve_orders
    from .validate import validate_ivp
    from .reduce import build_state_map, isolate_highest_derivatives, make_rhs
    from .compile import lambdify_rhs, lambdify_jac
    from .backends.scipy_ivp import ScipyIVPBackend, convert_ivp_to_state_vector
    import sympy as sp

    # Normalize equations to list
    eqs = [equation] if isinstance(equation, Eq) else list(equation)

    # Step 1: Analyze - infer orders and resolve
    variables = collect_variables(eqs)
    inferred_orders = infer_orders(eqs)
    orders = resolve_orders(list(variables), inferred_orders)

    # Route to appropriate backend
    if backend is None:
        backend = "scipy"  # Default to scipy for IVP

    if backend == "auto":
        # Try SymPy first, fall back to SciPy on failure
        try:
            from .backends.sympy_backend import SymPyBackend

            backend_instance = SymPyBackend()
            solution = backend_instance.solve(
                equations=eqs, t_symbol=t.symbol, **kwargs
            )
            return solution
        except Exception:
            # Fall back to SciPy backend
            backend = "scipy"

    if backend == "scipy":
        if ivp is None:
            raise ValueError("IVP conditions required for scipy backend")
        if tspan is None:
            raise ValueError("tspan required for scipy backend")

        # Step 2: Validate IVP
        t0 = tspan[0]
        validate_ivp(orders, ivp, t0)

        # Step 3: Reduce to first-order system
        mapping = build_state_map(orders)
        highest_rules = isolate_highest_derivatives(eqs, orders)
        f_sym_vec, jac_sym = make_rhs(t.symbol, mapping, highest_rules)

        # Step 4: Convert IVP to state vector
        x0 = convert_ivp_to_state_vector(ivp, mapping)

        # Step 5: Compile to numerical functions
        # Determine state symbols
        n_states = len(x0)
        state_syms = [sp.Symbol(f"x{i}") for i in range(n_states)]

        f_compiled = lambdify_rhs(f_sym_vec, t.symbol, state_syms)
        jac_compiled = (
            lambdify_jac(jac_sym, t.symbol, state_syms) if jac_sym is not None else None
        )

        # Step 6: Solve using SciPy backend
        backend_instance = ScipyIVPBackend()
        solution = backend_instance.solve(
            f_compiled=f_compiled,
            jac_compiled=jac_compiled,
            x0=x0,
            t0=t0,
            tspan=tspan,
            mapping=mapping,
            options=kwargs,
        )

        return solution

    elif backend == "sympy":
        # SymPy backend for symbolic solutions
        from .backends.sympy_backend import SymPyBackend

        # Create SymPy backend instance
        backend_instance = SymPyBackend()
        solution = backend_instance.solve(equations=eqs, t_symbol=t.symbol, **kwargs)

        return solution

    elif backend == "scipy_bvp":
        raise NotImplementedError("BVP backend will be implemented in Milestone 5")

    else:
        raise ValueError(f"Unknown backend: {backend}")


class BC:
    """
    Boundary condition for BVP problems.

    This is a placeholder for boundary condition specification.
    """

    def __init__(self, variable, *, t=None, value=None):
        """
        Create a boundary condition.

        Args:
            variable: The variable to constrain
            t: The time/position where condition applies
            value: The value at that position
        """
        self.variable = variable
        self.t = t
        self.value = value

    def __repr__(self):
        return f"BC({self.variable}, t={self.t}, value={self.value})"
