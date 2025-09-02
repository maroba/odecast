"""
Main APIs for odecast
"""

from typing import Dict, Any, Optional, Union, List
from .symbols import t, Variable
from .equation import Eq


def var(
    name: str, order: Optional[int] = None, shape: Optional[tuple] = None
) -> Variable:
    """
    Create a dependent variable for use in differential equations.

    The variable represents an unknown function y(t) where t is the independent variable.
    You can create derivatives using the .d(n) method.

    Args:
        name: Name of the variable (e.g., "y", "x", "theta")
        order: Maximum order of derivatives expected (optional, for validation).
               If specified and equations use higher-order derivatives, an
               OrderMismatchError will be raised.
        shape: Shape tuple for vector/matrix variables (e.g., (3,) for 3D vector,
               (2, 2) for 2x2 matrix). None for scalar variables.

    Returns:
        Variable object that supports:
        - Arithmetic operations: y + z, 2*y, y/3, etc.
        - Derivative notation: y.d() for y', y.d(2) for y'', etc.
        - Use in equations: Eq(y.d(2) + y, 0)
        - Component access for vectors: u[0], u[1] for vector components
        - Vectorized operations: u.d() + 2*u for vector equations

    Examples:
        >>> y = var("y")           # Create scalar variable y(t)
        >>> y.d()                  # First derivative y'(t)
        >>> y.d(2)                 # Second derivative y''(t)
        >>> Eq(y.d(2) + y, 0)     # Simple harmonic oscillator

        >>> u = var("u", shape=(2,))  # Create 2D vector variable u(t)
        >>> u[0]                      # First component u[0](t)
        >>> u.d()                     # Vector derivative u'(t)
        >>> Eq(u.d() + u, 0)         # Vector equation u' + u = 0

        >>> z = var("z", order=1)  # Declare z as first-order only
        >>> Eq(z.d() - z, 0)       # Valid: uses only first derivative
        >>> Eq(z.d(2) + z, 0)      # Would raise OrderMismatchError
    """
    return Variable(name, order, shape)


def solve(equation, *, ivp=None, bvp=None, tspan=None, backend=None, **kwargs):
    """
    Solve an ordinary differential equation or system of ODEs.

    This is the main entry point for solving differential equations. It supports
    both symbolic and numeric solutions through different backends.

    Args:
        equation: Single Eq object or list of Eq objects representing the ODE system
        ivp: Dictionary of initial conditions for initial value problems (IVP).
             Keys can be Variable objects (for y(t0)) or Derivative objects (for y'(t0)).
             Example: {y: 1.0, y.d(): 0.0} for y(0)=1, y'(0)=0
        bvp: List of boundary conditions for boundary value problems (BVP).
             Currently not implemented - will be available in future versions.
        tspan: Tuple of (t_start, t_end) for numeric solutions.
               Required for scipy backend, ignored for sympy backend.
        backend: Solution backend to use:
                 - "scipy": Numeric integration using SciPy (default)
                 - "sympy": Symbolic solution using SymPy
                 - "auto": Try SymPy first, fall back to SciPy if needed
                 - "scipy_bvp": BVP solver (not yet implemented)
        **kwargs: Additional backend-specific options:
                  For scipy: method, rtol, atol, max_step, etc.

    Returns:
        Solution object that depends on the backend:
        - SolutionIVP: For numeric solutions, supports sol[y], sol[y.d()], etc.
        - SolutionExpr: For symbolic solutions, supports sol.as_expr(y)

    Raises:
        MissingInitialConditionError: When required initial conditions are missing
        OverdeterminedConditionsError: When too many initial conditions provided
        OrderMismatchError: When declared variable order conflicts with equation usage
        BackendError: When the chosen backend cannot solve the equation
        ValueError: When required arguments are missing (e.g., tspan for scipy)

    Examples:
        Numeric solution of second-order ODE:
        >>> y = var("y")
        >>> eq = Eq(y.d(2) + 0.3*y.d() + y, 0)
        >>> sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 10))
        >>> y_values = sol[y]         # Get y(t) values
        >>> yprime_values = sol[y.d()] # Get y'(t) values

        Symbolic solution:
        >>> eq = Eq(y.d(2) + y, 0)
        >>> sol = solve(eq, backend="sympy")
        >>> expr = sol.as_expr(y)    # Get SymPy expression

        Coupled system:
        >>> y, z = var("y"), var("z")
        >>> eq1 = Eq(y.d(2) + z, 0)
        >>> eq2 = Eq(z.d() - y, 0)
        >>> sol = solve([eq1, eq2], ivp={y: 1.0, y.d(): 0.0, z: 0.0}, tspan=(0, 5))
    """
    from .analyze import collect_variables, infer_orders, resolve_orders
    from .validate import validate_ivp
    from .reduce import build_state_map, isolate_highest_derivatives, make_rhs
    from .compile import lambdify_rhs, lambdify_jac
    from .backends.scipy_ivp import ScipyIVPBackend, convert_ivp_to_state_vector
    from .equation import expand_vector_equations
    import sympy as sp

    # Normalize equations to list
    eqs = [equation] if isinstance(equation, Eq) else list(equation)

    # Expand vector equations to component equations
    eqs = expand_vector_equations(eqs)

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
        # Import BVP backend
        from .backends.scipy_bvp import ScipyBVPBackend

        # Create BVP backend instance
        backend_instance = ScipyBVPBackend()

        # For now, just raise NotImplementedError as the actual implementation
        # will come in Milestone 5
        raise NotImplementedError("BVP backend will be implemented in Milestone 5")

    else:
        raise ValueError(f"Unknown backend: {backend}")


class BC:
    """
    Boundary condition for boundary value problems (BVP).

    Boundary conditions specify the value of a variable or its derivative
    at specific points in the domain. This is used for BVP problems where
    conditions are given at multiple points rather than all at the initial point.

    Args:
        variable: The variable or derivative to constrain (Variable or Derivative object)
        t: The time/position where the condition applies
        value: The required value at that position

    Examples:
        >>> y = var("y")
        >>> bc1 = BC(y, t=0, value=0)      # y(0) = 0
        >>> bc2 = BC(y, t=1, value=1)      # y(1) = 1
        >>> bc3 = BC(y.d(), t=0, value=0)  # y'(0) = 0

        BVP problem:
        >>> eq = Eq(y.d(2) + y, 0)
        >>> sol = solve(eq, bvp=[bc1, bc2], tspan=(0, 1), backend="scipy_bvp")

    Note:
        BVP solving is not yet implemented and will be available in future versions.
    """

    def __init__(self, variable, *, t=None, value=None):
        """
        Create a boundary condition.

        Args:
            variable: The variable or derivative to constrain
            t: The time/position where condition applies
            value: The value at that position
        """
        self.variable = variable
        self.t = t
        self.value = value

    def __repr__(self):
        return f"BC({self.variable}, t={self.t}, value={self.value})"
