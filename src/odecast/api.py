"""
Main APIs for odecast
"""

import numpy as np
import sympy as sp

from .equation import Eq
from .symbols import Variable, substitute_params, t


def var(name: str, order: int | None = None, shape: tuple | None = None) -> Variable:
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


def solve(
    equation,
    *,
    ivp=None,
    bvp=None,
    tspan=None,
    backend=None,
    params=None,
    events=None,
    **kwargs,
):
    """
    Solve an ordinary differential equation or system of ODEs.

    This is the main entry point for solving differential equations. It supports
    both symbolic and numeric solutions through different backends.

    Args:
        equation: Single Eq object or list of Eq objects representing the ODE
                  system
        ivp: Dictionary of initial conditions for initial value problems (IVP).
             Keys can be Variable objects (for y(t0)) or Derivative objects (for y'(t0)).
             Example: {y: 1.0, y.d(): 0.0} for y(0)=1, y'(0)=0
        bvp: List of BC objects for two-point boundary value problems (BVP).
             Example: [BC(y, t=0, value=0), BC(y, t=1, value=1)]
        tspan: Tuple of (t_start, t_end). Required for numeric solutions;
               initial conditions apply at t_start. For the sympy backend it
               is optional and only sets the time of the initial conditions
               (default 0).
        backend: Solution backend to use:
                 - "scipy": Numeric solution using SciPy (default)
                 - "sympy": Symbolic solution using SymPy
                 - "auto": Try SciPy first, fall back to SymPy if needed
                 - "scipy_bvp": Same as "scipy" with bvp

                 SciPy uses solve_ivp for an IVP and solve_bvp for a BVP.
        **kwargs: Additional backend-specific options:
                  For an IVP: method, rtol, atol, max_step, etc.
                  For a BVP: guess, mesh, tol, max_nodes, bc_tol

    Returns:
        Solution object that depends on the backend:
        - SolutionIVP: For numeric IVPs, supports sol[y], sol[y.d()], etc.
        - SolutionBVP: For numeric BVPs, with the same interface
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
        >>> sol = solve(eq, backend="sympy", ivp={y: 1, y.d(): 0})
        >>> expr = sol.as_expr(y)    # cos(t)

        Boundary value problem:
        >>> sol = solve(eq, bvp=[BC(y, t=0, value=0), BC(y, t=1, value=1)], tspan=(0, 1))

        Coupled system:
        >>> y, z = var("y"), var("z")
        >>> eq1 = Eq(y.d(2) + z, 0)
        >>> eq2 = Eq(z.d() - y, 0)
        >>> sol = solve([eq1, eq2], ivp={y: 1.0, y.d(): 0.0, z: 0.0}, tspan=(0, 5))
    """
    from .analyze import infer_orders, resolve_orders
    from .equation import expand_vector_equations

    if ivp is not None and bvp is not None:
        raise ValueError("Pass either ivp or bvp, not both")

    # Normalize equations to list
    eqs = [equation] if isinstance(equation, Eq) else list(equation)

    # Expand vector equations to component equations
    eqs = expand_vector_equations(eqs)

    # Analyze - infer orders and resolve
    inferred_orders = infer_orders(eqs)
    orders = resolve_orders(list(inferred_orders), inferred_orders)

    if backend is None or backend == "scipy_bvp":
        backend = "scipy"

    if backend == "scipy":
        return _solve_numeric(eqs, orders, ivp, bvp, tspan, params, events, kwargs)

    if backend == "sympy":
        return _solve_symbolic(eqs, orders, ivp, bvp, tspan, params, events, kwargs)

    if backend == "auto":
        # Try SciPy first, fall back to SymPy on failure.
        # Invalid conditions are the caller's error, not a reason to
        # switch backends, so validate outside the try block.
        if (ivp is not None or bvp is not None) and tspan is not None:
            _validate_conditions(orders, *_bind_conditions(ivp, bvp, params), tspan)
            try:
                return _solve_numeric(
                    eqs, orders, ivp, bvp, tspan, params, events, kwargs
                )
            # Any numerical failure triggers the fallback below
            except Exception:  # noqa: BLE001, S110
                pass

        try:
            return _solve_symbolic(eqs, orders, ivp, bvp, tspan, params, events, kwargs)
        except Exception as sympy_error:
            raise RuntimeError(
                f"Auto backend failed: Both SciPy and SymPy backends "
                f"failed to solve the equation. SymPy error: {sympy_error}"
            ) from sympy_error

    raise ValueError(f"Unknown backend: {backend}")


def _validate_conditions(orders, ivp, bvp, tspan):
    """Validate initial or boundary conditions for a numeric solution."""
    from .validate import normalize_conditions, validate_bvp, validate_ivp

    if bvp is not None:
        validate_bvp(orders, normalize_conditions(bvp=bvp), tspan)
    else:
        validate_ivp(orders, ivp, tspan[0])


def _bind(value, params):
    """Substitute parameter values into a condition value."""
    if isinstance(value, (list, tuple)):
        return type(value)(_bind(v, params) for v in value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return np.vectorize(lambda v: _bind(v, params), otypes=[float])(value)
    if isinstance(value, sp.Basic):
        bound = substitute_params(value, params)
        if bound.free_symbols:
            names = ", ".join(sorted(str(s) for s in bound.free_symbols))
            raise ValueError(f"Missing values for parameters: {names}")
        return float(bound)
    return value


def _bind_conditions(ivp, bvp, params):
    """Copies of ivp/bvp with parameter values substituted."""
    if ivp is not None:
        ivp = {k: _bind(v, params) for k, v in ivp.items()}
    if bvp is not None:
        bvp = [BC(bc.variable, t=bc.t, value=_bind(bc.value, params)) for bc in bvp]
    return ivp, bvp


def _solve_numeric(eqs, orders, ivp, bvp, tspan, params, events, options):
    """Reduce to a first-order system and solve it with SciPy."""
    from .backends.scipy_bvp import ScipyBVPBackend
    from .backends.scipy_ivp import ScipyIVPBackend, convert_ivp_to_state_vector
    from .compile import compile_system
    from .events import compile_events
    from .reduce import build_state_map
    from .validate import normalize_conditions

    if ivp is None and bvp is None:
        raise ValueError(
            "IVP conditions required for scipy backend (or bvp for a boundary "
            "value problem)"
        )
    if tspan is None:
        raise ValueError("tspan required for scipy backend")

    ivp, bvp = _bind_conditions(ivp, bvp, params)
    _validate_conditions(orders, ivp, bvp, tspan)

    system = compile_system(eqs, orders)
    mapping = build_state_map(orders)

    if bvp is not None:
        if events is not None:
            raise ValueError("Events are only supported for initial value problems")
        return ScipyBVPBackend().solve(
            fun=system.rhs_vectorized(params),
            n_states=len(system.state_syms),
            conditions=normalize_conditions(bvp=bvp),
            tspan=tspan,
            mapping=mapping,
            **options,
        )

    method = options.get("method", "RK45")
    implicit = not isinstance(method, str) or method in ("Radau", "BDF", "LSODA")
    events = compile_events(events, mapping, system.state_syms, t.symbol, params)
    if events is not None:
        options = {**options, "events": events}

    return ScipyIVPBackend().solve(
        f_compiled=system.rhs(params),
        jac_compiled=system.jac(params) if implicit else None,
        x0=convert_ivp_to_state_vector(ivp, mapping),
        t0=tspan[0],
        tspan=tspan,
        mapping=mapping,
        options=options,
    )


def _solve_symbolic(eqs, orders, ivp, bvp, tspan, params, events, options):
    """Solve with SymPy, applying initial or boundary conditions if given."""
    from .backends.sympy_backend import SymPyBackend
    from .validate import normalize_conditions, validate_condition_levels

    if events is not None:
        raise ValueError("Events are only supported by the scipy backend")

    conditions = None
    if ivp is not None or bvp is not None:
        t0 = tspan[0] if tspan is not None else 0
        conditions = normalize_conditions(ivp=ivp, bvp=bvp, t0=t0)
        validate_condition_levels(orders, conditions)

    return SymPyBackend().solve(
        equations=eqs,
        t_symbol=t.symbol,
        conditions=conditions,
        params=params,
        **options,
    )


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
        >>> sol = solve(eq, bvp=[bc1, bc2], tspan=(0, 1))

        Vector variables take a sequence of values:
        >>> u = var("u", shape=2)
        >>> BC(u, t=0, value=[0, 1])
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
