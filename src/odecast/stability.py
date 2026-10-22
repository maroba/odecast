"""
Equilibria of autonomous systems and their linear stability
"""

import numpy as np
import sympy as sp

from .symbols import substitute_params, t


class Equilibrium:
    """
    An equilibrium of an autonomous system.

    Attributes:
        point: Values of the variables; all their derivatives are zero
        jacobian: Jacobian of the first-order system at the equilibrium
        eigenvalues: Eigenvalues of the Jacobian (complex numbers if the
            Jacobian is numeric, SymPy expressions otherwise)
        stability: Classification from the eigenvalues, e.g. "stable",
            "unstable", "stable focus", "saddle", "center". "marginal" means
            the linearization is inconclusive; "unknown" is used if the
            Jacobian contains symbols.
    """

    def __init__(self, point, jacobian, eigenvalues, stability):
        self.point = point
        self.jacobian = jacobian
        self.eigenvalues = eigenvalues
        self.stability = stability

    def __repr__(self):
        point = ", ".join(f"{v.name}={value}" for v, value in self.point.items())
        return f"Equilibrium({point}, {self.stability})"


def equilibria(equation, params: dict | None = None, *, tol: float = 1e-10):
    """
    Find the equilibria of an autonomous system and classify their stability.

    Equilibria are points where all derivatives vanish. They are found with
    :func:`sympy.solve`, so for periodic nonlinearities such as ``sin(y)``
    only a finite selection is returned. Complex solutions are discarded.

    Args:
        equation: Eq or list of Eq objects, as for :func:`odecast.solve`
        params: Values for parameters; parameters without a value stay
            symbolic in the results
        tol: Tolerance for treating an eigenvalue's real part as zero

    Returns:
        List of Equilibrium objects

    Raises:
        ValueError: If the system depends explicitly on t, or SymPy cannot
            solve for the equilibria

    Examples:
        >>> x, y = var("x"), var("y")
        >>> equilibria([Eq(x.d(), x - x*y), Eq(y.d(), x*y - y)])
        [Equilibrium(x=0, y=0, saddle), Equilibrium(x=1, y=1, center)]
    """
    from .analyze import infer_orders, resolve_orders
    from .compile import compile_system
    from .equation import Eq, expand_vector_equations
    from .reduce import build_state_map

    eqs = [equation] if isinstance(equation, Eq) else list(equation)
    eqs = expand_vector_equations(eqs)
    inferred = infer_orders(eqs)
    orders = resolve_orders(list(inferred), inferred)
    system = compile_system(eqs, orders)
    mapping = build_state_map(orders)
    from .delay import reject_delays

    reject_delays(system.f_sym, "equilibria()")
    if system.is_dae and (system.invariants or not system.algebraic_exprs):
        raise ValueError(
            "Equilibria of differential-algebraic systems are only supported if "
            "the algebraic equations are linear in the algebraic variables and "
            "determine them directly (index 1)"
        )

    f = substitute_params(system.f_sym, params)
    if t.symbol in f.free_symbols:
        raise ValueError(
            "Equilibria are only defined for autonomous systems (no explicit t)"
        )

    states = system.state_syms
    try:
        solutions = sp.solve(list(f), states, dict=True)
    except NotImplementedError as e:
        raise ValueError(f"SymPy cannot solve for the equilibria: {e}") from e

    # States that remain free (a continuum of equilibria) are shown by name
    names = {
        states[i]: sp.Symbol(v.name) for (v, level), i in mapping.items() if level == 0
    }

    result = []
    for sol in solutions:
        values = {s: sp.sympify(sol.get(s, s)).subs(names) for s in states}
        if any(v.is_real is False for v in values.values()):
            continue
        point = {
            v: values[states[i]] for (v, level), i in mapping.items() if level == 0
        }
        for v in orders:
            if v.name in system.algebraic_exprs:
                expr = substitute_params(system.algebraic_exprs[v.name], params)
                point[v] = expr.subs(sol).subs(names)
        jacobian = substitute_params(system.jac_sym, params).subs(sol).subs(names)
        result.append(_classify(point, jacobian, tol))
    return result


def _classify(point, jacobian: sp.Matrix, tol: float) -> Equilibrium:
    if jacobian.free_symbols:
        eigenvalues = [
            ev for ev, mult in jacobian.eigenvals().items() for _ in range(mult)
        ]
        return Equilibrium(point, jacobian, eigenvalues, "unknown")

    eigenvalues = np.linalg.eigvals(np.array(jacobian.evalf(), dtype=complex))
    re = eigenvalues.real
    scale = max(1.0, np.abs(eigenvalues).max())
    negative, positive = re < -tol * scale, re > tol * scale

    if negative.all():
        stability = "stable"
    elif positive.any():
        stability = "unstable"
    else:
        stability = "marginal"

    if len(eigenvalues) == 2:
        oscillating = abs(eigenvalues.imag).max() > tol * scale
        if positive.any() and negative.any():
            stability = "saddle"
        elif oscillating and not (positive.any() or negative.any()):
            stability = "center"
        elif oscillating:
            stability += " focus"
        elif stability != "marginal":
            stability += " node"

    return Equilibrium(point, jacobian, eigenvalues, stability)
