"""
Linearization around an operating point: state-space models and transfer
functions
"""

import sympy as sp

from .symbols import as_sympy, substitute_params


class LinearSystem:
    """
    Linear state-space model x' = A x + B u, y = C x + D u.

    The states x, inputs u and outputs y are deviations from the operating
    point. Matrices are SymPy matrices; they are numeric unless parameters
    were left without a value.

    Attributes:
        A, B, C, D: State-space matrices
        states: Names of the states, e.g. ["x", "x'"]
        inputs: Input symbols
        outputs: Output expressions
    """

    def __init__(self, A, B, C, D, states, inputs, outputs):
        self.A, self.B, self.C, self.D = A, B, C, D
        self.states = states
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self):
        return (
            f"LinearSystem(states={self.states}, inputs={self.inputs}, "
            f"outputs={self.outputs})"
        )

    def transfer_function(self, s: sp.Symbol | None = None) -> sp.Matrix:
        """
        Transfer function matrix G(s) = C (sI - A)^-1 B + D.

        Args:
            s: Laplace variable; ``sp.Symbol("s")`` if omitted

        Returns:
            SymPy matrix with one row per output and one column per input
        """
        s = s or sp.Symbol("s")
        n = self.A.shape[0]
        G = self.C * (s * sp.eye(n) - self.A).inv() * self.B + self.D
        return G.applyfunc(lambda g: _normalized(sp.cancel(g), s))

    def eigenvalues(self) -> list:
        """Eigenvalues of A (the poles of the system)."""
        return [ev for ev, mult in self.A.eigenvals().items() for _ in range(mult)]

    def to_scipy(self):
        """The model as scipy.signal.StateSpace (requires numeric matrices)."""
        from scipy.signal import StateSpace

        try:
            matrices = [
                sp.matrix2numpy(m, dtype=float)
                for m in (self.A, self.B, self.C, self.D)
            ]
        except TypeError as e:
            raise ValueError(
                "The matrices contain symbols; give all parameters a value"
            ) from e
        return StateSpace(*matrices)


def linearize(
    equation,
    around,
    inputs=(),
    outputs=None,
    params: dict | None = None,
) -> LinearSystem:
    """
    Linearize a system around an operating point.

    Args:
        equation: Eq or list of Eq objects. Inputs appear in them as SymPy
            symbols, e.g. a force ``F``.
        around: Operating point: values of the variables and inputs, e.g.
            ``{x: 0, F: 0}``, or an Equilibrium from :func:`odecast.equilibria`.
            Derivatives default to 0, inputs to 0.
        inputs: Input symbols
        outputs: Output expressions in the variables and their derivatives
            below the highest order; all variables if omitted
        params: Values of the remaining parameters; parameters without a
            value stay symbolic

    Returns:
        LinearSystem

    Raises:
        ValueError: If the operating point is not an equilibrium, or the
            system depends explicitly on t

    Examples:
        >>> x = var("x")
        >>> F = sp.Symbol("F")
        >>> lin = linearize(Eq(2*x.d(2) + 0.5*x.d() + 8*x, F), around={x: 0},
        ...                 inputs=[F], outputs=[x])
        >>> lin.transfer_function()   # 1/(2*s**2 + 0.5*s + 8)
    """
    from .analyze import infer_orders, resolve_orders
    from .compile import compile_system
    from .equation import Eq, expand_vector_equations
    from .reduce import _substitute_derivatives_with_states, build_state_map
    from .stability import Equilibrium
    from .symbols import t

    eqs = [equation] if isinstance(equation, Eq) else list(equation)
    eqs = expand_vector_equations(eqs)
    inferred = infer_orders(eqs)
    orders = resolve_orders(list(inferred), inferred)
    system = compile_system(eqs, orders)
    from .delay import reject_delays

    reject_delays(system.f_sym, "linearize()")
    if system.is_dae and (system.invariants or not system.algebraic_exprs):
        raise ValueError(
            "Linearization of differential-algebraic systems is only supported if "
            "the algebraic equations are linear in the algebraic variables and "
            "determine them directly (index 1)"
        )
    mapping = build_state_map(orders)
    states = system.state_syms
    if t.symbol in system.f_sym.free_symbols:
        raise ValueError("Linearization requires an autonomous system (no explicit t)")

    inputs = [sp.Symbol(i) if isinstance(i, str) else i for i in inputs]
    point = around.point if isinstance(around, Equilibrium) else dict(around)
    point_values = {}
    for key, value in point.items():
        if isinstance(key, sp.Symbol):
            continue
        level = getattr(key, "order", 0) if hasattr(key, "variable") else 0
        variable = key.variable if hasattr(key, "variable") else key
        if (variable, level) in mapping:
            point_values[states[mapping[(variable, level)]]] = value
    at = {s: point_values.get(s, 0) for s in states}
    at.update({u: point.get(u, point.get(u.name, 0)) for u in inputs})

    algebraic = {
        v.sympy(): system.algebraic_exprs[v.name]
        for v in orders
        if v.name in system.algebraic_exprs
    }

    def to_states(expr):
        expr = as_sympy(expr).xreplace(algebraic)
        expr = _substitute_derivatives_with_states(expr, mapping, states)
        expr = substitute_params(expr, params)
        if expr.atoms(sp.Derivative) or any(
            getattr(f, "args", None) == (t.symbol,) for f in expr.atoms(sp.Function)
        ):
            raise ValueError(
                f"Output {expr} may only involve variables and derivatives below "
                "their highest order"
            )
        return expr

    f = substitute_params(system.f_sym, params)
    residual = f.subs(at)
    if not all(_vanishes(r) for r in residual):
        raise ValueError(
            f"The operating point is not an equilibrium: the derivatives there are "
            f"{list(residual)}"
        )

    if outputs is None:
        outputs = [v for (v, level) in mapping if level == 0] + [
            v for v in orders if v.name in system.algebraic_exprs
        ]
    h = sp.Matrix([to_states(o) for o in outputs])
    u = list(inputs)

    def jacobian(m, wrt):
        if not wrt:
            return sp.zeros(m.shape[0], 0)
        return m.jacobian(wrt).subs(at)

    names = [""] * len(states)
    for (v, level), i in mapping.items():
        names[i] = v.name + "'" * level

    return LinearSystem(
        A=jacobian(f, states),
        B=jacobian(f, u),
        C=jacobian(h, states),
        D=jacobian(h, u),
        states=names,
        inputs=u,
        outputs=list(outputs),
    )


def _vanishes(expr, tol: float = 1e-9) -> bool:
    """Whether an expression is zero (up to rounding, if it is numeric)."""
    if expr.free_symbols:
        return sp.simplify(expr) == 0
    return abs(complex(expr)) <= tol


def _normalized(g, s):
    """Rational function in s with a monic denominator if it is numeric."""
    numerator, denominator = sp.fraction(g)
    if denominator.free_symbols - {s} or numerator.free_symbols - {s}:
        return sp.factor(g)
    lead = sp.Poly(denominator, s).LC()
    monic = sp.expand(denominator / lead).xreplace({sp.Float(1.0): sp.Integer(1)})
    return sp.expand(numerator / lead) / monic
