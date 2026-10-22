"""
Continuation of equilibria and detection of bifurcations
"""

import numpy as np
import sympy as sp

from .symbols import substitute_params


class Bifurcation:
    """
    A bifurcation point on a branch of equilibria.

    Attributes:
        kind: "fold" (saddle-node), "hopf", or "branch point" (e.g.
            transcritical or pitchfork)
        param: Parameter value
        point: Values of the variables
        eigenvalues: Eigenvalues of the Jacobian at the point
    """

    def __init__(self, kind, param, point, eigenvalues):
        self.kind = kind
        self.param = param
        self.point = point
        self.eigenvalues = eigenvalues

    def __repr__(self):
        point = ", ".join(f"{v.name}={value:.6g}" for v, value in self.point.items())
        return f"Bifurcation({self.kind} at param={self.param:.6g}, {point})"


class Branch:
    """
    A branch of equilibria, as returned by :func:`odecast.continuation`.

    Attributes:
        param: Parameter values along the branch
        points: Values of the variables along the branch, keyed by variable
        stable: Whether each point is (linearly) stable
        eigenvalues: Eigenvalues of the Jacobian at each point
        bifurcations: Bifurcation points found on the branch
    """

    def __init__(self, param, points, stable, eigenvalues, bifurcations, symbol):
        self.param = param
        self.points = points
        self.stable = stable
        self.eigenvalues = eigenvalues
        self.bifurcations = bifurcations
        self.symbol = symbol

    def __repr__(self):
        return (
            f"Branch({len(self.param)} points, param from {self.param.min():.4g} "
            f"to {self.param.max():.4g}, bifurcations={self.bifurcations})"
        )

    def __getitem__(self, variable):
        return self.points[variable]

    def plot(self, variable, ax=None, **kwargs):
        """
        Bifurcation diagram: stable parts solid, unstable parts dashed.

        Args:
            variable: Variable on the vertical axis
            ax: Matplotlib axes; the current axes if omitted
            **kwargs: Passed to ``ax.plot``

        Returns:
            The Matplotlib axes
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        values = self.points[variable]
        kwargs.setdefault("color", "black")
        for stable, style in ((True, "-"), (False, "--")):
            masked = np.where(self.stable == stable, values, np.nan)
            # Join the segments at stability changes
            edges = np.flatnonzero(np.diff(self.stable.astype(int)))
            for k in edges:
                j = k if self.stable[k] != stable else k + 1
                masked[j] = values[j]
            ax.plot(self.param, masked, style, **kwargs)
        markers = {"fold": "s", "hopf": "o", "branch point": "D"}
        for b in self.bifurcations:
            ax.plot(
                b.param,
                b.point[variable],
                markers[b.kind],
                color="tab:red",
                markersize=7,
                label=b.kind,
            )
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            unique = dict(zip(labels, handles, strict=True))
            ax.legend(unique.values(), unique.keys(), fontsize="small")
        ax.set_xlabel(str(self.symbol))
        ax.set_ylabel(variable.name)
        return ax


def continuation(
    equation,
    param,
    start: dict,
    p_range: tuple[float, float],
    *,
    params: dict | None = None,
    step: float | None = None,
    max_points: int = 2000,
):
    """
    Follow a branch of equilibria as a parameter varies.

    Uses pseudo-arclength continuation, so the branch is followed around
    folds where the parameter turns back. Folds, Hopf bifurcations and
    branch points are detected along the way and located precisely.

    Args:
        equation: Eq or list of Eq objects of an autonomous system
        param: The parameter to vary (symbol or name)
        start: An equilibrium (or a point close to one) to start from: values
            of the variables and of ``param``, e.g. ``{x: -1, r: -1}``
        p_range: The branch is followed until the parameter leaves this
            interval; it starts in the direction of ``p_range[1]``
        params: Values of the other parameters
        step: Initial arclength step; 1/200 of the parameter range by default
        max_points: Maximum number of points on the branch

    Returns:
        Branch

    Examples:
        >>> x = var("x")
        >>> r = sp.Symbol("r")
        >>> branch = continuation(Eq(x.d(), r + x**2), r, {x: -1, r: -1}, (-1, 1))
        >>> branch.bifurcations   # [Bifurcation(fold at param=0, x=0)]
    """
    from .analyze import infer_orders, resolve_orders
    from .compile import compile_system
    from .equation import Eq, expand_vector_equations
    from .reduce import build_state_map
    from .symbols import t

    eqs = [equation] if isinstance(equation, Eq) else list(equation)
    eqs = expand_vector_equations(eqs)
    inferred = infer_orders(eqs)
    orders = resolve_orders(list(inferred), inferred)
    system = compile_system(eqs, orders)
    from .delay import reject_delays

    reject_delays(system.f_sym, "continuation()")
    if system.is_dae and (system.invariants or not system.algebraic_exprs):
        raise ValueError(
            "Continuation of differential-algebraic systems is only supported for "
            "index 1 with algebraic equations linear in the algebraic variables"
        )
    mapping = build_state_map(orders)
    states = system.state_syms
    name = getattr(param, "name", param)
    p_symbol = next((s for s in system.param_syms if s.name == name), None)
    if p_symbol is None:
        raise ValueError(f"Parameter {name} does not occur in the equations")
    f = substitute_params(
        system.f_sym,
        {k: v for k, v in (params or {}).items() if getattr(k, "name", k) != name},
    )
    if t.symbol in f.free_symbols:
        raise ValueError("Continuation requires an autonomous system (no explicit t)")
    missing = f.free_symbols - set(states) - {p_symbol}
    if missing:
        raise ValueError(
            f"Missing values for parameters: {', '.join(sorted(s.name for s in missing))}"
        )

    args = [*states, p_symbol]
    F = sp.lambdify(args, list(f), modules="numpy", cse=True)
    Fx = sp.lambdify(args, f.jacobian(states).tolist(), modules="numpy", cse=True)
    Fp = sp.lambdify(args, list(f.diff(p_symbol)), modules="numpy", cse=True)
    n = len(states)

    def residual(y):
        return np.asarray(F(*y), dtype=float)

    def jac_x(y):
        return np.asarray(Fx(*y), dtype=float).reshape(n, n)

    def full_jacobian(y):
        return np.column_stack([jac_x(y), np.asarray(Fp(*y), dtype=float)])

    def tangent(y, previous=None):
        _, _, vh = np.linalg.svd(full_jacobian(y))
        tau = vh[-1]
        if previous is not None and tau @ previous < 0:
            tau = -tau
        return tau

    def correct(y_pred, tau):
        """Newton on F(y) = 0, tau . (y - y_pred) = 0."""
        y = y_pred.copy()
        for _ in range(20):
            r = np.append(residual(y), tau @ (y - y_pred))
            A = np.vstack([full_jacobian(y), tau])
            try:
                dy = np.linalg.solve(A, -r)
            except np.linalg.LinAlgError:
                return None
            y = y + dy
            if np.linalg.norm(dy) < 1e-12 * (1 + np.linalg.norm(y)):
                return y
        return None

    # Starting point: Newton at the fixed parameter value
    x0 = np.zeros(n)
    for (v, level), i in mapping.items():
        if level == 0:
            value = start.get(v, start.get(v.name, 0.0))
            x0[i] = float(value)
    p0 = float(start.get(p_symbol, start.get(name)))
    for _ in range(50):
        r = residual(np.append(x0, p0))
        dx = np.linalg.solve(jac_x(np.append(x0, p0)), -r)
        x0 = x0 + dx
        if np.linalg.norm(dx) < 1e-13 * (1 + np.linalg.norm(x0)):
            break
    else:
        raise RuntimeError("Could not find an equilibrium near the starting point")

    lo, hi = sorted(p_range)
    ds = step or (hi - lo) / 200
    ds_max, ds_min = 3 * ds, 1e-6 * ds

    y = np.append(x0, p0)
    tau = tangent(y)
    if tau[-1] * (p_range[1] - p0) < 0:
        tau = -tau
    points, tangents = [y], [tau]

    while len(points) < max_points:
        y_new = correct(y + ds * tau, tau)
        if y_new is None or np.linalg.norm(y_new - y) > 3 * ds:
            ds /= 2
            if ds < ds_min:
                break
            continue
        if not lo <= y_new[-1] <= hi:
            break
        tau_new = tangent(y_new, tau)
        points.append(y_new)
        tangents.append(tau_new)
        y, tau = y_new, tau_new
        ds = min(1.3 * ds, ds_max)

    Y = np.array(points)
    T = np.array(tangents)
    eigenvalues = [np.linalg.eigvals(jac_x(yi)) for yi in Y]
    stable = np.array([np.all(ev.real < 0) for ev in eigenvalues])

    def test_functions(yi, ti):
        ev = np.linalg.eigvals(jac_x(yi))
        pair_sums = [ev[a] + ev[b] for a in range(n) for b in range(a + 1, n)]
        return {
            "fold": ti[-1],
            "det": np.prod(ev).real,
            "hopf": np.prod(pair_sums).real if pair_sums else 1.0,
        }

    bifurcations = []
    values = [test_functions(yi, ti) for yi, ti in zip(Y, T, strict=True)]
    for k in range(len(Y) - 1):
        a, b = values[k], values[k + 1]
        fold = np.sign(a["fold"]) != np.sign(b["fold"])
        if fold or np.sign(a["det"]) != np.sign(b["det"]):
            key = "fold" if fold else "det"
            y_b = _bisect(Y[k], T[k], Y[k + 1], key, correct, tangent, test_functions)
            kind = "fold" if fold else "branch point"
        elif np.sign(a["hopf"]) != np.sign(b["hopf"]):
            y_b = _bisect(
                Y[k], T[k], Y[k + 1], "hopf", correct, tangent, test_functions
            )
            ev = np.linalg.eigvals(jac_x(y_b))
            crossing = ev[np.argmin(np.abs(ev.real))]
            if abs(crossing.imag) < 1e-8:
                continue  # neutral saddle, not a Hopf bifurcation
            kind = "hopf"
        else:
            continue
        bifurcations.append(
            Bifurcation(
                kind,
                float(y_b[-1]),
                _point(y_b, mapping),
                np.linalg.eigvals(jac_x(y_b)),
            )
        )

    point_arrays = {v: Y[:, i] for (v, level), i in mapping.items() if level == 0}
    return Branch(Y[:, -1], point_arrays, stable, eigenvalues, bifurcations, p_symbol)


def _point(y, mapping) -> dict:
    return {v: float(y[i]) for (v, level), i in mapping.items() if level == 0}


def _bisect(y0, tau0, y1, key, correct, tangent, test_functions):
    """Locate a sign change of a test function between two branch points."""
    total = np.linalg.norm(y1 - y0)
    lo, hi = 0.0, total
    f_lo = test_functions(y0, tau0)[key]
    best = y1
    for _ in range(60):
        mid = (lo + hi) / 2
        y_mid = correct(y0 + mid * tau0, tau0)
        if y_mid is None:
            break
        f_mid = test_functions(y_mid, tangent(y_mid, tau0))[key]
        best = y_mid
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
        if hi - lo < 1e-12 * (1 + total):
            break
    return best
