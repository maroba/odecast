"""
Phase portraits of planar autonomous systems
"""

import numpy as np

_MARKERS = {
    "stable": {"marker": "o", "markerfacecolor": "black"},
    "unstable": {"marker": "o", "markerfacecolor": "white"},
    "saddle": {"marker": "X", "markerfacecolor": "black"},
}


def phase_portrait(
    equation,
    x=None,
    y=None,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    params: dict | None = None,
    trajectories=(),
    tspan: tuple[float, float] = (0.0, 20.0),
    density: int = 25,
    nullclines: bool = True,
    equilibria: bool = True,
    ax=None,
):
    """
    Draw the phase portrait of a system with two states.

    The system must be autonomous and have exactly two first-order states,
    e.g. two first-order equations, or one second-order equation (states y
    and y').

    Args:
        equation: Eq or list of Eq objects
        x, y: The states on the horizontal and vertical axis, e.g. ``theta``
            and ``theta.d()``; the two states in order if omitted
        xlim, ylim: Plot ranges
        params: Parameter values
        trajectories: Initial values (dicts as for ``ivp``) of trajectories
            to draw over ``tspan``
        tspan: Time span of the trajectories
        density: Grid points per axis for the direction field and nullclines
        nullclines: Draw the curves on which x' = 0 and y' = 0
        equilibria: Mark the equilibria found by :func:`odecast.equilibria`,
            filled for stable, open for unstable, crosses for saddles. For
            periodic nonlinearities only representatives are found (e.g.
            0 and pi for sin).
        ax: Matplotlib axes; the current axes if omitted

    Returns:
        The Matplotlib axes

    Examples:
        >>> theta = var("theta")
        >>> phase_portrait(Eq(theta.d(2) + 0.3*theta.d() + sp.sin(theta), 0),
        ...                xlim=(-7, 7), ylim=(-3, 3))
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .analyze import infer_orders, resolve_orders
    from .api import solve
    from .compile import compile_system
    from .equation import Eq, expand_vector_equations
    from .reduce import build_state_map
    from .stability import equilibria as find_equilibria
    from .symbols import t

    eqs = [equation] if isinstance(equation, Eq) else list(equation)
    expanded = expand_vector_equations(eqs)
    inferred = infer_orders(expanded)
    orders = resolve_orders(list(inferred), inferred)
    system = compile_system(expanded, orders)
    from .delay import reject_delays

    reject_delays(system.f_sym, "phase_portrait()")
    mapping = build_state_map(orders)
    if len(mapping) != 2:
        raise ValueError(
            f"A phase portrait needs a system with two states, this one has {len(mapping)}"
        )
    if t.symbol in system.f_sym.free_symbols:
        raise ValueError("A phase portrait needs an autonomous system (no explicit t)")

    def index(target):
        variable = target.variable if hasattr(target, "variable") else target
        level = getattr(target, "order", 0) if hasattr(target, "variable") else 0
        if (variable, level) not in mapping:
            raise ValueError(f"{target!r} is not a state of the system")
        return mapping[(variable, level)]

    states = sorted(mapping, key=mapping.get)
    labels = [v.name + "'" * level for v, level in states]
    i = index(x) if x is not None else 0
    j = index(y) if y is not None else 1 - i
    if i == j:
        raise ValueError("x and y must be different states")

    ax = ax or plt.gca()
    gx = np.linspace(*xlim, density)
    gy = np.linspace(*ylim, density)
    X, Y = np.meshgrid(gx, gy)
    points = np.zeros((2, X.size))
    points[i], points[j] = X.ravel(), Y.ravel()
    derivatives = system.rhs_vectorized(params)(np.zeros(X.size), points)
    U, V = derivatives[i].reshape(X.shape), derivatives[j].reshape(X.shape)

    ax.streamplot(X, Y, U, V, color="0.65", linewidth=0.8, density=1.2)
    handles = []
    if nullclines:
        # A finer grid than for the direction field gives smooth curves
        fx, fy = np.meshgrid(np.linspace(*xlim, 300), np.linspace(*ylim, 300))
        fine = np.zeros((2, fx.size))
        fine[i], fine[j] = fx.ravel(), fy.ravel()
        values = system.rhs_vectorized(params)(np.zeros(fx.size), fine)
        for k, color in ((i, "tab:blue"), (j, "tab:orange")):
            grid = values[k].reshape(fx.shape)
            if np.nanmin(grid) < 0 < np.nanmax(grid):
                ax.contour(fx, fy, grid, levels=[0], colors=color, linewidths=1.5)
                handles.append(Line2D([], [], color=color, label=f"{labels[k]}' = 0"))

    for ivp in trajectories:
        sol = solve(eqs, ivp=ivp, tspan=tspan, params=params, rtol=1e-8, atol=1e-10)
        times = np.linspace(*tspan, 2000)
        states_t = sol._interpolant(times)
        ax.plot(states_t[i], states_t[j], color="tab:red", linewidth=1.2)

    if equilibria:
        seen = set()
        for e in find_equilibria(eqs, params):
            point = np.zeros(2)
            for (v, level), k in mapping.items():
                if level == 0 and v in e.point:
                    point[k] = float(e.point[v])
            px, py = point[i], point[j]
            if not (xlim[0] <= px <= xlim[1] and ylim[0] <= py <= ylim[1]):
                continue
            kind = next(
                (k for k in ("saddle", "unstable", "stable") if k in e.stability), None
            )
            style = _MARKERS.get(kind, {"marker": "s", "markerfacecolor": "0.5"})
            ax.plot(px, py, linestyle="none", markersize=8, color="black", **style)
            label = e.stability if kind is None else kind
            if label not in seen:
                seen.add(label)
                handles.append(
                    Line2D(
                        [],
                        [],
                        linestyle="none",
                        color="black",
                        markersize=8,
                        label=label,
                        **style,
                    )
                )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(labels[i])
    ax.set_ylabel(labels[j])
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize="small")
    return ax
