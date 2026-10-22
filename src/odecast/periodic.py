"""
Periodic orbits of autonomous systems
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

from .solution import SolutionBVP


class PeriodicOrbit:
    """
    A periodic orbit, as returned by :func:`odecast.periodic_orbit`.

    Attributes:
        period: The period T
        solution: Solution over one period, t in [0, T]; supports ``sol[y]``,
            ``sol.eval(y, t)`` and ``sol.plot()`` like other solutions
        multipliers: Floquet multipliers, the eigenvalues of the monodromy
            matrix. One of them is 1 (perturbations along the orbit).
        stability: "stable" if all other multipliers lie inside the unit
            circle, "unstable" if one lies outside, "marginal" otherwise
    """

    def __init__(self, period, solution, multipliers, stability):
        self.period = period
        self.solution = solution
        self.multipliers = multipliers
        self.stability = stability

    def __repr__(self):
        return f"PeriodicOrbit(period={self.period:.6g}, {self.stability})"


def periodic_orbit(
    equation,
    ivp: dict,
    period: float,
    *,
    params: dict | None = None,
    transient: float = 0.0,
    mesh: int = 100,
    tol: float = 1e-6,
):
    """
    Find a periodic orbit (limit cycle) of an autonomous system.

    A trajectory is computed from ``ivp`` for ``transient + period`` time
    units. Its last stretch of length ``period`` is the starting guess for a
    boundary value problem in which the period is unknown: x(0) = x(T), with
    the phase fixed by requiring that the derivative of the most strongly
    varying state vanishes at t = 0.

    Args:
        equation: Eq or list of Eq objects
        ivp: Initial values for the starting trajectory
        period: Estimate of the period
        params: Parameter values
        transient: Time to integrate before the guess is taken, so that the
            trajectory approaches the orbit
        mesh: Number of mesh nodes for the boundary value problem
        tol: Tolerance for scipy.integrate.solve_bvp

    Returns:
        PeriodicOrbit

    Raises:
        ValueError: If the system depends explicitly on t
        RuntimeError: If the boundary value problem does not converge

    Examples:
        >>> y = var("y")
        >>> orbit = periodic_orbit(Eq(y.d(2) - (1 - y**2)*y.d() + y, 0),
        ...                        ivp={y: 2, y.d(): 0}, period=6.5, transient=20)
        >>> orbit.period      # 6.6633
        >>> orbit.stability   # "stable"
    """
    from .analyze import infer_orders, resolve_orders
    from .api import solve
    from .compile import compile_system
    from .equation import Eq, expand_vector_equations
    from .reduce import build_state_map
    from .symbols import t

    eqs = [equation] if isinstance(equation, Eq) else list(equation)
    expanded = expand_vector_equations(eqs)
    inferred = infer_orders(expanded)
    orders = resolve_orders(list(inferred), inferred)
    system = compile_system(expanded, orders)
    from .delay import reject_delays

    reject_delays(system.f_sym, "periodic_orbit()")
    if t.symbol in system.f_sym.free_symbols:
        raise ValueError("Periodic orbits are only computed for autonomous systems")
    mapping = build_state_map(orders)
    n = len(system.state_syms)

    # Starting guess: one period of a trajectory, beginning at a maximum of
    # the state that varies most
    start = solve(
        eqs,
        ivp=ivp,
        tspan=(0.0, transient + period),
        params=params,
        rtol=1e-8,
        atol=1e-10,
    )
    t_guess = np.linspace(transient, transient + period, 4 * mesh)
    x_guess = start._interpolant(t_guess)[:n]
    k = int(np.argmax(np.ptp(x_guess, axis=1)))
    t0 = t_guess[int(np.argmax(x_guess[k]))]
    tau = np.linspace(0.0, 1.0, mesh)
    y_guess = start._interpolant(np.minimum(t0 + tau * period, start.t[-1]))[:n]
    if t0 + period > start.t[-1]:  # extend by one period, assuming periodicity
        late = t0 + tau * period > start.t[-1]
        y_guess[:, late] = start._interpolant(t0 + tau[late] * period - period)[:n]

    f = system.rhs_vectorized(params)

    def fun(tau, y, p):
        return p[0] * f(tau * p[0], y)

    def bc(ya, yb, p):
        derivative = f(np.zeros(1), ya[:, None])[k, 0]
        return np.concatenate([ya - yb, [derivative]])

    result = solve_bvp(fun, bc, tau, y_guess, p=[period], tol=tol, max_nodes=100000)
    if not result.success:
        raise RuntimeError(f"Periodic orbit not found: {result.message}")
    T = float(result.p[0])

    def interpolant(times):
        return result.sol(np.asarray(times, dtype=float) / T)

    solution = SolutionBVP(
        t=result.x * T,
        y=result.y,
        mapping=mapping,
        interpolant=interpolant,
        f_compiled=system.rhs(params),
    )

    multipliers = _floquet_multipliers(system, params, result.y[:, 0], T)
    others = np.delete(multipliers, np.argmin(np.abs(multipliers - 1.0)))
    if len(others) == 0 or np.all(np.abs(others) < 1 - 1e-6):
        stability = "stable"
    elif np.any(np.abs(others) > 1 + 1e-6):
        stability = "unstable"
    else:
        stability = "marginal"
    return PeriodicOrbit(T, solution, multipliers, stability)


def _floquet_multipliers(system, params, x0, period) -> np.ndarray:
    """Eigenvalues of the monodromy matrix, from the variational equation."""
    n = len(x0)
    f, jac = system.rhs(params), system.jac(params)

    def rhs(t, y):
        x, phi = y[:n], y[n:].reshape(n, n)
        return np.concatenate([f(t, x), (jac(t, x) @ phi).ravel()])

    y0 = np.concatenate([x0, np.eye(n).ravel()])
    sol = solve_ivp(rhs, (0.0, period), y0, rtol=1e-10, atol=1e-12)
    monodromy = sol.y[n:, -1].reshape(n, n)
    return np.linalg.eigvals(monodromy)
