"""
Stochastic differential equations
"""

import numpy as np
import sympy as sp


class SDESolution:
    """
    Ensemble of sample paths of a stochastic differential equation.

    Attributes:
        t: Time points (the integration grid)
        W: Wiener processes that drove the paths, shape
            (number of noise terms, paths, len(t))
    """

    def __init__(self, t, states, mapping, W):
        self.t = t
        self._states = states  # (n_states, paths, n_t)
        self.mapping = mapping
        self.W = W

    def __getitem__(self, target) -> np.ndarray:
        """
        Sample paths of a variable or derivative.

        Returns:
            Array of shape (paths, len(t)); for vector and matrix targets
            the component axes come first
        """
        from .symbols import Derivative, VectorDerivative

        if isinstance(target, (Derivative, VectorDerivative)):
            variable, level = target.variable, target.order
        else:
            variable, level = target, 0
        components = [variable] if variable.shape is None else variable.components()
        rows = []
        for c in components:
            if (c, level) not in self.mapping:
                raise KeyError(f"{target!r} is not part of the solution")
            rows.append(self._states[self.mapping[(c, level)]])
        values = np.stack(rows)
        return (
            values[0]
            if variable.shape is None
            else values.reshape(*variable.shape, *values.shape[1:])
        )

    @property
    def paths(self) -> int:
        return self._states.shape[1]

    def mean(self, target) -> np.ndarray:
        """Ensemble mean over the paths, as a function of t."""
        return self[target].mean(axis=-2)

    def std(self, target) -> np.ndarray:
        """Ensemble standard deviation over the paths, as a function of t."""
        return self[target].std(axis=-2, ddof=1)

    def quantile(self, target, q) -> np.ndarray:
        """Ensemble quantile(s) over the paths, as a function of t."""
        return np.quantile(self[target], q, axis=-2)

    def plot(self, target, paths: int = 20, ax=None, **kwargs):
        """
        Plot sample paths, the ensemble mean, and mean +- one standard deviation.

        Args:
            target: Scalar variable or derivative
            paths: Number of sample paths to draw
            ax: Matplotlib axes; the current axes if omitted
            **kwargs: Passed to ``ax.plot`` for the sample paths

        Returns:
            The Matplotlib axes
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        values = self[target]
        kwargs.setdefault("color", "tab:blue")
        kwargs.setdefault("alpha", 0.3)
        kwargs.setdefault("linewidth", 0.8)
        for path in values[:paths]:
            ax.plot(self.t, path, **kwargs)
        mean, std = self.mean(target), self.std(target)
        ax.plot(self.t, mean, color="black", linewidth=1.5, label="mean")
        ax.fill_between(
            self.t, mean - std, mean + std, color="0.5", alpha=0.25, label="mean ± std"
        )
        ax.set_xlabel("t")
        ax.set_ylabel(repr(target))
        ax.legend(fontsize="small")
        return ax


def solve_sde(
    equation,
    noise: dict,
    ivp: dict,
    tspan: tuple[float, float],
    *,
    dt: float,
    paths: int = 1,
    method: str = "euler",
    params: dict | None = None,
    seed: int | None = None,
) -> SDESolution:
    """
    Simulate a stochastic differential equation (Ito interpretation).

    The deterministic part is given by the equations, as for
    :func:`odecast.solve`. ``noise`` adds a term g dW to the equation of a
    state, with an independent Wiener process W for each entry:

    * ``noise={y: sigma * y}`` for a first-order variable y gives
      dy = f dt + sigma y dW;
    * for a variable of higher order the noise enters the equation of its
      highest derivative: ``noise={x.d(): sigma}`` for x'' = ... gives the
      Langevin equation dx' = (...) dt + sigma dW.

    Args:
        equation: Eq or list of Eq objects (the drift)
        noise: Noise intensities g, keyed by state (variables and
            derivatives below the highest order, like ``ivp``)
        ivp: Initial values
        tspan: Time interval
        dt: Time step
        paths: Number of sample paths, simulated together
        method: "euler" (Euler-Maruyama, strong order 1/2) or "milstein"
            (strong order 1; uses dg/dx of each noise term with respect to
            its own state)
        params: Parameter values
        seed: Seed for the random number generator

    Returns:
        SDESolution

    Examples:
        >>> y = var("y")
        >>> sol = solve_sde(Eq(y.d(), 0.1*y), noise={y: 0.2*y}, ivp={y: 1.0},
        ...                 tspan=(0, 1), dt=1e-3, paths=1000, seed=1)
        >>> sol.mean(y)[-1]   # close to exp(0.1)
    """
    from .analyze import infer_orders, resolve_orders
    from .api import _bind_conditions
    from .backends.scipy_ivp import convert_ivp_to_state_vector
    from .compile import compile_system
    from .delay import reject_delays
    from .equation import Eq, expand_vector_equations
    from .reduce import _substitute_derivatives_with_states, build_state_map
    from .symbols import as_sympy, substitute_params
    from .validate import normalize_ivp, validate_ivp

    if method not in ("euler", "milstein"):
        raise ValueError(f"Unknown method {method!r}; use 'euler' or 'milstein'")

    eqs = [equation] if isinstance(equation, Eq) else list(equation)
    eqs = expand_vector_equations(eqs)
    inferred = infer_orders(eqs)
    orders = resolve_orders(list(inferred), inferred)
    system = compile_system(eqs, orders)
    reject_delays(system.f_sym, "solve_sde()")
    if system.invariants:
        raise ValueError("solve_sde() does not support constraints of higher index")
    mapping = build_state_map(orders)
    states = system.state_syms
    t_symbol = system.t_symbol

    ivp, _ = _bind_conditions(ivp, None, params)
    validate_ivp(orders, ivp, tspan[0])
    x0 = convert_ivp_to_state_vector(ivp, mapping)

    algebraic = {
        v.sympy(): system.algebraic_exprs[v.name]
        for v in orders
        if v.name in system.algebraic_exprs
    }
    rows, intensities = [], []
    for (v, level), value in normalize_ivp(noise).items():
        if (v, level) not in mapping:
            raise ValueError(
                f"Noise for {v.name}^({level}): noise can only be added to states "
                "(variables and derivatives below the highest order)"
            )
        if level != orders[v] - 1:
            raise ValueError(
                f"Noise for {v.name}^({level}) must be given for the state whose "
                f"derivative is the highest derivative, {v.name}^({orders[v] - 1})"
            )
        g = as_sympy(value).xreplace(algebraic)
        g = substitute_params(
            _substitute_derivatives_with_states(g, mapping, states), params
        )
        unknown = g.free_symbols - set(states) - {t_symbol}
        if unknown:
            names = ", ".join(sorted(s.name for s in unknown))
            raise ValueError(f"Missing values for parameters in noise: {names}")
        rows.append(mapping[(v, level)])
        intensities.append(g)

    args = [t_symbol, *states]
    g_num = sp.lambdify(args, intensities, modules="numpy", cse=True)
    dg_num = sp.lambdify(
        args,
        [sp.diff(g, states[r]) for g, r in zip(intensities, rows, strict=True)],
        modules="numpy",
        cse=True,
    )
    f = system.rhs_vectorized(params)

    t0, t1 = map(float, tspan)
    steps = int(np.ceil((t1 - t0) / dt - 1e-12))
    t = np.linspace(t0, t1, steps + 1)
    h = np.diff(t)
    rng = np.random.default_rng(seed)

    n, m = len(states), len(rows)
    X = np.empty((n, paths, steps + 1))
    X[:, :, 0] = x0[:, None]
    W = np.zeros((m, paths, steps + 1))
    dW = rng.standard_normal((m, paths, steps)) * np.sqrt(h)
    np.cumsum(dW, axis=2, out=W[:, :, 1:])

    def broadcast(values, shape):
        return np.array([np.broadcast_to(v, shape) for v in values], dtype=float)

    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(steps):
            x = X[:, :, k]
            tk = np.full(paths, t[k])
            drift = f(tk, x)
            g = broadcast(g_num(t[k], *x), (paths,))
            step = x + drift * h[k]
            step[rows] += g * dW[:, :, k]
            if method == "milstein":
                dg = broadcast(dg_num(t[k], *x), (paths,))
                step[rows] += 0.5 * g * dg * (dW[:, :, k] ** 2 - h[k])
            X[:, :, k + 1] = step

    return SDESolution(t, X, mapping, W)
