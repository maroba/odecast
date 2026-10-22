"""
Delay differential equations, solved with the method of steps
"""

import numpy as np
import sympy as sp
from scipy.integrate import OdeSolution, solve_ivp
from sympy.core.function import AppliedUndef

from .solution import SolutionIVP
from .symbols import delayed_target, substitute_params


def find_delays(expr) -> list[tuple]:
    """Delayed values in an expression: (function, variable, level) tuples."""
    found = []
    for app in sorted(expr.atoms(AppliedUndef), key=str):
        target = delayed_target(app.func.__name__)
        if target is not None:
            found.append((app, *target))
    return found


def reject_delays(expr, what: str) -> None:
    """Raise if an expression contains delayed values."""
    if find_delays(expr):
        raise ValueError(f"{what} does not support delay differential equations")


def solve_dde(system, mapping, x0, tspan, params, history, options) -> SolutionIVP:
    """
    Solve a system with constant delays.

    The interval is split at t0 + k*tau for every delay tau, where
    discontinuities of the derivatives propagate. Each piece is integrated
    with scipy.integrate.solve_ivp; delayed values come from the history
    (t <= t0) or from the dense output of the pieces already computed.

    Args:
        system: CompiledSystem whose right-hand side contains delayed values
        mapping: State mapping
        x0: Initial state
        tspan: Integration interval
        params: Parameter values
        history: Values for t <= t0, keyed like ``ivp``; numbers or functions
            of t. Unspecified states keep their initial value.
        options: Keyword arguments for scipy.integrate.solve_ivp
    """
    from .validate import normalize_ivp

    t_symbol = system.t_symbol
    t0, t_end = map(float, tspan)
    delays = find_delays(system.f_sym)

    lags = []
    for app, variable, level in delays:
        if (variable, level) not in mapping:
            raise ValueError(
                f"Delayed values of {variable.name}^({level}) are not supported: only "
                "variables and derivatives below the highest order can be delayed "
                "(neutral delay equations are not supported)"
            )
        tau = substitute_params(t_symbol - app.args[0], params)
        if tau.free_symbols:
            raise ValueError(
                f"Delays must be constant; got {tau}. Give all parameters a value."
            )
        if float(tau) <= 0:
            raise ValueError(f"Delays must be positive, got {float(tau)}")
        lags.append(float(tau))

    placeholders = [sp.Symbol(f"_odecast_delayed{i}") for i in range(len(delays))]
    f = system.f_sym.xreplace(
        {app: d for (app, _, _), d in zip(delays, placeholders, strict=True)}
    )
    args = [t_symbol, *system.state_syms, *placeholders, *system.param_syms]
    f_num = sp.lambdify(args, list(f), modules="numpy", cse=True)
    p = system.param_values(params)

    given = normalize_ivp(history or {})
    initial = {key: x0[i] for key, i in mapping.items()}

    def history_value(key, s):
        value = given.get(key, initial[key])
        return float(value(s)) if callable(value) else float(value)

    rows = [mapping[(variable, level)] for _, variable, level in delays]
    keys = [(variable, level) for _, variable, level in delays]
    segments, ends = [], []

    def delayed_values(t):
        values = []
        for key, row, lag in zip(keys, rows, lags, strict=True):
            s = t - lag
            if s <= t0 or not segments:
                values.append(history_value(key, min(s, t0)))
            else:
                k = min(int(np.searchsorted(ends, s)), len(segments) - 1)
                values.append(segments[k](s)[row])
        return values

    def rhs(t, x):
        return np.array(f_num(t, *x, *delayed_values(t), *p), dtype=float)

    breakpoints = {t_end}
    for lag in set(lags):
        breakpoints.update(t0 + k * lag for k in range(1, int((t_end - t0) / lag) + 1))
    breakpoints = sorted(b for b in breakpoints if t0 < b <= t_end)

    options = dict(options)
    options.setdefault("rtol", 1e-3)
    options.setdefault("atol", 1e-6)
    options["dense_output"] = True
    t_eval = options.pop("t_eval", None)

    times, states = [], []
    a, x = t0, np.asarray(x0, dtype=float)
    for b in breakpoints:
        if t_eval is not None:
            t_eval = np.asarray(t_eval, dtype=float)
            options["t_eval"] = t_eval[(t_eval >= a) & (t_eval <= b)]
        sol = solve_ivp(rhs, (a, b), x, **options)
        if not sol.success:
            raise RuntimeError(f"SciPy solve_ivp failed: {sol.message}")
        seg_t = np.asarray(sol.t, dtype=float)
        seg_y = np.asarray(sol.y, dtype=float).reshape(len(x), -1)
        last = next((ts[-1] for ts in reversed(times) if len(ts)), None)
        skip = 1 if len(seg_t) and seg_t[0] == last else 0
        times.append(seg_t[skip:])
        states.append(seg_y[:, skip:])
        segments.append(sol.sol)
        ends.append(b)
        a, x = b, sol.sol(b)

    interpolant = OdeSolution(
        np.concatenate([segments[0].ts] + [s.ts[1:] for s in segments[1:]]),
        [i for s in segments for i in s.interpolants],
    )
    return SolutionIVP(
        t=np.concatenate(times),
        y=np.concatenate(states, axis=1),
        mapping=mapping,
        t0=t0,
        f_compiled=rhs,
        x0=np.asarray(x0, dtype=float),
        interpolant=interpolant,
    )
