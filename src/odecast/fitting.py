"""
Estimation of parameters from measured data
"""

import numpy as np
from scipy.optimize import least_squares


class FitResult:
    """
    Result of :func:`odecast.fit`.

    Attributes:
        params: Fitted parameter values, keyed by symbol
        stderr: Standard errors of the fitted parameters
        covariance: Covariance matrix, ordered like ``params``
        solution: Solution of the system with the fitted parameters
        cost: Half the sum of squared (weighted) residuals
        success: Whether the optimizer converged
        message: Message from the optimizer
        result: The full scipy.optimize.OptimizeResult
    """

    def __init__(self, params, stderr, covariance, solution, result):
        self.params = params
        self.stderr = stderr
        self.covariance = covariance
        self.solution = solution
        self.cost = result.cost
        self.success = result.success
        self.message = result.message
        self.result = result

    def __repr__(self):
        lines = [
            f"  {s} = {v:.6g} ± {self.stderr[s]:.2g}" for s, v in self.params.items()
        ]
        return "FitResult(\n" + "\n".join(lines) + f"\n  cost={self.cost:.6g})"


def fit(
    equation,
    data: dict,
    guess: dict,
    *,
    ivp: dict,
    tspan: tuple[float, float] | None = None,
    params: dict | None = None,
    bounds: dict | None = None,
    sigma: dict | None = None,
    **solve_options,
) -> FitResult:
    """
    Fit parameters of an initial value problem to measured data.

    The parameters are SymPy symbols in the equations and, optionally, in the
    initial conditions. They are determined by nonlinear least squares
    (:func:`scipy.optimize.least_squares`).

    Args:
        equation: Eq or list of Eq objects
        data: Measurements as ``{target: (times, values)}``, where target is
            a variable or derivative. Several targets may be given, each
            with its own time points.
        guess: Parameters to fit with their starting values,
            e.g. ``{k: 1.0}``
        ivp: Initial conditions, as for :func:`odecast.solve`. Values may be
            parameters, e.g. ``{y: y0}`` with ``y0`` in ``guess``.
        tspan: Integration interval; by default from the earliest to the
            latest measurement. Initial conditions apply at ``tspan[0]``.
        params: Values of parameters that are not fitted
        bounds: Optional ``{symbol: (lower, upper)}``
        sigma: Optional measurement uncertainties ``{target: sigma}``
            (scalar or array); residuals are divided by them
        **solve_options: Passed to :func:`odecast.solve`. The default
            tolerances are tightened to rtol=1e-8, atol=1e-10, since solver
            noise disturbs the optimization.

    Returns:
        FitResult

    Examples:
        >>> k = sp.Symbol("k")
        >>> y = var("y")
        >>> result = fit(Eq(y.d(), -k*y), {y: (t_data, y_data)},
        ...              guess={k: 1.0}, ivp={y: 1.0})
        >>> result.params[k]
    """
    from .api import solve

    symbols = list(guess)
    x0 = np.array([float(guess[s]) for s in symbols])
    fixed = dict(params or {})
    sigma = sigma or {}

    series = []
    for target, (times, values) in data.items():
        times = np.asarray(times, dtype=float)
        values = np.asarray(values, dtype=float)
        series.append((target, times, values, np.asarray(sigma.get(target, 1.0))))
    if tspan is None:
        tspan = (
            min(s[1].min() for s in series),
            max(s[1].max() for s in series),
        )

    options = {"rtol": 1e-8, "atol": 1e-10, **solve_options}

    def solve_for(p):
        return solve(
            equation,
            ivp=ivp,
            tspan=tspan,
            params={**fixed, **dict(zip(symbols, p, strict=True))},
            backend="scipy",
            **options,
        )

    def residuals(p):
        sol = solve_for(p)
        return np.concatenate(
            [
                np.ravel((sol.eval(target, times) - values) / s)
                for target, times, values, s in series
            ]
        )

    lower = [-np.inf] * len(symbols)
    upper = [np.inf] * len(symbols)
    for i, s in enumerate(symbols):
        if bounds and s in bounds:
            lower[i], upper[i] = bounds[s]

    result = least_squares(residuals, x0, bounds=(lower, upper))

    n_data, n_params = result.fun.size, len(symbols)
    jac = result.jac
    dof = max(n_data - n_params, 1)
    covariance = np.linalg.pinv(jac.T @ jac) * (2 * result.cost / dof)
    stderr = np.sqrt(np.diag(covariance))

    return FitResult(
        params={s: float(v) for s, v in zip(symbols, result.x, strict=True)},
        stderr={s: float(e) for s, e in zip(symbols, stderr, strict=True)},
        covariance=covariance,
        solution=solve_for(result.x),
        result=result,
    )
