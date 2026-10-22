Delay and stochastic equations
==============================

.. code-block:: python

   import numpy as np
   import sympy as sp
   from odecast import var, Eq, solve, solve_sde

Delay differential equations
----------------------------

In a delay differential equation the rate of change depends on earlier
values of the solution. ``y.delay(tau)`` stands for :math:`y(t - \tau)`,
and ``y.d().delay(tau)`` for :math:`y'(t - \tau)`:

.. code-block:: python

   # Delayed logistic growth (Hutchinson's equation)
   y = var("y")
   sol = solve(Eq(y.d(), 1.8*y*(1 - y.delay(1))), ivp={y: 0.5}, tspan=(0, 50))

The delay may be a parameter, e.g. ``y.delay(tau)`` with ``params={tau: 1}``,
but must be constant in time and positive.

The equation needs the solution on the interval :math:`[t_0 - \tau, t_0]`,
the *history*. By default the solution is taken to be constant at its
initial value before :math:`t_0`; ``history`` gives other values, as
numbers or functions of :math:`t`:

.. code-block:: python

   sol = solve(Eq(y.d(), -y.delay(1)), ivp={y: 1}, tspan=(0, 10),
               history={y: lambda s: np.cos(s)})

The equations are solved with the method of steps. The interval is divided
at :math:`t_0 + k\tau` for every delay :math:`\tau`, where discontinuities
of the derivatives of the solution propagate, and each piece is integrated
with :func:`scipy.integrate.solve_ivp`. Delayed values come from the history
or from the dense output of the pieces computed before.

Limitations: variables and derivatives below the highest order can be
delayed, but not the highest derivative itself (neutral equations). Delay
equations cannot be combined with events, sensitivities, boundary value
problems or algebraic equations, and the analysis functions and the SymPy
backend do not support them.

Stochastic differential equations
---------------------------------

:func:`~odecast.solve_sde` simulates equations with random forcing, in the
Itô interpretation. The equations give the deterministic part (the drift);
``noise`` adds a term :math:`g\,dW` with an independent Wiener process
:math:`W` for each entry:

.. code-block:: python

   # Geometric Brownian motion dY = mu Y dt + sigma Y dW
   y = var("y")
   sol = solve_sde(Eq(y.d(), 0.1*y), noise={y: 0.3*y}, ivp={y: 1.0},
                   tspan=(0, 1), dt=1e-3, paths=1000, seed=1)

   sol[y]          # all paths, shape (1000, len(sol.t))
   sol.mean(y)     # ensemble mean as a function of t
   sol.std(y)      # ensemble standard deviation
   sol.quantile(y, [0.05, 0.95])
   sol.plot(y)     # sample paths with mean and standard deviation

For a variable of higher order the noise enters the equation of its highest
derivative. It is given for the state whose derivative that is; for the
Langevin equation :math:`x'' + \gamma x' + \omega^2 x = \sigma\xi(t)`:

.. code-block:: python

   x = var("x")
   sol = solve_sde(Eq(x.d(2) + 0.5*x.d() + 4*x, 0), noise={x.d(): 0.4},
                   ivp={x: 0, x.d(): 0}, tspan=(0, 30), dt=1e-3, paths=500)

All paths are computed together with a fixed time step ``dt``. Two methods
are available:

``"euler"`` (default)
   Euler-Maruyama, strong order 1/2.
``"milstein"``
   Milstein, strong order 1. It uses the derivative of each noise term with
   respect to its own state, computed symbolically.

``sol.W`` contains the Wiener processes that drove the paths, so that
results can be compared with exact solutions. ``seed`` makes the
simulation reproducible.
