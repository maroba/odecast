Analysis
========

Besides solving, odecast provides tools to study the qualitative behaviour
of a system: its equilibria and their stability, linearizations, phase
portraits, bifurcations, periodic orbits, and the estimation of parameters
from data. The examples use

.. code-block:: python

   import numpy as np
   import sympy as sp
   from odecast import var, Eq, solve

Equilibria and stability
------------------------

:func:`~odecast.equilibria` finds the points where all derivatives of an
autonomous system vanish and classifies them by the eigenvalues of the
Jacobian:

.. code-block:: python

   from odecast import equilibria

   x, y = var("x"), var("y")
   equilibria([Eq(x.d(), x - x*y), Eq(y.d(), x*y - y)])
   # [Equilibrium(x=0, y=0, saddle), Equilibrium(x=1, y=1, center)]

   theta = var("theta")
   equilibria(Eq(theta.d(2) + 0.3*theta.d() + sp.sin(theta), 0))
   # [Equilibrium(theta=0.0, stable focus), Equilibrium(theta=3.14159..., saddle)]

Each result has the attributes ``point`` (values of the variables),
``jacobian``, ``eigenvalues`` and ``stability``. The classification is
``"stable"`` if all eigenvalues have negative real part, ``"unstable"`` if
one has positive real part, and ``"marginal"`` otherwise, in which case the
linearization does not decide stability. Two-dimensional systems are
further classified as node, focus, saddle or center.

The equilibria are found with :func:`sympy.solve`. For periodic
nonlinearities such as ``sin`` it returns representatives only (here
:math:`0` and :math:`\pi`). Parameters may be given values with ``params``;
otherwise the Jacobian and eigenvalues stay symbolic and the stability is
``"unknown"``.

Linearization
-------------

:func:`~odecast.linearize` linearizes a system around an operating point
(usually an equilibrium) into a state-space model

.. math::

   \dot x = A x + B u, \qquad y = C x + D u,

where :math:`x`, :math:`u` and :math:`y` are deviations of the states,
inputs and outputs from the operating point. Inputs are SymPy symbols in the
equations; outputs are expressions in the variables.

.. code-block:: python

   from odecast import linearize

   x = var("x")
   F = sp.Symbol("F")
   m, c, k = sp.symbols("m c k", positive=True)

   lin = linearize(Eq(m*x.d(2) + c*x.d() + k*x, F), around={x: 0},
                   inputs=[F], outputs=[x])
   lin.A                    # Matrix([[0, 1], [-k/m, -c/m]])
   lin.transfer_function()  # Matrix([[1/(c*s + k + m*s**2)]])

``around`` may also be an equilibrium returned by
:func:`~odecast.equilibria`. Derivatives and inputs that are not given are
zero at the operating point, which must be an equilibrium. The matrices are
SymPy matrices; with numeric values, ``lin.to_scipy()`` returns a
:class:`scipy.signal.StateSpace` for use with the control functions of
SciPy, and ``lin.eigenvalues()`` gives the poles.

Phase portraits
---------------

For systems with two states (two first-order equations, or one of second
order), :func:`~odecast.phase_portrait` draws the direction field, the
nullclines, the equilibria with their stability, and optionally
trajectories:

.. code-block:: python

   from odecast import phase_portrait

   theta = var("theta")
   phase_portrait(
       Eq(theta.d(2) + 0.3*theta.d() + sp.sin(theta), 0),
       xlim=(-4, 7), ylim=(-3, 3),
       trajectories=[{theta: -np.pi + 0.01, theta.d(): 0}],
   )

The axes default to the states in order (here :math:`\theta` and
:math:`\dot\theta`); ``x=`` and ``y=`` select them explicitly.

Continuation and bifurcations
-----------------------------

:func:`~odecast.continuation` follows a branch of equilibria while a
parameter varies, and detects where the equilibria change their stability:

.. code-block:: python

   from odecast import continuation

   x = var("x")
   r = sp.Symbol("r")
   branch = continuation(Eq(x.d(), r + x - x**3/3), r, start={x: -2.5, r: -3},
                         p_range=(-3, 3))
   branch.bifurcations
   # [Bifurcation(fold at param=0.666667, x=-1),
   #  Bifurcation(fold at param=-0.666667, x=1)]
   branch.plot(x)   # stable parts solid, unstable parts dashed

The continuation follows the branch around folds, where the parameter turns
back (pseudo-arclength continuation). Three kinds of bifurcations are
detected and located precisely:

``"fold"``
   Saddle-node bifurcation: two equilibria merge and disappear.
``"hopf"``
   A pair of complex eigenvalues crosses the imaginary axis; a periodic
   orbit appears or disappears.
``"branch point"``
   A real eigenvalue crosses zero while another branch crosses this one, as
   in transcritical and pitchfork bifurcations. The other branch can be
   followed by starting a new continuation on it.

The branch is available as ``branch.param``, ``branch[x]``,
``branch.stable`` and ``branch.eigenvalues``.

Periodic orbits
---------------

:func:`~odecast.periodic_orbit` computes a periodic orbit (limit cycle) and
its period, starting from an estimate of the period and an initial point
near the orbit:

.. code-block:: python

   from odecast import periodic_orbit

   v = var("v")
   orbit = periodic_orbit(Eq(v.d(2) - (1 - v**2)*v.d() + v, 0),
                          ivp={v: 2, v.d(): 0}, period=6.5, transient=20)
   orbit.period        # 6.6633
   orbit.stability     # "stable"
   orbit.solution.plot(v)

A trajectory is integrated for ``transient`` time units to approach the
orbit; one period of it is the starting guess for a boundary value problem
in which the period is unknown.

The stability follows from the Floquet multipliers ``orbit.multipliers``,
the eigenvalues of the linearized map over one period. One of them is 1; the
orbit is stable if all others lie inside the unit circle. They are computed
from the variational equation with the exact Jacobian.

Fitting parameters to data
--------------------------

:func:`~odecast.fit` determines parameters of an initial value problem from
measurements by nonlinear least squares:

.. code-block:: python

   from odecast import fit

   y = var("y")
   c, k, y0 = sp.symbols("c k y0")
   eq = Eq(y.d(2) + c*y.d() + k*y, 0)

   result = fit(
       eq,
       {y: (t_data, y_data)},          # measurements of y
       guess={c: 1.0, k: 2.0, y0: 1.0},
       ivp={y: y0, y.d(): 0},           # the initial value is fitted too
   )
   result.params     # {c: 0.40..., k: 3.00..., y0: 1.50...}
   result.stderr     # standard errors
   result.solution   # solution with the fitted parameters

Measurements of several variables or derivatives may be given, each on its
own time grid. ``params`` fixes parameters that are not fitted, ``bounds``
restricts the fitted ones, and ``sigma`` weights the residuals by the
measurement uncertainties. The standard errors are computed from the
Jacobian of the residuals, as in :func:`scipy.optimize.curve_fit`.

The derivatives of the residuals with respect to the parameters are computed
exactly from the sensitivities of the solution (see the
:doc:`user_guide`), which makes the optimization fast and reliable. Trial
parameters for which the integration fails are rejected as bad steps.
Like any local optimization, the fit can end in a local minimum, in
particular for oscillating solutions; good starting values matter.

