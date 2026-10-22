User Guide
==========

Systems of equations
--------------------

A system is passed to :func:`~odecast.solve` as a list of equations. The
variables need not have the same order; odecast determines the highest
derivative of each variable separately and builds the first-order system
accordingly.

.. code-block:: python

   from odecast import var, Eq, solve

   x, z = var("x"), var("z")
   eqs = [
       Eq(x.d(2) + z, 0),   # second order in x
       Eq(z.d() - x, 0),    # first order in z
   ]
   sol = solve(eqs, ivp={x: 1.0, x.d(): 0.0, z: 0.0}, tspan=(0, 5))

Each equation must be solvable for the highest derivative of one variable.

Declaring the order
~~~~~~~~~~~~~~~~~~~

The order of a variable may be declared explicitly. This is optional and
serves as a consistency check: if an equation uses a higher derivative, an
:class:`~odecast.errors.OrderMismatchError` is raised.

.. code-block:: python

   w = var("w", order=1)

Vector and matrix variables
---------------------------

A variable with ``shape`` represents a vector- or matrix-valued function.
Its components are scalar variables and can be used anywhere a scalar
variable can:

.. code-block:: python

   u = var("u", shape=2)
   u[0], u[1]          # components
   u.d(), u.d(2)       # vector derivatives

   M = var("M", shape=(2, 3))
   M[1, 2]             # component in row 1, column 2

An equation involving a vector variable as a whole is expanded into one
equation per component. The following are equivalent:

.. code-block:: python

   Eq(u.d(2) + u, 0)

   [Eq(u[0].d(2) + u[0], 0),
    Eq(u[1].d(2) + u[1], 0)]

Arithmetic operations are element-wise, and all array-valued terms in one
equation must have the same shape. Scalars and scalar variables are
broadcast. Functions such as ``sin`` must be applied to components,
``sp.sin(u[0])``, not to the variable as a whole.

Matrix products are written with ``@``. The matrix may be a NumPy array, a
nested list or a SymPy matrix, and may contain parameters (see
`Parameters`_):

.. code-block:: python

   A = np.array([[0.0, 1.0], [-2.0, -0.3]])
   eq = Eq(u.d(), A @ u)                        # u' = A u

   K = np.array([[2, -1], [-1, 2]])
   eq = Eq(u.d(2) + K @ u, 0)                   # coupled oscillators

   M = var("M", shape=(2, 2))
   eq = Eq(M.d(), A @ M)                        # matrix equation M' = A M

Initial conditions for array-valued variables are given as (nested)
sequences or NumPy arrays of matching shape:

.. code-block:: python

   sol = solve(
       Eq(u.d(2) + u, 0),
       ivp={u: [1.0, 0.5], u.d(): [0.0, -0.2]},
       tspan=(0, 10),
   )

   sol[u]      # shape (2, n)
   sol[u[0]]   # shape (n,)

For a matrix variable, ``sol[M]`` has shape ``(2, 3, n)``.

Scalar and vector variables can be mixed in one system:

.. code-block:: python

   x = var("x")
   u = var("u", shape=2)

   eqs = [
       Eq(x.d(2) + x - u[0], 0),
       Eq(u[0].d() + u[1], x),
       Eq(u[1].d() + u[0], 0),
   ]
   sol = solve(eqs, ivp={x: 1.0, x.d(): 0.0, u: [0.5, 0.0]}, tspan=(0, 5))

Nonlinear and time-dependent terms
----------------------------------

Equations may contain powers and functions of the unknowns. Both SymPy and
NumPy functions can be used:

.. code-block:: python

   import numpy as np
   import sympy as sp

   # Van der Pol oscillator: y'' - mu (1 - y^2) y' + y = 0
   y = var("y")
   mu = 1.0
   eq = Eq(y.d(2) - mu*(1 - y**2)*y.d() + y, 0)

   # Pendulum
   theta = var("theta")
   eq = Eq(theta.d(2) + sp.sin(theta), 0)   # or np.sin(theta)

The independent variable is exported as :data:`odecast.t` and can be used in
the same way:

.. code-block:: python

   from odecast import t

   eq = Eq(y.d(2) + y, sp.cos(2*t))   # driven oscillator
   sol = solve(eq, ivp={y: 0.0, y.d(): 1.0}, tspan=(0, 10))

The SymPy symbol for :math:`t` is ``t.symbol``, which is useful when working
with the expressions returned by the SymPy backend.

Each equation must be solvable for the highest derivative of one of the
variables; implicit equations such as ``sp.sin(y.d()) = y`` are solved with
:func:`sympy.solve`, and the first solution is used.

Parameters
----------

SymPy symbols in the equations are parameters. Their values are passed to
:func:`~odecast.solve` with ``params``, keyed by symbol or by name:

.. code-block:: python

   k, c = sp.symbols("k c")
   eq = Eq(y.d(2) + c*y.d() + k*y, 0)

   sol = solve(eq, ivp={y: 1, y.d(): 0}, tspan=(0, 10), params={k: 4.0, c: 0.1})

Initial and boundary values may contain parameters as well, e.g.
``ivp={y: y0, y.d(): 0}`` with ``params={y0: 2.0, ...}``.

The equations are reduced and compiled once; solving them again with other
parameter values reuses the compiled system. This makes parameter studies
cheap:

.. code-block:: python

   for kv in np.linspace(1, 4, 100):
       sol = solve(eq, ivp={y: 1, y.d(): 0}, tspan=(0, 10), params={k: kv, c: 0.1})

The SymPy backend keeps parameters without a value symbolic. Declaring
assumptions helps SymPy simplify the result:

.. code-block:: python

   k = sp.Symbol("k", positive=True)
   sol = solve(Eq(y.d(2) + k*y, 0), backend="sympy", ivp={y: 1, y.d(): 0})
   sol.as_expr(y)   # cos(sqrt(k)*t)

Boundary value problems
-----------------------

A two-point boundary value problem specifies conditions at both ends of the
interval ``tspan``. Conditions are given as a list of :class:`~odecast.BC`
objects:

.. code-block:: python

   from odecast import BC

   # y'' + y = 0, y(0) = 0, y(1) = 1
   sol = solve(
       Eq(y.d(2) + y, 0),
       bvp=[BC(y, t=0, value=0), BC(y, t=1, value=1)],
       tspan=(0, 1),
   )
   sol.eval(y, 0.5)

Conditions may involve derivatives below the highest order, e.g.
``BC(y.d(), t=0, value=0)``, and array-valued variables, e.g.
``BC(u, t=0, value=[0, 1])``. The total number of conditions must equal the
total order of the system.

The problem is solved with :func:`scipy.integrate.solve_bvp`. Further
keyword arguments are passed on (``tol``, ``max_nodes``, ``bc_tol``), and two
more are available:

``mesh``
   Number of nodes of the initial mesh (default 50), or the mesh itself.

``guess``
   Initial guess, keyed like an ``ivp`` dictionary. Values may be numbers,
   arrays over the mesh, or functions of :math:`t`. States without a guess
   start at zero.

Nonlinear boundary value problems may have several solutions, and the guess
determines which one is found. The Bratu problem
:math:`y'' + e^y = 0`, :math:`y(0) = y(1) = 0` has two:

.. code-block:: python

   eq = Eq(y.d(2) + sp.exp(y), 0)
   bcs = [BC(y, t=0, value=0), BC(y, t=1, value=0)]

   lower = solve(eq, bvp=bcs, tspan=(0, 1))   # y(0.5) = 0.14
   upper = solve(eq, bvp=bcs, tspan=(0, 1),
                 guess={y: lambda t: 12*t*(1 - t)})   # y(0.5) = 4.09

Events
------

An event is a condition whose occurrences are located during integration,
for example the time at which a falling body reaches the ground. Events are
passed to :func:`~odecast.solve` as a list of :class:`~odecast.Event`
objects, or simply as equations:

.. code-block:: python

   from odecast import Event

   y = var("y")
   hits_ground = Event(Eq(y, 0), terminal=True, direction=-1)
   sol = solve(Eq(y.d(2), -9.81), ivp={y: 10, y.d(): 0}, tspan=(0, 10),
               events=[hits_ground])

   sol.t_events[0]               # [1.428]
   sol.event_values(0, y.d())    # velocity at impact: [-14.01]

An event occurs where the difference of the two sides of its equation
changes sign. ``terminal=True`` stops the integration at the first
occurrence; ``direction`` restricts events to crossings from negative to
positive (1) or from positive to negative (-1). Conditions may involve
variables, their derivatives below the highest order, :math:`t` and
parameters. Plain functions ``g(t, x)`` of the state vector, as accepted by
:func:`scipy.integrate.solve_ivp`, are passed through unchanged.

Backends
--------

The ``backend`` argument of :func:`~odecast.solve` selects the solver.

``"scipy"`` (default)
   Numerical solution with :func:`scipy.integrate.solve_ivp` for initial
   value problems and :func:`scipy.integrate.solve_bvp` for boundary value
   problems. Requires ``ivp`` or ``bvp``, and ``tspan``. Further keyword
   arguments to ``solve`` are passed on to the SciPy solver, for example
   ``method``, ``rtol``, ``atol``, ``max_step``, ``first_step`` or
   ``t_eval``:

   .. code-block:: python

      solve(eq, ivp=ic, tspan=(0, 10), method="Radau", rtol=1e-8)

   For the implicit methods ``"Radau"``, ``"BDF"`` and ``"LSODA"``, which
   are suited for stiff problems, odecast supplies the exact Jacobian,
   computed symbolically.

``"sympy"``
   Symbolic solution with :func:`sympy.dsolve`. Initial or boundary
   conditions, if given, determine the integration constants; initial
   conditions apply at ``tspan[0]`` if ``tspan`` is given, otherwise at
   :math:`t = 0`. Conditions may also be incomplete, in which case the
   remaining constants stay in the result.

   .. code-block:: python

      sol = solve(Eq(y.d(2) + y, 0), backend="sympy")
      sol.as_expr(y)   # C1*sin(t) + C2*cos(t)

      sol = solve(Eq(y.d(2) + y, 0), backend="sympy", ivp={y: 1, y.d(): 0})
      sol.as_expr(y)   # cos(t)

   Independent equations are solved one at a time. Coupled systems are
   passed to ``dsolve`` as a whole, which handles linear systems but
   generally not nonlinear ones.

``"auto"``
   Solves numerically if ``ivp`` or ``bvp`` and ``tspan`` are given, and
   falls back to ``"sympy"`` if they are not or if the numerical solution
   fails. Invalid conditions are reported as errors rather than triggering
   the fallback.

Working with solutions
----------------------

A numerical solution provides:

``sol.t``
   The time points returned by the solver; set them with ``t_eval``.

``sol[y]``, ``sol[y.d()]``
   Values of a variable or derivative at ``sol.t``.

``sol.eval(y, times)``
   Values at arbitrary times within ``tspan``. ``times`` may be a scalar or
   an array. The values come from the interpolant of the solver
   (``dense_output`` of ``solve_ivp``, or the cubic interpolant of
   ``solve_bvp``) and are as accurate as the solution itself.

``sol.plot(y, y.d())``
   Plots variables or derivatives against :math:`t` with Matplotlib; all
   variables if none are given. Returns the axes.

``sol.to_pandas()``
   A :class:`pandas.DataFrame` with one column per variable, indexed by
   :math:`t`. Pass targets and ``t=...`` to choose columns and rows.

``sol.t_events``, ``sol.event_values(i, y)``
   Event times and solution values there (see `Events`_).

``sol.as_first_order()``
   The first-order system constructed by odecast: right-hand side function,
   Jacobian (or ``None``), initial state vector, initial time, and a mapping
   from ``(variable, derivative order)`` to state vector index. Useful for
   debugging or for passing the system to a different integrator.

Matplotlib and pandas are optional; install them with
``pip install "odecast[examples]"``.

A symbolic solution provides ``sol.as_expr(target)``, which returns a SymPy
expression for a variable or derivative (a SymPy array for vector
variables), and ``sol.eval(target, times)`` once all integration constants
are determined.

In Jupyter, equations and symbolic solutions are displayed as formulas.
:func:`odecast.latex` returns the LaTeX code:

.. code-block:: python

   from odecast import latex

   latex(Eq(y.d(2) + 0.3*y.d() + y, sp.sin(t)))
   # y + 0.3 y' + y'' = \sin{\left(t \right)}

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

Errors
------

Problems with the equations or conditions raise exceptions derived from
:class:`~odecast.errors.OdecastError`. Conditions are checked before
solving:

.. code-block:: python

   from odecast.errors import MissingInitialConditionError

   y = var("y")
   try:
       solve(Eq(y.d(2) + y, 0), ivp={y: 1.0}, tspan=(0, 10))
   except MissingInitialConditionError as e:
       print(e)
   # Missing initial condition for y^(1). Variable has order 2 and
   # requires conditions for levels 0 to 1.

Supplying a condition for the highest derivative (e.g. ``y.d(2)`` above)
raises :class:`~odecast.errors.OverdeterminedConditionsError`. Invalid
arguments raise :class:`ValueError` or :class:`TypeError`, for example an
initial condition whose shape does not match the variable, or a boundary
condition that is not at an end point of ``tspan``. If the numerical solver
fails, :class:`RuntimeError` is raised.
