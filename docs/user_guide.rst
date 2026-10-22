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

All operations are element-wise, and all array-valued variables in one
equation must have the same shape. Scalars and scalar variables are
broadcast. Functions such as ``sin`` must be applied to components,
``sp.sin(u[0])``, not to the variable as a whole.

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

Backends
--------

The ``backend`` argument of :func:`~odecast.solve` selects the solver.

``"scipy"`` (default)
   Numerical solution with :func:`scipy.integrate.solve_ivp` for initial
   value problems and :func:`scipy.integrate.solve_bvp` for boundary value
   problems. Requires ``ivp`` or ``bvp``, and ``tspan``. Further keyword
   arguments to ``solve`` are passed on to the SciPy solver:

   .. code-block:: python

      solve(eq, ivp=ic, tspan=(0, 10), method="Radau", rtol=1e-8, max_step=0.01)

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
   The time points returned by the solver.

``sol[y]``, ``sol[y.d()]``
   Values of a variable or derivative at ``sol.t``.

``sol.eval(y, times)``
   Values at arbitrary times within ``tspan``. ``times`` may be a scalar or
   an array. Initial value solutions are interpolated linearly; boundary
   value solutions use the cubic interpolant computed by ``solve_bvp``.

``sol.as_first_order()``
   The first-order system constructed by odecast: right-hand side function,
   Jacobian (or ``None``), initial state vector, initial time, and a mapping
   from ``(variable, derivative order)`` to state vector index. Useful for
   debugging or for passing the system to a different integrator.

For initial value problems the integrator chooses the time points itself.
For plotting, evaluate on a fine grid with ``sol.eval`` or limit the step
size with ``max_step``.

A symbolic solution provides ``sol.as_expr(target)``, which returns a SymPy
expression for a variable or derivative (a SymPy array for vector
variables), and ``sol.eval(target, times)`` once all integration constants
are determined.

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
