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

Vector variables
----------------

A variable with ``shape`` represents a vector-valued function. Its
components are scalar variables and can be used anywhere a scalar variable
can:

.. code-block:: python

   u = var("u", shape=2)
   u[0], u[1]          # components
   u.d(), u.d(2)       # vector derivatives

An equation involving a vector variable as a whole is expanded into one
equation per component. The following are equivalent:

.. code-block:: python

   Eq(u.d(2) + u, 0)

   [Eq(u[0].d(2) + u[0], 0),
    Eq(u[1].d(2) + u[1], 0)]

Initial conditions for vector variables are given as sequences:

.. code-block:: python

   sol = solve(
       Eq(u.d(2) + u, 0),
       ivp={u: [1.0, 0.5], u.d(): [0.0, -0.2]},
       tspan=(0, 10),
   )

   sol[u]      # shape (2, n)
   sol[u[0]]   # shape (n,)

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

Only vector variables (``shape=n``) are currently supported; matrix-valued
variables are planned.

Time-dependent terms
--------------------

The independent variable is exported as :data:`odecast.t`. Inside SymPy
functions use its symbol, ``t.symbol``:

.. code-block:: python

   import sympy as sp
   from odecast import t

   y = var("y")
   eq = Eq(y.d(2) + y, sp.cos(2*t.symbol))   # driven oscillator
   sol = solve(eq, ivp={y: 0.0, y.d(): 1.0}, tspan=(0, 10))

Nonlinear equations
-------------------

Equations may be nonlinear in the dependent variables as long as they are
polynomial. Write powers as repeated products:

.. code-block:: python

   # Van der Pol oscillator: y'' - mu (1 - y^2) y' + y = 0
   mu = 1.0
   eq = Eq(y.d(2) - mu*(1 - y*y)*y.d() + y, 0)
   sol = solve(eq, ivp={y: 0.1, y.d(): 0.0}, tspan=(0, 20))

The ``**`` operator and functions such as ``sin`` applied to a dependent
variable are not supported at present.

Backends
--------

The ``backend`` argument of :func:`~odecast.solve` selects the solver.

``"scipy"`` (default)
   Numerical integration with :func:`scipy.integrate.solve_ivp`. Requires
   ``ivp`` and ``tspan``. Further keyword arguments to ``solve`` are passed
   on to ``solve_ivp``:

   .. code-block:: python

      solve(eq, ivp=ic, tspan=(0, 10), method="Radau", rtol=1e-8, max_step=0.01)

``"sympy"``
   Symbolic solution with :func:`sympy.dsolve`. Returns the general
   solution; initial conditions are not applied, and ``ivp`` and ``tspan``
   are not used.

   .. code-block:: python

      sol = solve(Eq(y.d(2) + y, 0), backend="sympy")
      sol.as_expr(y)   # C1*sin(t) + C2*cos(t)

``"auto"``
   Solves numerically if ``ivp`` and ``tspan`` are given, and falls back to
   ``"sympy"`` if they are not or if numerical integration fails. Invalid
   initial conditions are reported as errors rather than triggering the
   fallback.

Working with solutions
----------------------

A numerical solution provides:

``sol.t``
   The time points returned by the integrator.

``sol[y]``, ``sol[y.d()]``
   Values of a variable or derivative at ``sol.t``.

``sol.eval(y, times)``
   Values at arbitrary times within ``tspan``, obtained by interpolation.
   ``times`` may be a scalar or an array.

``sol.as_first_order()``
   The first-order system constructed by odecast: right-hand side function,
   Jacobian (or ``None``), initial state vector, initial time, and a mapping
   from ``(variable, derivative order)`` to state vector index. Useful for
   debugging or for passing the system to a different integrator.

The integrator chooses the time points itself. For plotting, evaluate on a
fine grid with ``sol.eval`` or limit the step size with ``max_step``.

Errors
------

All exceptions raised by odecast derive from
:class:`~odecast.errors.OdecastError`. Initial conditions are checked before
integration:

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
raises :class:`~odecast.errors.OverdeterminedConditionsError`. An initial
condition for a vector variable whose length does not match the variable's
shape raises :class:`ValueError`.
