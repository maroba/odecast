User Guide
==========

This guide covers advanced features and usage patterns in Odecast.

Vector Variables
----------------

Vector variables enable solving systems of coupled differential equations with
elegant syntax.

Creating Vector Variables
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from odecast import var

   # 2D vector variable
   u = var("u", shape=2)  # Components: u[0], u[1]

   # 3D vector variable
   v = var("v", shape=3)  # Components: v[0], v[1], v[2]

Component Access
~~~~~~~~~~~~~~~~

Individual components behave like scalar variables:

.. code-block:: python

   u = var("u", shape=2)

   # Access components
   u0 = u[0]  # First component
   u1 = u[1]  # Second component

   # Use in equations
   eq1 = Eq(u[0].d(2) + u[0], 0)  # u₀'' + u₀ = 0
   eq2 = Eq(u[1].d() + u[0], 0)   # u₁' + u₀ = 0

Vector Operations
~~~~~~~~~~~~~~~~~

Vector variables support mathematical operations:

.. code-block:: python

   u = var("u", shape=2)

   # Vector derivatives
   u_dot = u.d()    # [u[0]', u[1]']
   u_ddot = u.d(2)  # [u[0]'', u[1]'']

   # Vector equations automatically expand
   eq = Eq(u.d(2) + u, 0)
   # Equivalent to: u[0]'' + u[0] = 0, u[1]'' + u[1] = 0

Vector Initial Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~

Specify initial conditions using lists or arrays:

.. code-block:: python

   u = var("u", shape=2)
   eq = Eq(u.d(2) + u, 0)

   # Vector initial conditions
   ivp = {
       u: [1.0, 0.5],        # u(0) = [1.0, 0.5]
       u.d(): [0.0, -0.2]    # u'(0) = [0.0, -0.2]
   }

   sol = solve(eq, ivp=ivp, tspan=(0, 10))

   # Access vector solutions
   u_trajectory = sol[u]      # Shape: (2, n_points)
   u0_trajectory = sol[u[0]]  # Shape: (n_points,)
   u1_trajectory = sol[u[1]]  # Shape: (n_points,)

Mixed Systems
~~~~~~~~~~~~~

Combine scalar and vector variables in the same system:

.. code-block:: python

   # Coupled scalar-vector system
   x = var("x")           # Scalar
   u = var("u", shape=2)  # Vector

   equations = [
       Eq(x.d(2) + x - u[0], 0),    # x'' + x - u₀ = 0
       Eq(u[0].d() + u[1], x),      # u₀' + u₁ = x
       Eq(u[1].d() + u[0], 0)       # u₁' + u₀ = 0
   ]

   initial_conditions = {
       x: 1.0,
       x.d(): 0.0,
       u: [0.5, 0.0],
       u.d(): [0.0, 0.0]
   }

   solution = solve(equations, ivp=initial_conditions, tspan=(0, 5))

Backend Selection
-----------------

Odecast supports multiple solver backends for different use cases.

SciPy Backend
~~~~~~~~~~~~~

The SciPy backend provides fast numerical solutions:

.. code-block:: python

   # Numeric solution using SciPy
   solution = solve(equation, ivp=conditions, tspan=(0, 10), backend="scipy")

   # Access numeric arrays
   times = solution.t          # NumPy array
   values = solution[y]        # NumPy array

SymPy Backend
~~~~~~~~~~~~~

The SymPy backend provides exact symbolic solutions:

.. code-block:: python

   # Symbolic solution using SymPy
   solution = solve(equation, backend="sympy")

   # Get symbolic expression
   symbolic_expr = solution.as_expr(y)
   print(symbolic_expr)  # Prints SymPy expression

   # Evaluate symbolically
   result = solution.eval(y, 5.0)  # Exact evaluation

Auto Backend
~~~~~~~~~~~~

The auto backend tries numeric first, falling back to symbolic:

.. code-block:: python

   # Automatic backend selection
   solution = solve(equation, ivp=conditions, tspan=(0, 10), backend="auto")

Error Handling and Validation
------------------------------

Odecast provides comprehensive validation with clear error messages.

Missing Initial Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   y = var("y")
   eq = Eq(y.d(2) + y, 0)

   # This will raise ODEValidationError
   try:
       sol = solve(eq, ivp={y: 1.0}, tspan=(0, 10))  # Missing y.d()
   except odecast.ODEValidationError as e:
       print(f"Validation error: {e}")

Inconsistent Dimensions
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   u = var("u", shape=2)
   eq = Eq(u.d(2) + u, 0)

   # This will raise an error
   try:
       sol = solve(eq, ivp={u: [1.0]}, tspan=(0, 10))  # Wrong dimension
   except ValueError as e:
       print(f"Dimension error: {e}")

Advanced Features
-----------------

Time-Dependent Forcing
~~~~~~~~~~~~~~~~~~~~~~

Include time-dependent terms in your equations:

.. code-block:: python

   from odecast import t
   import sympy as sp

   y = var("y")
   # Driven oscillator: y'' + y = cos(2*t)
   eq = Eq(y.d(2) + y, sp.cos(2*t))
   sol = solve(eq, ivp={y: 0, y.d(): 1}, tspan=(0, 10))

Nonlinear Systems
~~~~~~~~~~~~~~~~~

Odecast handles nonlinear differential equations:

.. code-block:: python

   # Van der Pol oscillator: y'' - μ(1 - y²)y' + y = 0
   y = var("y")
   mu = 1.0
   eq = Eq(y.d(2) - mu*(1 - y**2)*y.d() + y, 0)
   sol = solve(eq, ivp={y: 0.1, y.d(): 0}, tspan=(0, 20))

Parameter Studies
~~~~~~~~~~~~~~~~~

Solve equations with different parameters:

.. code-block:: python

   import numpy as np

   y = var("y")
   damping_values = [0.1, 0.3, 0.5, 1.0]
   solutions = []

   for damping in damping_values:
       eq = Eq(y.d(2) + damping*y.d() + y, 0)
       sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 10))
       solutions.append(sol)

Best Practices
--------------

1. **Use descriptive variable names**: ``position = var("x")`` instead of ``x = var("x")``
2. **Group related equations**: Use lists for systems of equations
3. **Validate inputs**: Check dimensions and initial conditions before solving
4. **Choose appropriate backends**: Use SymPy for exact solutions, SciPy for large systems
5. **Handle errors gracefully**: Wrap solve calls in try-except blocks for production code
