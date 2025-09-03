Quick Start Guide
=================

This guide provides a quick introduction to Odecast's basic functionality.

Basic Concepts
--------------

Odecast operates on three main concepts:

1. **Variables**: Mathematical variables and their derivatives
2. **Equations**: Mathematical relationships between variables
3. **Solutions**: Numerical or symbolic results

Creating Variables
------------------

Variables are created using the ``var`` function:

.. code-block:: python

   from odecast import var

   # Scalar variable
   y = var("y")

   # Vector variable
   u = var("u", shape=2)  # 2D vector
   v = var("v", shape=3)  # 3D vector

Derivatives are accessed using the ``.d()`` method:

.. code-block:: python

   y_prime = y.d()      # First derivative: y'
   y_double_prime = y.d(2)  # Second derivative: y''

   # For vector variables
   u_dot = u.d()        # Vector of first derivatives
   u_ddot = u.d(2)      # Vector of second derivatives

Writing Equations
-----------------

Equations are created using the ``Eq`` function:

.. code-block:: python

   from odecast import Eq

   # Simple harmonic oscillator: y'' + y = 0
   eq1 = Eq(y.d(2) + y, 0)

   # Damped oscillator: y'' + 0.1*y' + y = 0
   eq2 = Eq(y.d(2) + 0.1*y.d() + y, 0)

   # Non-homogeneous: y'' + y = sin(t)
   from odecast import t
   import sympy as sp
   eq3 = Eq(y.d(2) + y, sp.sin(t))

Solving ODEs
------------

Use the ``solve`` function to solve differential equations:

.. code-block:: python

   from odecast import solve

   # Define the problem
   y = var("y")
   equation = Eq(y.d(2) + 0.3*y.d() + y, 0)

   # Solve with initial conditions
   solution = solve(
       equation,
       ivp={y: 1.0, y.d(): 0.0},  # Initial conditions
       tspan=(0, 10),              # Time span
       backend="scipy"             # Solver backend
   )

Accessing Results
-----------------

Solution objects provide easy access to results:

.. code-block:: python

   # Time points
   times = solution.t

   # Variable values
   y_values = solution[y]
   y_prime_values = solution[y.d()]

   # Evaluate at specific times
   y_at_5 = solution.eval(y, 5.0)
   y_prime_at_5 = solution.eval(y.d(), 5.0)

Complete Example
----------------

Here's a complete example solving a damped harmonic oscillator:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, solve

   # Define the system: y'' + 0.3*y' + y = 0
   y = var("y")
   equation = Eq(y.d(2) + 0.3*y.d() + y, 0)

   # Set initial conditions: y(0) = 1, y'(0) = 0
   initial_conditions = {y: 1.0, y.d(): 0.0}

   # Solve over time interval [0, 10]
   solution = solve(equation, ivp=initial_conditions, tspan=(0, 10))

   # Plot results
   plt.figure(figsize=(10, 6))
   plt.plot(solution.t, solution[y], label='Position')
   plt.plot(solution.t, solution[y.d()], label='Velocity')
   plt.xlabel('Time')
   plt.ylabel('Value')
   plt.title('Damped Harmonic Oscillator')
   plt.legend()
   plt.grid(True)
   plt.show()

Next Steps
----------

* Learn about :doc:`user_guide` for more advanced features
* Explore :doc:`examples` for real-world applications
* Check the :doc:`api_reference` for detailed function documentation
