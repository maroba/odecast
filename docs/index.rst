Odecast Documentation
=====================

.. image:: odecast-logo.png
   :width: 150px
   :align: center

Odecast is a Python library for solving higher-order ordinary differential equations (ODEs) 
without manual reduction to first-order systems. It provides an intuitive syntax that allows 
you to write differential equations in natural mathematical form.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api_reference
   examples
   contributing

Features
--------

* **Natural syntax**: Write ODEs using mathematical notation like ``y.d(2)`` for second derivatives
* **Vector variables**: Support for multi-dimensional systems with automatic equation expansion
* **Multiple backends**: Choose between SciPy (numeric) and SymPy (symbolic) solvers
* **Automatic validation**: Comprehensive error checking for initial and boundary conditions
* **Flexible interface**: Support for both initial value problems (IVP) and boundary value problems (BVP)

Quick Example
-------------

.. code-block:: python

   from odecast import var, Eq, solve

   # Define a damped harmonic oscillator: y'' + 0.3*y' + y = 0
   y = var("y")
   equation = Eq(y.d(2) + 0.3*y.d() + y, 0)

   # Solve with initial conditions
   solution = solve(equation, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 10))

   # Access results
   times = solution.t
   positions = solution[y]
   velocities = solution[y.d()]

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
