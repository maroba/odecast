Quickstart
==========

Solving an equation with odecast involves three steps: declaring the unknown
functions, writing the equations, and calling :func:`odecast.solve`.

Variables
---------

:func:`odecast.var` declares an unknown function of the independent
variable :math:`t`. Derivatives are written with ``.d()``:

.. code-block:: python

   from odecast import var

   y = var("y")
   y.d()     # y'
   y.d(2)    # y''

Vector-valued unknowns are declared with ``shape``; see :doc:`user_guide`.

Equations
---------

:class:`odecast.Eq` takes a left-hand and a right-hand side:

.. code-block:: python

   from odecast import Eq

   Eq(y.d(2) + y, 0)               # y'' + y = 0
   Eq(y.d(2) + 0.1*y.d() + y, 0)   # y'' + 0.1 y' + y = 0

Terms that depend explicitly on :math:`t` are written with SymPy functions
of ``t.symbol``:

.. code-block:: python

   import sympy as sp
   from odecast import t

   Eq(y.d(2) + y, sp.sin(t.symbol))   # y'' + y = sin(t)

Solving
-------

An initial value problem needs a value for each variable and each of its
derivatives below the highest order that occurs, here :math:`y(0)` and
:math:`y'(0)`:

.. code-block:: python

   from odecast import solve

   eq = Eq(y.d(2) + 0.3*y.d() + y, 0)
   sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 10))

The initial conditions are taken at ``tspan[0]``. By default the equation is
integrated with :func:`scipy.integrate.solve_ivp`.

Results
-------

.. code-block:: python

   sol.t                # time points chosen by the integrator
   sol[y]               # y at those points
   sol[y.d()]           # y' at those points
   sol.eval(y, 5.0)     # y(5), interpolated
   sol.eval(y, [1, 2])  # y at several points

Full example
------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from odecast import var, Eq, solve

   y = var("y")
   eq = Eq(y.d(2) + 0.3*y.d() + y, 0)
   sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 10))

   t = np.linspace(0, 10, 500)
   plt.plot(t, sol.eval(y, t), label="y")
   plt.plot(t, sol.eval(y.d(), t), label="y'")
   plt.xlabel("t")
   plt.legend()
   plt.show()

Where to go next: the :doc:`user_guide` covers systems, vector variables and
backends; :doc:`examples` contains worked problems from several fields.
