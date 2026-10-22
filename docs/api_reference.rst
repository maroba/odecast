API Reference
=============

Public interface
----------------

.. autofunction:: odecast.var

.. autoclass:: odecast.Eq

.. autodata:: odecast.t
   :annotation:

.. autofunction:: odecast.solve

.. autoclass:: odecast.BC

.. autoclass:: odecast.Event

.. autofunction:: odecast.latex

Analysis
--------

.. autofunction:: odecast.equilibria

.. autoclass:: odecast.stability.Equilibrium

.. autofunction:: odecast.fit

.. autoclass:: odecast.fitting.FitResult

Solutions
---------

.. autoclass:: odecast.solution.SolutionIVP
   :members:

.. autoclass:: odecast.solution.SolutionBVP

.. autoclass:: odecast.backends.sympy_backend.SolutionExpr
   :members:

Exceptions
----------

.. automodule:: odecast.errors
   :members:
   :undoc-members:
   :show-inheritance:

Internals
---------

The modules below implement the reduction to a first-order system. They are
not part of the stable interface but are documented for contributors.

Equation analysis
~~~~~~~~~~~~~~~~~

.. automodule:: odecast.analyze
   :members:
   :undoc-members:
   :show-inheritance:

Compilation
~~~~~~~~~~~

.. automodule:: odecast.compile
   :members:
   :undoc-members:
   :show-inheritance:

Order reduction
~~~~~~~~~~~~~~~

.. automodule:: odecast.reduce
   :members:
   :undoc-members:
   :show-inheritance:

Validation
~~~~~~~~~~

.. automodule:: odecast.validate
   :members:
   :undoc-members:
   :show-inheritance:

Backends
~~~~~~~~

.. automodule:: odecast.backends.scipy_ivp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: odecast.backends.scipy_bvp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: odecast.backends.sympy_backend
   :members:
   :exclude-members: SolutionExpr
   :undoc-members:
   :show-inheritance:
