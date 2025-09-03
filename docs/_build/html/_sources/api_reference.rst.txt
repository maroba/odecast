API Reference
=============

This section provides detailed documentation for all Odecast functions and classes.

Core Functions
--------------

.. automodule:: odecast.api
   :members:
   :undoc-members:
   :show-inheritance:

Variables and Equations
-----------------------

.. autofunction:: odecast.var

.. autofunction:: odecast.Eq

.. autodata:: odecast.t

Solving Functions
-----------------

.. autofunction:: odecast.solve

Solution Classes
----------------

.. autoclass:: odecast.solution.Solution
   :members:
   :undoc-members:
   :show-inheritance:

Validation and Errors
---------------------

.. automodule:: odecast.errors
   :members:
   :undoc-members:
   :show-inheritance:

Internal Modules
----------------

The following modules are primarily for internal use but may be useful for 
advanced users or contributors.

Equation Analysis
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

Order Reduction
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

Backend Implementations
~~~~~~~~~~~~~~~~~~~~~~~

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
   :undoc-members:
   :show-inheritance:
