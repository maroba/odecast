Installation
============

Requirements
------------

Odecast requires Python 3.10 or later and depends on the following packages:

* NumPy (>= 1.20)
* SciPy (>= 1.7)
* SymPy (>= 1.10)

Installing from PyPI
---------------------

The recommended way to install Odecast is using pip:

.. code-block:: bash

   pip install odecast

This will automatically install all required dependencies.

Installing from Source
----------------------

To install the latest development version from GitHub:

.. code-block:: bash

   git clone https://github.com/maroba/odecast.git
   cd odecast
   pip install -e .

Development Installation
------------------------

For development work, install with additional dependencies:

.. code-block:: bash

   git clone https://github.com/maroba/odecast.git
   cd odecast
   pip install -e ".[dev]"

This includes testing and code quality tools like pytest, black, and ruff.

Verification
------------

To verify your installation, run:

.. code-block:: python

   import odecast
   print(odecast.__version__)

   # Test basic functionality
   from odecast import var, Eq, solve
   y = var("y")
   eq = Eq(y.d() + y, 0)
   sol = solve(eq, ivp={y: 1}, tspan=(0, 1))
   print("Installation successful!")
