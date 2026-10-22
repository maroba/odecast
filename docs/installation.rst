Installation
============

odecast requires Python 3.10 or later.

.. code-block:: bash

   pip install odecast

This also installs the required dependencies NumPy, SciPy and SymPy. The
example scripts and the plotting code in :doc:`examples` additionally use
Matplotlib, which is available as an extra:

.. code-block:: bash

   pip install "odecast[examples]"

From source
-----------

.. code-block:: bash

   git clone https://github.com/maroba/odecast.git
   cd odecast
   pip install -e .

For development, use the ``dev`` extra, which adds pytest, pytest-cov, Ruff
and Black:

.. code-block:: bash

   pip install -e ".[dev]"

Checking the installation
-------------------------

.. code-block:: python

   import odecast
   from odecast import var, Eq, solve

   print(odecast.__version__)

   y = var("y")
   sol = solve(Eq(y.d() + y, 0), ivp={y: 1.0}, tspan=(0, 1))
   print(sol[y][-1])   # approximately exp(-1) = 0.3679
