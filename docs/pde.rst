Partial differential equations
==============================

Time-dependent partial differential equations are solved with the method of
lines: the spatial derivatives are replaced by finite differences on a grid,
and the resulting large system of ordinary differential equations is
integrated in time. The finite differences come from
`findiff <https://github.com/maroba/findiff>`_, which is installed with

.. code-block:: bash

   pip install "odecast[pde]"

Fields and grids
----------------

A :class:`~odecast.pde.Grid` defines the spatial coordinates, a field the
unknown function on it:

.. code-block:: python

   import numpy as np
   from odecast import Eq
   from odecast.pde import Grid, field, solve_pde, Dirichlet, Neumann, Periodic

   grid = Grid(x=np.linspace(0, 1, 101))
   u = field("u", grid)

Fields have time derivatives ``u.d()`` and ``u.d(2)`` and spatial
derivatives ``u.dx(n)``, ``u.dy(n)``, ``u.dz(n)``, ``u.diff(x=1, y=1)`` for
mixed derivatives, and ``u.laplace()``. Equations may contain nonlinear
terms, the coordinates (``grid.x``), the time ``t`` and parameters:

.. code-block:: python

   # Heat equation
   Eq(u.d(), 0.1*u.dx(2))

   # Fisher-KPP reaction-diffusion equation
   Eq(u.d(), D*u.dx(2) + r*u*(1 - u))

   # Wave equation (second order in time)
   Eq(u.d(2), c**2*u.dx(2))

Grids have one to three axes, e.g. ``Grid(x=xs, y=ys)``. The spacing may be
non-uniform, except on periodic axes.

Solving
-------

:func:`~odecast.pde.solve_pde` takes the equations, initial values,
boundary conditions and the time interval:

.. code-block:: python

   sol = solve_pde(
       Eq(u.d(), 0.1*u.dx(2)),
       ic={u: lambda x: np.sin(np.pi*x)},
       bc={u: {"x": (Dirichlet(0), Dirichlet(0))}},
       tspan=(0, 1),
   )

   sol[u]               # values on the grid, shape (101, len(sol.t))
   sol.eval(u, 0.5)     # values at t = 0.5, shape (101,)
   sol.plot(u)          # profiles at several times (2D: image)

Initial values are arrays of the grid shape, numbers, or functions of the
coordinates. For equations of second order in time the initial time
derivative is given as well: ``ic={u: ..., u.d(): ...}``. Several fields are
solved together by passing a list of equations, one per field.

Boundary conditions
~~~~~~~~~~~~~~~~~~~

Boundary conditions are given per field and axis, as a pair for the lower
and upper end, or as ``Periodic()``:

``Dirichlet(value)``
   The field has the given value. The value may be a function of ``t``.
``Neumann(value)``
   The derivative along the axis has the given value (note: along the axis,
   not the outward normal).
``Periodic()``
   The field is periodic; the grid must not contain the end point, e.g.
   ``np.linspace(0, 2*np.pi, 128, endpoint=False)``.

Boundary values are not integrated in time but computed from the boundary
conditions whenever the right-hand side is evaluated, so they hold exactly.

Accuracy and stability
~~~~~~~~~~~~~~~~~~~~~~

``acc`` sets the accuracy order of the finite differences (default 2).
Higher orders give much smaller errors for smooth solutions, but use
one-sided stencils near non-periodic boundaries. These can make undamped
equations of second order in time, such as the wave equation, unstable;
odecast warns in that case. Use ``acc=2`` there, or periodic boundaries.

Diffusion makes the system stiff, so the default time integrator is
``"BDF"``, which receives the sparsity pattern of the Jacobian. Other
options of :func:`scipy.integrate.solve_ivp` (``method``, ``rtol``,
``atol``, ...) are passed through.
