Examples
========

The examples below are rendered with Matplotlib when the documentation is
built. Runnable scripts are in the ``examples/`` directory of the
repository.

Mechanics
---------

Damped harmonic oscillator
~~~~~~~~~~~~~~~~~~~~~~~~~~

A mass :math:`m` on a spring with stiffness :math:`k` and damping
coefficient :math:`c` obeys

.. math::

   m\ddot x + c\dot x + kx = 0,

or, in terms of the natural frequency :math:`\omega_n = \sqrt{k/m}` and the
damping ratio :math:`\zeta = c/(2\sqrt{km})`,

.. math::

   \ddot x + 2\zeta\omega_n\dot x + \omega_n^2 x = 0.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, solve

   omega_n = 2.0
   zeta = 0.3

   x = var("x")
   eq = Eq(x.d(2) + 2*zeta*omega_n*x.d() + omega_n**2*x, 0)
   sol = solve(eq, ivp={x: 1.0, x.d(): 0.0}, tspan=(0, 5), max_step=0.01)

   t = np.linspace(0, 5, 1000)
   pos = sol.eval(x, t)
   vel = sol.eval(x.d(), t)

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

   ax1.plot(t, pos, label="$x$")
   ax1.plot(t, vel, "--", label=r"$\dot x$")
   ax1.set_xlabel("$t$")
   ax1.legend()
   ax1.set_title("Time series")

   ax2.plot(pos, vel)
   ax2.set_xlabel("$x$")
   ax2.set_ylabel(r"$\dot x$")
   ax2.set_title("Phase portrait")

   plt.tight_layout()

With :math:`\zeta < 1` the system is underdamped: the amplitude decays
exponentially while the trajectory spirals into the origin.

Pendulum at large amplitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The undamped pendulum satisfies

.. math::

   \ddot\theta + \frac{g}{L}\sin\theta = 0.

For small angles :math:`\sin\theta \approx \theta`, which gives the
harmonic oscillator with period :math:`2\pi\sqrt{L/g}`. At larger angles
the restoring force is weaker than in the linear approximation and the
period grows. The comparison below starts both at
:math:`\theta_0 = 60^\circ`.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   import sympy as sp
   from odecast import var, Eq, solve

   g, L = 9.81, 1.0
   theta0 = np.pi / 3

   theta = var("theta")
   eq_lin = Eq(theta.d(2) + (g/L)*theta, 0)
   eq_full = Eq(theta.d(2) + (g/L)*sp.sin(theta), 0)

   ic = {theta: theta0, theta.d(): 0.0}
   sol_lin = solve(eq_lin, ivp=ic, tspan=(0, 10), max_step=0.02)
   sol_full = solve(eq_full, ivp=ic, tspan=(0, 10), max_step=0.02)

   t = np.linspace(0, 10, 1000)
   plt.figure(figsize=(8, 4))
   plt.plot(t, np.degrees(sol_lin.eval(theta, t)), "--", label=r"$\sin\theta \approx \theta$")
   plt.plot(t, np.degrees(sol_full.eval(theta, t)), label=r"$\sin\theta$")
   plt.xlabel("$t$ [s]")
   plt.ylabel(r"$\theta$ [deg]")
   plt.ylim(-75, 90)
   plt.legend(loc="upper center", ncol=2)
   plt.tight_layout()

Isotropic oscillator in two dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A particle in the potential :math:`V = \tfrac12\omega^2|\mathbf r|^2`
obeys

.. math::

   \ddot{\mathbf r} + \omega^2\mathbf r = 0,
   \qquad \mathbf r = (x, y).

This is a natural use case for a vector variable. Generic initial
conditions lead to an elliptical orbit, and the total energy is conserved.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, solve

   omega = 1.0

   r = var("r", shape=2)
   eq = Eq(r.d(2) + omega**2 * r, 0)
   sol = solve(eq, ivp={r: [1.0, 0.0], r.d(): [0.0, 1.5]},
               tspan=(0, 4*np.pi), max_step=0.05)

   t = np.linspace(0, 4*np.pi, 1000)
   x, y = sol.eval(r, t)
   vx, vy = sol.eval(r.d(), t)

   kinetic = 0.5 * (vx**2 + vy**2)
   potential = 0.5 * omega**2 * (x**2 + y**2)

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

   ax1.plot(x, y)
   ax1.set_aspect("equal")
   ax1.set_xlabel("$x$")
   ax1.set_ylabel("$y$")
   ax1.set_title("Orbit")

   ax2.plot(t, kinetic, label="kinetic")
   ax2.plot(t, potential, label="potential")
   ax2.plot(t, kinetic + potential, "k", label="total")
   ax2.set_xlabel("$t$")
   ax2.set_title("Energy")
   ax2.legend(loc="center right")

   plt.tight_layout()

Population dynamics
-------------------

Lotka–Volterra model
~~~~~~~~~~~~~~~~~~~~

The predator–prey model

.. math::

   \dot x &= ax - bxy, \\
   \dot y &= -cy + dxy,

with prey :math:`x`, predators :math:`y` and positive parameters
:math:`a, b, c, d`, has the conserved quantity

.. math::

   V(x, y) = dx - c\ln x + by - a\ln y.

Solutions are therefore closed orbits around the coexistence equilibrium
:math:`(c/d,\ a/b)`. Plotting :math:`V` along the numerical solution is a
convenient check of the integration accuracy.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, solve

   a, b, c, d = 1.0, 0.1, 1.5, 0.075

   x, y = var("x"), var("y")
   eqs = [
       Eq(x.d() - a*x + b*x*y, 0),
       Eq(y.d() + c*y - d*x*y, 0),
   ]
   sol = solve(eqs, ivp={x: 10.0, y: 5.0}, tspan=(0, 15), max_step=0.05)

   t = np.linspace(0, 15, 1000)
   xs = sol.eval(x, t)
   ys = sol.eval(y, t)
   V = d*xs - c*np.log(xs) + b*ys - a*np.log(ys)

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))

   ax1.plot(t, xs, label="prey")
   ax1.plot(t, ys, label="predators")
   ax1.set_xlabel("$t$")
   ax1.legend()
   ax1.set_title("Populations")

   ax2.plot(xs, ys)
   ax2.plot(c/d, a/b, "k+", markersize=10)
   ax2.set_xlabel("prey")
   ax2.set_ylabel("predators")
   ax2.set_title("Phase portrait")

   ax3.plot(t, V - V[0])
   ax3.set_xlabel("$t$")
   ax3.set_title("$V(t) - V(0)$")

   plt.tight_layout()

Electrical circuits
-------------------

Series RLC circuit
~~~~~~~~~~~~~~~~~~

Kirchhoff's voltage law for a series RLC circuit driven by a voltage
:math:`V(t)` gives, for the capacitor charge :math:`q`,

.. math::

   L\ddot q + R\dot q + \frac{q}{C} = V(t).

The current is :math:`i = \dot q`. The natural frequency and damping ratio
are :math:`\omega_n = 1/\sqrt{LC}` and
:math:`\zeta = \tfrac{R}{2}\sqrt{C/L}`. We switch on a constant voltage
:math:`V_0` at :math:`t = 0` with the capacitor initially discharged.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, solve

   L = 1e-3    # H
   R = 10.0    # Ohm
   C = 1e-6    # F
   V0 = 12.0   # V

   q = var("q")
   eq = Eq(L*q.d(2) + R*q.d() + q/C, V0)
   sol = solve(eq, ivp={q: 0.0, q.d(): 0.0}, tspan=(0, 5e-3), max_step=5e-6)

   t = np.linspace(0, 5e-3, 1000)
   v_C = sol.eval(q, t) / C
   i = sol.eval(q.d(), t)

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

   ax1.plot(t*1e3, v_C)
   ax1.axhline(V0, color="k", linestyle=":")
   ax1.set_xlabel("$t$ [ms]")
   ax1.set_ylabel("$v_C$ [V]")
   ax1.set_title("Capacitor voltage")

   ax2.plot(t*1e3, i*1e3)
   ax2.set_xlabel("$t$ [ms]")
   ax2.set_ylabel("$i$ [mA]")
   ax2.set_title("Current")

   plt.tight_layout()

With these values :math:`\zeta \approx 0.16`; the capacitor voltage
overshoots and rings before settling at :math:`V_0`.

Step response of a second-order system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard second-order plant with unit DC gain,

.. math::

   \ddot y + 2\zeta\omega_n\dot y + \omega_n^2 y = \omega_n^2 r,

responds to a unit step :math:`r = 1` as follows.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, solve

   omega_n, zeta, r = 2.0, 0.1, 1.0

   y = var("y")
   eq = Eq(y.d(2) + 2*zeta*omega_n*y.d() + omega_n**2*y, omega_n**2*r)
   sol = solve(eq, ivp={y: 0.0, y.d(): 0.0}, tspan=(0, 20), max_step=0.01)

   t = np.linspace(0, 20, 2000)
   plt.plot(t, sol.eval(y, t))
   plt.axhline(r, color="k", linestyle=":")
   plt.xlabel("$t$")
   plt.ylabel("$y$")
   plt.show()

   overshoot = sol.eval(y, t).max() - r
   print(f"overshoot: {overshoot:.2f}")   # theory: exp(-pi*zeta/sqrt(1-zeta^2)) = 0.73

Boundary value problems
-----------------------

Deflection of a beam
~~~~~~~~~~~~~~~~~~~~

The deflection :math:`w(x)` of a beam with bending stiffness :math:`EI`
under a uniform load :math:`q` satisfies the fourth-order equation

.. math::

   EI\,w^{(4)} = q.

For a beam of length :math:`\ell` that is simply supported at both ends, the
deflection and the bending moment :math:`EI\,w''` vanish at
:math:`x = 0` and :math:`x = \ell`. The exact solution is

.. math::

   w(x) = \frac{q\,x}{24\,EI}\left(\ell^3 - 2\ell x^2 + x^3\right),

with maximum :math:`5q\ell^4/(384\,EI)` at the centre. Here the independent
variable is a position rather than a time; odecast still calls it
:math:`t`.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, solve, BC

   EI, q, ell = 1.0, 1.0, 1.0

   w = var("w")
   eq = Eq(EI*w.d(4), q)
   bcs = [
       BC(w, t=0, value=0), BC(w.d(2), t=0, value=0),
       BC(w, t=ell, value=0), BC(w.d(2), t=ell, value=0),
   ]
   sol = solve(eq, bvp=bcs, tspan=(0, ell))

   x = np.linspace(0, ell, 200)
   exact = q*x*(ell**3 - 2*ell*x**2 + x**3) / (24*EI)

   plt.figure(figsize=(8, 3.5))
   plt.plot(x, -sol.eval(w, x), label="odecast")
   plt.plot(x, -exact, "k:", label="exact")
   plt.xlabel("$x$")
   plt.ylabel("$-w(x)$")
   plt.legend()
   plt.tight_layout()

   print(f"max deflection: {sol.eval(w, ell/2):.6f}")   # 5/384 = 0.013021

Differential-algebraic equations
--------------------------------

Pendulum in Cartesian coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pendulum of length :math:`L = 1` described by the position
:math:`(x, y)` of the bob and the string tension :math:`\lambda` (per unit
mass and length):

.. math::

   \ddot x = -\lambda x, \qquad \ddot y = -\lambda y - g, \qquad
   x^2 + y^2 = L^2.

The constraint does not contain :math:`\lambda`; odecast differentiates it
twice (index 3). The result is compared with the pendulum in polar form,
:math:`\ddot\theta + g\sin\theta = 0`, for which the tension is
:math:`\lambda = g\cos\theta + \dot\theta^2`.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   import sympy as sp
   from odecast import var, Eq, solve

   g, theta0 = 9.81, np.pi / 3

   x, y, lam = var("x"), var("y"), var("lam")
   eqs = [Eq(x.d(2), -lam*x), Eq(y.d(2), -lam*y - g), Eq(x**2 + y**2, 1)]
   ic = {x: np.sin(theta0), x.d(): 0.0, y: -np.cos(theta0), y.d(): 0.0}
   sol = solve(eqs, ivp=ic, tspan=(0, 10), rtol=1e-8, atol=1e-10)

   theta = var("theta")
   polar = solve(Eq(theta.d(2) + g*sp.sin(theta), 0),
                 ivp={theta: theta0, theta.d(): 0.0}, tspan=(0, 10),
                 rtol=1e-8, atol=1e-10)

   t = np.linspace(0, 10, 1000)
   tension = g*np.cos(polar.eval(theta, t)) + polar.eval(theta.d(), t)**2

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
   ax1.plot(t, sol.eval(lam, t), label="Cartesian (DAE)")
   ax1.plot(t, tension, "k:", label="polar")
   ax1.set_xlabel("$t$")
   ax1.set_title(r"Tension $\lambda$")
   ax1.set_ylim(3, 24)
   ax1.legend(loc="upper center", ncol=2)

   ax2.semilogy(t, np.abs(sol.constraint_residuals(t)[0]) + 1e-17)
   ax2.set_xlabel("$t$")
   ax2.set_title(r"Constraint violation $|x^2 + y^2 - 1|$")
   plt.tight_layout()

The constraint violation stays at the level of the solver tolerance.

Tank draining through a pipe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A tank with cross-section :math:`A_t` drains through a pipe of length
:math:`\ell` and diameter :math:`D`. The water level :math:`h` falls as

.. math::

   A_t \dot h = -A_p v,

where the outflow velocity :math:`v` follows from the energy balance with
pipe friction,

.. math::

   g h = \left(1 + K + f\frac{\ell}{D}\right)\frac{v^2}{2},

and the friction factor :math:`f` from the Colebrook equation for turbulent
flow,

.. math::

   \frac{1}{\sqrt f} = -2\log_{10}\left(\frac{\varepsilon}{3.7 D}
   + \frac{2.51}{\mathrm{Re}\sqrt f}\right),
   \qquad \mathrm{Re} = \frac{vD}{\nu}.

The algebraic equations for :math:`v` and :math:`f` have no closed-form
solution, so odecast solves them numerically at every step. The starting
guess for the first step is taken from ``ivp``. The integration stops when
the level reaches 0.5 m.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   import sympy as sp
   from odecast import var, Eq, Event, solve

   g = 9.81
   A_t, D, ell = 1.0, 0.05, 20.0      # m^2, m, m
   eps, nu, K = 1e-4, 1e-6, 0.5       # roughness (m), viscosity (m^2/s), inlet loss
   A_p = np.pi * D**2 / 4

   h, v, f = var("h"), var("v"), var("f")
   Re = v * D / nu
   eqs = [
       Eq(A_t * h.d(), -A_p * v),
       Eq(g * h, (1 + K + f * ell / D) * v**2 / 2),
       Eq(1 / sp.sqrt(f), -2 * sp.log(eps / (3.7 * D) + 2.51 / (Re * sp.sqrt(f)), 10)),
   ]
   sol = solve(eqs, ivp={h: 2.0, v: 1.0, f: 0.02}, tspan=(0, 2000),
               events=Event(Eq(h, 0.5), terminal=True))

   t = np.linspace(0, sol.t[-1], 300)
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
   ax1.plot(t, sol.eval(h, t))
   ax1.set_xlabel("$t$ [s]")
   ax1.set_title("Water level $h$ [m]")
   ax2.plot(t, sol.eval(f, t))
   ax2.set_xlabel("$t$ [s]")
   ax2.set_title("Friction factor $f$")
   plt.tight_layout()

   print(f"level 0.5 m reached after {sol.t_events[0][0]:.0f} s")

As the flow slows down, the Reynolds number falls and the friction factor
rises.

Analysis
--------

Phase portrait of the damped pendulum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The damped pendulum :math:`\ddot\theta + 0.3\,\dot\theta + \sin\theta = 0`
has equilibria at :math:`\theta = k\pi`. The downward position is a stable
focus, the upward one a saddle. Trajectories starting at the upward position
:math:`\theta = -\pi` with increasing velocity show how the saddle at
:math:`\pi` separates swings that settle at :math:`0` from those that go
over the top and settle at :math:`2\pi`.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   import sympy as sp
   from odecast import var, Eq, solve, equilibria

   theta = var("theta")
   eq = Eq(theta.d(2) + 0.3*theta.d() + sp.sin(theta), 0)

   fig, ax = plt.subplots(figsize=(8, 4.5))
   t = np.linspace(0, 30, 3000)
   for v0 in np.linspace(0.3, 3.3, 11):
       sol = solve(eq, ivp={theta: -np.pi, theta.d(): v0}, tspan=(0, 30))
       ax.plot(sol.eval(theta, t), sol.eval(theta.d(), t), lw=1)

   for e in equilibria(eq):
       th = float(e.point[theta])
       marker = "ko" if e.stability.startswith("stable") else "kx"
       for shift in (-2*np.pi, 0, 2*np.pi):
           ax.plot(th + shift, 0, marker, ms=8)
       ax.annotate(e.stability, (th, 0), xytext=(6, -14), textcoords="offset points")

   ax.set_xlim(-np.pi - 0.3, 2*np.pi + 1.5)
   ax.set_xlabel(r"$\theta$")
   ax.set_ylabel(r"$\dot\theta$")
   plt.tight_layout()

Estimating parameters from measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Noisy measurements of a damped oscillation are used to estimate the
damping :math:`c`, the stiffness :math:`k` and the initial displacement
:math:`y_0` of :math:`y'' + c\,y' + k\,y = 0`. The data are generated here
from known values, so the result can be checked.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   import sympy as sp
   from odecast import var, Eq, solve, fit

   y = var("y")
   c, k, y0 = sp.symbols("c k y0")
   eq = Eq(y.d(2) + c*y.d() + k*y, 0)

   # Synthetic measurements
   true = {c: 0.4, k: 3.0, y0: 1.5}
   t_data = np.linspace(0, 10, 60)
   exact = solve(eq, ivp={y: y0, y.d(): 0}, tspan=(0, 10), params=true)
   y_data = exact.eval(y, t_data) + np.random.default_rng(0).normal(0, 0.05, t_data.size)

   result = fit(eq, {y: (t_data, y_data)}, guess={c: 1.0, k: 2.0, y0: 1.0},
                ivp={y: y0, y.d(): 0})
   print(result)

   t = np.linspace(0, 10, 500)
   plt.figure(figsize=(8, 3.5))
   plt.plot(t_data, y_data, "k.", label="measurements")
   plt.plot(t, result.solution.eval(y, t), label="fit")
   plt.xlabel("$t$")
   plt.legend()
   plt.tight_layout()

The fitted values deviate from the true ones by less than two standard
errors.

Symbolic solutions
------------------

Initial conditions
~~~~~~~~~~~~~~~~~~

With conditions, the SymPy backend determines the integration constants:

.. code-block:: python

   import sympy as sp
   from odecast import var, Eq, solve, t

   # Driven oscillator at resonance: y'' + y = cos(t), y(0) = y'(0) = 0
   y = var("y")
   sol = solve(Eq(y.d(2) + y, sp.cos(t)), backend="sympy", ivp={y: 0, y.d(): 0})
   sol.as_expr(y)   # t*sin(t)/2

The amplitude grows linearly in time.

General solution of a linear equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/02_symbolic_simple.py
   :language: python
   :linenos:

Mixed-order systems
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/03_mixed_orders.py
   :language: python
   :linenos:

Example scripts
---------------

The ``examples/`` directory contains:

``01_ivp_damped_oscillator.py``
   Numerical solution of the damped oscillator, with plots.
``02_symbolic_simple.py``
   Symbolic solution with the SymPy backend.
``03_mixed_orders.py``
   Coupled system with variables of different order.
``04_vector_harmonic_oscillator.py``
   Two-dimensional oscillator using a vector variable.
``05_vector_mixed_system.py``
   System combining scalar and vector variables.
``06_vector_simple.py``
   Introduction to vector variables.

Run a script with, e.g.,

.. code-block:: bash

   python examples/01_ivp_damped_oscillator.py
