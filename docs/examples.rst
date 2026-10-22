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

Bouncing ball
~~~~~~~~~~~~~

A ball dropped from a height of 1 m loses 20% of its speed at every bounce.
The bounce is an event that resets the velocity,
:math:`\dot y \to -0.8\,\dot y`. The bounces follow at intervals that
shrink by the factor 0.8.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, Event, solve

   y = var("y")
   bounce = Event(Eq(y, 0), direction=-1, reset={y.d(): -0.8*y.d()})
   sol = solve(Eq(y.d(2), -9.81), ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 4),
               events=bounce, rtol=1e-8, atol=1e-10)

   t = np.linspace(0, 4, 2000)
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
   ax1.plot(t, sol.eval(y, t))
   ax1.plot(sol.t_events[0], 0*sol.t_events[0], "k.", label="bounces")
   ax1.set_xlabel("$t$ [s]")
   ax1.set_ylabel("$y$ [m]")
   ax1.legend()
   ax2.plot(t, sol.eval(y.d(), t))
   ax2.set_xlabel("$t$ [s]")
   ax2.set_ylabel(r"$\dot y$ [m/s]")
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

Quantum harmonic oscillator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The stationary Schrödinger equation of the harmonic oscillator, in units
where :math:`\hbar = m = \omega = 1`,

.. math::

   \psi'' = (x^2 - 2E)\,\psi,

has normalizable solutions only for the energies
:math:`E_n = n + \tfrac12`. The energy is an unknown parameter of the
boundary value problem: besides :math:`\psi(\pm L) = 0`, the condition
:math:`\psi'(-L) = 10^{-6}` fixes the otherwise arbitrary amplitude. The
starting values select the eigenvalue. Here the independent variable is the
position; odecast still calls it :math:`t`.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   import sympy as sp
   from odecast import var, Eq, solve, BC, t

   psi = var("psi")
   E = sp.Symbol("E")
   L = 6
   eq = Eq(psi.d(2), (t**2 - 2*E)*psi)
   bcs = [BC(psi, t=-L, value=0), BC(psi, t=L, value=0),
          BC(psi.d(), t=-L, value=1e-6)]

   x = np.linspace(-L, L, 400)
   plt.figure(figsize=(8, 4))
   for n in range(4):
       hermite = np.polynomial.hermite.hermval(x, [0]*n + [1]) * np.exp(-x**2/2)
       sol = solve(eq, bvp=bcs, tspan=(-L, L), unknowns={E: n + 0.4}, mesh=200,
                   tol=1e-6, guess={psi: lambda s: np.interp(s, x, hermite)})
       energy = sol.unknowns[E]
       values = sol.eval(psi, x)
       values /= np.sqrt(np.sum(values**2) * (x[1] - x[0]))
       plt.plot(x, energy + 0.5*values, label=f"$E_{n} = {energy:.4f}$")
       plt.axhline(energy, color="0.8", linewidth=0.8)
   plt.plot(x, x**2/2, "k:", label="$V = x^2/2$")
   plt.ylim(0, 4.5)
   plt.xlim(-4, 4)
   plt.xlabel("$x$")
   plt.legend(loc="upper center", ncol=3, fontsize="small")
   plt.tight_layout()

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
focus, the upward one a saddle. The trajectory starting just beside the
upright position :math:`\theta = -\pi` swings through, settles at
:math:`0`, and shows how the saddles separate the basins of attraction.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   import sympy as sp
   from odecast import var, Eq, phase_portrait

   theta = var("theta")
   plt.figure(figsize=(8, 4.5))
   phase_portrait(
       Eq(theta.d(2) + 0.3*theta.d() + sp.sin(theta), 0),
       xlim=(-4, 7), ylim=(-3, 3),
       trajectories=[{theta: -np.pi + 0.01, theta.d(): 0},
                     {theta: -np.pi, theta.d(): 3.2}],
   )
   plt.tight_layout()

The second trajectory starts with enough energy to pass over the top at
:math:`\pi` once and settles at :math:`2\pi`. Only the equilibria :math:`0` and :math:`\pi` are
marked: :func:`sympy.solve` returns representatives of periodic solutions.

Hopf bifurcation in the Brusselator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Brusselator is a model of an oscillating chemical reaction,

.. math::

   \dot x = a - (b + 1)\,x + x^2 y, \qquad \dot y = b\,x - x^2 y.

Its equilibrium :math:`(a, b/a)` loses stability in a Hopf bifurcation at
:math:`b = 1 + a^2`, where a stable limit cycle appears. The continuation
finds the bifurcation; the limit cycles beyond it are computed with
:func:`~odecast.periodic_orbit`, and their range of :math:`x` is drawn into
the bifurcation diagram.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   import sympy as sp
   from odecast import var, Eq, continuation, periodic_orbit

   a = 1.0
   b = sp.Symbol("b")
   x, y = var("x"), var("y")
   eqs = [Eq(x.d(), a - (b + 1)*x + x**2*y), Eq(y.d(), b*x - x**2*y)]

   branch = continuation(eqs, b, start={x: a, y: 1/a, b: 1.0}, p_range=(1, 3.5))
   (hopf,) = branch.bifurcations

   bs, lows, highs, periods = [], [], [], []
   for bv in np.linspace(hopf.param + 0.02, 3.5, 12):
       orbit = periodic_orbit(eqs, ivp={x: 1.2, y: 2.0}, period=6.5,
                              params={b: bv}, transient=60)
       xs = orbit.solution.eval(x, np.linspace(0, orbit.period, 400))
       bs.append(bv)
       lows.append(xs.min())
       highs.append(xs.max())
       periods.append(orbit.period)

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
   branch.plot(x, ax=ax1)
   ax1.plot(bs, highs, "o-", color="tab:blue", markersize=3, label="limit cycle")
   ax1.plot(bs, lows, "o-", color="tab:blue", markersize=3)
   ax1.legend(fontsize="small")
   ax2.plot(bs, periods, "o-", markersize=3)
   ax2.axhline(2*np.pi/a, color="0.6", linestyle=":", label=r"$2\pi/a$")
   ax2.set_xlabel("b")
   ax2.set_ylabel("period")
   ax2.legend(fontsize="small")
   plt.tight_layout()

Near the bifurcation the period approaches :math:`2\pi/a`, given by the
imaginary part of the critical eigenvalues.

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

Delay equations
---------------

Mackey-Glass equation
~~~~~~~~~~~~~~~~~~~~~

The Mackey-Glass equation, a model of blood cell production,

.. math::

   \dot y = \frac{\beta\, y(t - \tau)}{1 + y(t - \tau)^{10}} - \gamma y,

becomes chaotic for long delays. With :math:`\tau = 17` the solution is
irregular, and the delay embedding :math:`(y(t), y(t - \tau))` shows a
strange attractor.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, solve

   y = var("y")
   tau = 17
   eq = Eq(y.d(), 0.2*y.delay(tau)/(1 + y.delay(tau)**10) - 0.1*y)
   sol = solve(eq, ivp={y: 1.2}, tspan=(0, 1000), rtol=1e-8, atol=1e-10)

   t = np.linspace(300, 1000, 7000)
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8),
                                  gridspec_kw={"width_ratios": [2, 1]})
   ax1.plot(t, sol.eval(y, t), linewidth=0.8)
   ax1.set_xlabel("$t$")
   ax1.set_ylabel("$y$")
   ax2.plot(sol.eval(y, t), sol.eval(y, t - tau), linewidth=0.4)
   ax2.set_xlabel("$y(t)$")
   ax2.set_ylabel(r"$y(t - \tau)$")
   plt.tight_layout()

Stochastic equations
--------------------

Langevin equation
~~~~~~~~~~~~~~~~~

A damped oscillator driven by white noise,

.. math::

   \ddot x + \gamma \dot x + \omega^2 x = \sigma \xi(t),

reaches a stationary state in which :math:`x` has the variance
:math:`\sigma^2 / (2\gamma\omega^2)`. The plot shows sample paths, the
ensemble mean and standard deviation, and the variance over time.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, solve_sde

   gamma, omega, sigma = 0.5, 2.0, 0.4
   x = var("x")
   sol = solve_sde(Eq(x.d(2) + gamma*x.d() + omega**2*x, 0), noise={x.d(): sigma},
                   ivp={x: 0.3, x.d(): 0}, tspan=(0, 20), dt=5e-3, paths=2000,
                   seed=1)

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
   sol.plot(x, paths=5, ax=ax1)
   ax2.plot(sol.t, sol.std(x)**2, label="ensemble")
   ax2.axhline(sigma**2/(2*gamma*omega**2), color="k", linestyle=":",
               label="stationary")
   ax2.set_xlabel("$t$")
   ax2.set_ylabel("Var $x$")
   ax2.legend()
   plt.tight_layout()

Partial differential equations
------------------------------

Travelling front of the Fisher-KPP equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Fisher-KPP equation :math:`u_t = D u_{xx} + r u (1 - u)` describes a
population that spreads by diffusion and grows logistically. From a
localized initial population a front develops that travels at the speed
:math:`2\sqrt{rD}`.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import Eq
   from odecast.pde import Grid, field, solve_pde, Neumann

   x = np.linspace(0, 100, 501)
   u = field("u", Grid(x=x))
   sol = solve_pde(Eq(u.d(), u.dx(2) + u*(1 - u)),
                   ic={u: lambda x: np.where(x < 5, 1.0, 0.0)},
                   bc={u: {"x": (Neumann(0), Neumann(0))}},
                   tspan=(0, 40))

   plt.figure(figsize=(8, 3.5))
   sol.plot(u, times=[0, 10, 20, 30, 40])
   plt.tight_layout()

Plucked string
~~~~~~~~~~~~~~

The wave equation :math:`u_{tt} = c^2 u_{xx}` for a string fixed at both
ends, plucked in the middle. The initial triangle splits into two waves
that travel to the ends and are reflected with opposite sign. The small
ripples behind the corners are numerical dispersion of the finite
differences; they shrink on a finer grid.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import Eq
   from odecast.pde import Grid, field, solve_pde, Dirichlet

   x = np.linspace(0, 1, 201)
   u = field("u", Grid(x=x))
   sol = solve_pde(Eq(u.d(2), u.dx(2)),
                   ic={u: lambda x: 0.1*(1 - np.abs(2*x - 1)), u.d(): 0},
                   bc={u: {"x": (Dirichlet(0), Dirichlet(0))}},
                   tspan=(0, 1), rtol=1e-6, atol=1e-9)

   plt.figure(figsize=(8, 3.5))
   sol.plot(u, times=[0, 0.125, 0.25, 0.5, 0.75])
   plt.tight_layout()

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
