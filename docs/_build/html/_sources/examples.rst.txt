Examples
========

This section provides comprehensive examples demonstrating Odecast's capabilities
across various domains.

Physics Examples
----------------

Damped Harmonic Oscillator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A classic example from mechanics: a mass-spring sys   ax4.plot(t_fine, kinetic, 'g-', linewidth=2, label='Kinetic')
   ax4.plot(t_fine, potential, 'r-', linewidth=2, label='Potential')
   ax4.plot(t_fine, total_energy, 'k-', linewidth=2, label='Total') with damping.

**Mathematical Model:**

The equation of motion for a damped harmonic oscillator is:

.. math::

   m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = 0

This can be written in standard form as:

.. math::

   \frac{d^2x}{dt^2} + 2\zeta\omega_n\frac{dx}{dt} + \omega_n^2 x = 0

where :math:`\omega_n = \sqrt{k/m}` is the natural frequency and :math:`\zeta = c/(2\sqrt{km})` is the damping ratio.

**Odecast Implementation:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, solve

   # Define the variable
   x = var("x")

   # System parameters
   omega_n = 2.0  # Natural frequency
   zeta = 0.3     # Damping ratio

   # Damped harmonic oscillator: x'' + 2*zeta*omega_n*x' + omega_n^2*x = 0
   equation = Eq(x.d(2) + 2*zeta*omega_n*x.d() + omega_n**2*x, 0)

   # Initial conditions: x(0) = 1, x'(0) = 0
   initial_conditions = {x: 1.0, x.d(): 0.0}

   # Solve the equation with fine resolution
   solution = solve(equation, ivp=initial_conditions, tspan=(0, 5), 
                   backend="scipy", max_step=0.01)

   # Create fine time grid for smooth plotting
   t_fine = np.linspace(0, 5, 1000)
   x_smooth = solution.eval(x, t_fine)
   xdot_smooth = solution.eval(x.d(), t_fine)

   # Create the plot
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   
   # Time series plot
   ax1.plot(t_fine, x_smooth, 'b-', linewidth=2, label='Position')
   ax1.plot(t_fine, xdot_smooth, 'r--', linewidth=2, label='Velocity')
   ax1.set_xlabel('Time (s)')
   ax1.set_ylabel('Amplitude')
   ax1.set_title('Damped Harmonic Oscillator')
   ax1.legend()
   ax1.grid(True, alpha=0.3)
   
   # Phase portrait
   ax2.plot(x_smooth, xdot_smooth, 'g-', linewidth=2)
   ax2.plot(x_smooth[0], xdot_smooth[0], 'ro', markersize=8, label='Start')
   ax2.plot(x_smooth[-1], xdot_smooth[-1], 'bs', markersize=8, label='End')
   ax2.set_xlabel('Position')
   ax2.set_ylabel('Velocity')
   ax2.set_title('Phase Portrait')
   ax2.legend()
   ax2.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

The solution shows the characteristic exponential decay of a damped oscillator. The phase portrait 
spirals inward to the equilibrium point at the origin.

Simple Pendulum
~~~~~~~~~~~~~~~

A nonlinear pendulum demonstrating chaotic behavior for large angles.

**Mathematical Model:**

The equation of motion for a simple pendulum is:

.. math::

   \frac{d^2\theta}{dt^2} + \frac{g}{L}\sin(\theta) = 0

For small angles, :math:`\sin(\theta) \approx \theta`, giving the linear approximation:

.. math::

   \frac{d^2\theta}{dt^2} + \frac{g}{L}\theta = 0

For this demonstration, we'll compare the linear case with a nonlinear variant 
that includes a cubic restoring force to show the effect of nonlinearity.

**Implementation:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   import sympy as sp
   from odecast import var, Eq, solve

   # Parameters
   g = 9.81  # acceleration due to gravity (m/s²)
   L = 1.0   # pendulum length (m)

   # Define the variable
   theta = var("theta")

   # Compare linear vs nonlinear pendulum (using small-angle approximation for demo)
   eq_linear = Eq(theta.d(2) + (g/L)*theta, 0)
   # For nonlinear demo, we'll use a cubic term to show nonlinearity
   eq_nonlinear = Eq(theta.d(2) + (g/L)*theta*(1 + 0.1*theta*theta), 0)

   # Initial conditions: large angle
   theta0 = np.pi/3  # 60 degrees
   initial_conditions = {theta: theta0, theta.d(): 0.0}

   # Solve both equations with fine resolution  
   sol_linear = solve(eq_linear, ivp=initial_conditions, tspan=(0, 10), 
                     backend="scipy", max_step=0.02)
   sol_nonlinear = solve(eq_nonlinear, ivp=initial_conditions, tspan=(0, 10), 
                        backend="scipy", max_step=0.02)

   # Create fine time grid for smooth evaluation
   t_fine = np.linspace(0, 10, 1000)
   theta_linear = sol_linear.eval(theta, t_fine)
   theta_nonlinear = sol_nonlinear.eval(theta, t_fine)
   thetadot_linear = sol_linear.eval(theta.d(), t_fine)
   thetadot_nonlinear = sol_nonlinear.eval(theta.d(), t_fine)

   # Create comparison plot
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   
   # Time series comparison
   ax1.plot(t_fine, theta_linear*180/np.pi, 'b--', 
           linewidth=2, label='Linear pendulum')
   ax1.plot(t_fine, theta_nonlinear*180/np.pi, 'r-', 
           linewidth=2, label='Nonlinear pendulum')
   ax1.set_xlabel('Time (s)')
   ax1.set_ylabel('Angle (degrees)')
   ax1.set_title('Linear vs Nonlinear Pendulum Dynamics')
   ax1.legend()
   ax1.grid(True, alpha=0.3)
   
   # Phase portraits
   ax2.plot(theta_linear*180/np.pi, thetadot_linear*180/np.pi, 
           'b--', linewidth=2, label='Linear')
   ax2.plot(theta_nonlinear*180/np.pi, thetadot_nonlinear*180/np.pi, 
           'r-', linewidth=2, label='Nonlinear')
   ax2.set_xlabel('Angle (degrees)')
   ax2.set_ylabel('Angular velocity (deg/s)')
   ax2.set_title('Phase Portraits')
   ax2.legend()
   ax2.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

The nonlinear pendulum shows a longer period than the linear approximation, 
especially for large initial angles.

2D Harmonic Oscillator
~~~~~~~~~~~~~~~~~~~~~~~

A particle moving in a 2D harmonic potential, demonstrating vector variables.

**Mathematical Model:**

The equations of motion for a 2D isotropic harmonic oscillator are:

.. math::

   \frac{d^2\mathbf{r}}{dt^2} + \omega^2 \mathbf{r} = 0

where :math:`\mathbf{r} = (x, y)` is the position vector and :math:`\omega` is the angular frequency.

**Implementation:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, solve

   # Define 2D position vector
   r = var("r", shape=2)

   # System parameter
   omega = 1.0  # Angular frequency

   # 2D harmonic oscillator: r'' + omega^2 * r = 0
   equation = Eq(r.d(2) + omega**2 * r, 0)

   # Initial conditions: elliptical orbit
   initial_conditions = {
       r: [1.0, 0.0],      # r(0) = [1, 0]
       r.d(): [0.0, 1.5]   # r'(0) = [0, 1.5]
   }

   # Solve the system with fine resolution
   solution = solve(equation, ivp=initial_conditions, tspan=(0, 4*np.pi), 
                   backend="scipy", max_step=0.05)
   
   # Create fine time grid for smooth evaluation
   t_fine = np.linspace(0, 4*np.pi, 1000)
   r_vals = solution.eval(r, t_fine)  # This returns a 2D array
   rdot_vals = solution.eval(r.d(), t_fine)
   
   # Extract x and y components
   x = r_vals[0]
   y = r_vals[1]
   vx = rdot_vals[0] 
   vy = rdot_vals[1]

   # Create visualization
   fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
   
   # Trajectory in phase space
   ax1.plot(x, y, 'b-', linewidth=2)
   ax1.plot(x[0], y[0], 'go', markersize=8, label='Start')
   ax1.plot(x[-1], y[-1], 'ro', markersize=8, label='End')
   ax1.set_xlabel('x position')
   ax1.set_ylabel('y position')
   ax1.set_title('2D Trajectory')
   ax1.legend()
   ax1.grid(True, alpha=0.3)
   ax1.set_aspect('equal')
   
   # Time series for x component
   ax2.plot(t_fine, x, 'r-', linewidth=2, label='x(t)')
   ax2.plot(t_fine, vx, 'r--', linewidth=2, label="x'(t)")
   ax2.set_xlabel('Time')
   ax2.set_ylabel('x component')
   ax2.set_title('X Component vs Time')
   ax2.legend()
   ax2.grid(True, alpha=0.3)
   
   # Time series for y component
   ax3.plot(t_fine, y, 'b-', linewidth=2, label='y(t)')
   ax3.plot(t_fine, vy, 'b--', linewidth=2, label="y'(t)")
   ax3.set_xlabel('Time')
   ax3.set_ylabel('y component')
   ax3.set_title('Y Component vs Time')
   ax3.legend()
   ax3.grid(True, alpha=0.3)
   
   # Energy conservation check
   kinetic = 0.5 * (vx**2 + vy**2)
   potential = 0.5 * omega**2 * (x**2 + y**2)
   total_energy = kinetic + potential
   
   ax4.plot(t_fine, kinetic, 'g-', linewidth=2, label='Kinetic')
   ax4.plot(t_fine, potential, 'r-', linewidth=2, label='Potential')
   ax4.plot(t_fine, total_energy, 'k--', linewidth=2, label='Total')
   ax4.set_xlabel('Time')
   ax4.set_ylabel('Energy')
   ax4.set_title('Energy Conservation')
   ax4.legend()
   ax4.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

The solution demonstrates perfect energy conservation and the characteristic 
elliptical motion of a 2D harmonic oscillator.

Biology Examples
----------------

Lotka-Volterra Model
~~~~~~~~~~~~~~~~~~~~

The classic predator-prey model from population dynamics.

**Mathematical Model:**

The Lotka-Volterra equations describe the dynamics of biological systems 
with predator and prey interactions:

.. math::

   \frac{dx}{dt} &= ax - bxy \\
   \frac{dy}{dt} &= -cy + dxy

where:
- :math:`x(t)` is the prey population
- :math:`y(t)` is the predator population  
- :math:`a, c > 0` are growth/death rates
- :math:`b, d > 0` are interaction coefficients

**Conservation Property:**

The system has a conserved quantity (Hamiltonian):

.. math::

   H(x,y) = d \cdot x + c \ln(x) + b \cdot y + a \ln(y) = \text{constant}

**Implementation:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, solve

   # Define variables
   x = var("x")  # Prey population
   y = var("y")  # Predator population

   # Model parameters
   a, b, c, d = 1.0, 0.1, 1.5, 0.075

   # Lotka-Volterra equations
   equations = [
       Eq(x.d() - a*x + b*x*y, 0),  # dx/dt = ax - bxy
       Eq(y.d() + c*y - d*x*y, 0)   # dy/dt = -cy + dxy
   ]

   # Initial conditions
   initial_conditions = {x: 10, y: 5}

   # Solve the system with fine resolution
   solution = solve(equations, ivp=initial_conditions, tspan=(0, 15), 
                   backend="scipy", max_step=0.05)

   # Create fine time grid for smooth evaluation
   t_fine = np.linspace(0, 15, 1000)
   x_vals = solution.eval(x, t_fine)
   y_vals = solution.eval(y, t_fine)

   # Create comprehensive visualization
   fig = plt.figure(figsize=(15, 10))
   
   # Population vs time
   ax1 = plt.subplot(2, 3, 1)
   plt.plot(t_fine, x_vals, 'b-', linewidth=2, label='Prey')
   plt.plot(t_fine, y_vals, 'r-', linewidth=2, label='Predator')
   plt.xlabel('Time')
   plt.ylabel('Population')
   plt.title('Population Dynamics')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   # Phase portrait
   ax2 = plt.subplot(2, 3, 2)
   plt.plot(x_vals, y_vals, 'g-', linewidth=2)
   plt.plot(x_vals[0], y_vals[0], 'go', markersize=8, label='Start')
   plt.plot(x_vals[-1], y_vals[-1], 'ro', markersize=8, label='End')
   plt.xlabel('Prey Population')
   plt.ylabel('Predator Population')
   plt.title('Phase Portrait')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   # Conservation quantity verification
   ax3 = plt.subplot(2, 3, 3)
   H = d*x_vals + c*np.log(x_vals) + b*y_vals + a*np.log(y_vals)
   plt.plot(t_fine, H, 'k-', linewidth=2)
   plt.xlabel('Time')
   plt.ylabel('H (conserved quantity)')
   plt.title('Conservation Check')
   plt.grid(True, alpha=0.3)
   
   # Growth rates
   ax4 = plt.subplot(2, 3, 4)
   prey_growth = a*x_vals - b*x_vals*y_vals
   pred_growth = -c*y_vals + d*x_vals*y_vals
   plt.plot(t_fine, prey_growth, 'b-', linewidth=2, label='Prey growth rate')
   plt.plot(t_fine, pred_growth, 'r-', linewidth=2, label='Predator growth rate')
   plt.xlabel('Time')
   plt.ylabel('Growth rate')
   plt.title('Population Growth Rates')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   # Vector field (direction field)
   ax5 = plt.subplot(2, 3, 5)
   x_range = np.linspace(2, 18, 15)
   y_range = np.linspace(2, 12, 12)
   X, Y = np.meshgrid(x_range, y_range)
   
   # Calculate direction field
   DX = a*X - b*X*Y
   DY = -c*Y + d*X*Y
   
   # Normalize arrows
   M = np.sqrt(DX**2 + DY**2)
   M[M == 0] = 1
   DX_norm = DX/M
   DY_norm = DY/M
   
   plt.quiver(X, Y, DX_norm, DY_norm, M, cmap='viridis', alpha=0.7)
   plt.plot(x_vals, y_vals, 'r-', linewidth=3, label='Solution trajectory')
   plt.xlabel('Prey Population')
   plt.ylabel('Predator Population')
   plt.title('Direction Field')
   plt.legend()
   
   # Equilibrium analysis
   ax6 = plt.subplot(2, 3, 6)
   # Equilibrium points: (0,0) and (c/d, a/b)
   eq_x, eq_y = c/d, a/b
   
   plt.plot(x_vals, y_vals, 'g-', linewidth=2, label='Trajectory')
   plt.plot(0, 0, 'ks', markersize=10, label='Extinction equilibrium')
   plt.plot(eq_x, eq_y, 'ro', markersize=10, label='Coexistence equilibrium')
   plt.xlabel('Prey Population')
   plt.ylabel('Predator Population')
   plt.title('Equilibrium Points')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

The Lotka-Volterra model shows periodic oscillations with perfect conservation 
of the Hamiltonian. The coexistence equilibrium at :math:`(c/d, a/b) = (20, 10)` 
is a center surrounded by closed orbits.

Engineering Examples
--------------------

RLC Circuit
~~~~~~~~~~~

An electrical circuit with resistance, inductance, and capacitance.

**Mathematical Model:**

The voltage equation for an RLC circuit is derived from Kirchhoff's voltage law:

.. math::

   L\frac{d^2q}{dt^2} + R\frac{dq}{dt} + \frac{q}{C} = V(t)

where:
- :math:`q(t)` is the charge on the capacitor
- :math:`L` is inductance (H)
- :math:`R` is resistance (Ω) 
- :math:`C` is capacitance (F)
- :math:`V(t)` is the applied voltage

The current is :math:`i(t) = dq/dt` and the capacitor voltage is :math:`v_C = q/C`.

**Circuit Analysis:**

This is a second-order system with characteristic equation:

.. math::

   Ls^2 + Rs + \frac{1}{C} = 0

The damping ratio is :math:`\zeta = \frac{R}{2}\sqrt{\frac{C}{L}}` and 
natural frequency is :math:`\omega_n = \frac{1}{\sqrt{LC}}`.

**Implementation:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from odecast import var, Eq, solve

   # Circuit parameters
   L = 1e-3   # Inductance (H)
   R = 10     # Resistance (Ω)
   C = 1e-6   # Capacitance (F)
   V0 = 12    # Applied voltage (V)

   # Calculate characteristic parameters
   omega_n = 1/np.sqrt(L*C)
   zeta = R/2 * np.sqrt(C/L)
   
   print(f"Natural frequency: {omega_n:.1f} rad/s")
   print(f"Damping ratio: {zeta:.3f}")

   # Define charge variable
   q = var("q")

   # RLC equation: L*q'' + R*q' + q/C = V0
   equation = Eq(L*q.d(2) + R*q.d() + q/C, V0)

   # Initial conditions: no initial charge or current
   initial_conditions = {q: 0, q.d(): 0}

   # Solve the circuit with fine resolution
   solution = solve(equation, ivp=initial_conditions, tspan=(0, 5e-3), 
                   backend="scipy", max_step=5e-6)

   # Create fine time grid for smooth evaluation
   t_fine = np.linspace(0, 5e-3, 1000)
   q_vals = solution.eval(q, t_fine)
   current_vals = solution.eval(q.d(), t_fine)
   
   # Calculate derived quantities
   v_capacitor = q_vals / C
   v_resistor = R * current_vals

   # Create comprehensive circuit analysis
   fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
   
   # Charge vs time
   ax1.plot(t_fine * 1000, q_vals * 1e6, 'b-', linewidth=2)
   ax1.set_xlabel('Time (ms)')
   ax1.set_ylabel('Charge (μC)')
   ax1.set_title('Capacitor Charge vs Time')
   ax1.grid(True, alpha=0.3)
   
   # Current vs time
   ax2.plot(t_fine * 1000, current_vals * 1000, 'r-', linewidth=2)
   ax2.set_xlabel('Time (ms)')
   ax2.set_ylabel('Current (mA)')
   ax2.set_title('Circuit Current vs Time')
   ax2.grid(True, alpha=0.3)
   
   # Voltage analysis
   ax3.plot(t_fine * 1000, v_capacitor, 'g-', linewidth=2, label='Capacitor')
   ax3.plot(t_fine * 1000, v_resistor, 'r-', linewidth=2, label='Resistor')
   ax3.axhline(y=V0, color='k', linestyle='--', linewidth=2, label='Applied')
   ax3.set_xlabel('Time (ms)')
   ax3.set_ylabel('Voltage (V)')
   ax3.set_title('Component Voltages')
   ax3.legend()
   ax3.grid(True, alpha=0.3)
   
   # Energy analysis
   energy_C = 0.5 * C * v_capacitor**2  # Capacitor energy
   energy_L = 0.5 * L * current_vals**2      # Inductor energy
   power_R = R * current_vals**2             # Resistor power dissipation
   
   ax4.plot(t_fine * 1000, energy_C * 1e9, 'g-', linewidth=2, label='Capacitor energy')
   ax4.plot(t_fine * 1000, energy_L * 1e9, 'b-', linewidth=2, label='Inductor energy')
   ax4_twin = ax4.twinx()
   ax4_twin.plot(t_fine * 1000, power_R * 1000, 'r-', linewidth=2, label='Resistor power')
   
   ax4.set_xlabel('Time (ms)')
   ax4.set_ylabel('Energy (nJ)', color='k')
   ax4_twin.set_ylabel('Power (mW)', color='r')
   ax4.set_title('Energy Storage and Power Dissipation')
   ax4.legend(loc='upper left')
   ax4_twin.legend(loc='upper right')
   ax4.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

The circuit shows :math:`\zeta < 1` (underdamped) behavior with oscillatory 
approach to steady state. Energy oscillates between capacitor and inductor 
while being dissipated in the resistor.

Control Systems
~~~~~~~~~~~~~~~

A second-order control system with PID controller.

**Mathematical Model:**

.. math::

   \ddot{y} + 2\zeta\omega_n\dot{y} + \omega_n^2 y = \omega_n^2 u

Where :math:`u` is the control input.

**Implementation:**

.. code-block:: python

   from odecast import var, Eq, solve
   import numpy as np
   import matplotlib.pyplot as plt

   # System parameters
   omega_n = 2.0  # Natural frequency
   zeta = 0.1     # Damping ratio

   # Define variables
   y = var("y")     # Output
   r = 1.0          # Reference (step input)

   # PID controller parameters
   Kp, Ki, Kd = 1.0, 0.5, 0.1

   # For simplicity, we'll solve the closed-loop system directly
   # Open-loop transfer function: G(s) = ωn²/(s² + 2ζωn*s + ωn²)
   # With unity feedback: y'' + 2ζωn*y' + ωn²*y = ωn²*r
   
   equation = Eq(y.d(2) + 2*zeta*omega_n*y.d() + omega_n**2*y, omega_n**2*r)

   # Initial conditions: system at rest
   initial_conditions = {y: 0, y.d(): 0}

   # Solve
   solution = solve(equation, ivp=initial_conditions, tspan=(0, 5))

   # Plot step response
   plt.figure(figsize=(10, 6))
   plt.plot(solution.t, solution[y], label='System Output')
   plt.axhline(y=r, color='r', linestyle='--', label='Reference')
   plt.xlabel('Time (s)')
   plt.ylabel('Output')
   plt.title('Step Response of Second-Order System')
   plt.legend()
   plt.grid(True)
   plt.show()

   # Calculate performance metrics
   steady_state_value = solution[y][-1]
   steady_state_error = abs(r - steady_state_value)
   print(f"Steady-state error: {steady_state_error:.3f}")

Symbolic Examples
-----------------

Exact Solutions
~~~~~~~~~~~~~~~

Using SymPy backend for exact symbolic solutions.

.. literalinclude:: ../examples/02_symbolic_simple.py
   :language: python
   :linenos:

Mixed Systems
~~~~~~~~~~~~~

Systems with different equation orders.

.. literalinclude:: ../examples/03_mixed_orders.py
   :language: python
   :linenos:

Complete Example Scripts
------------------------

All examples shown above are available as complete, runnable scripts in the 
``examples/`` directory of the Odecast repository:

* ``01_ivp_damped_oscillator.py`` - Damped harmonic oscillator with visualization
* ``02_symbolic_simple.py`` - Symbolic solutions using SymPy
* ``03_mixed_orders.py`` - Systems with mixed derivative orders
* ``04_vector_harmonic_oscillator.py`` - 2D harmonic oscillator
* ``05_vector_mixed_system.py`` - Mixed scalar-vector systems
* ``06_vector_simple.py`` - Introduction to vector variables

To run any example:

.. code-block:: bash

   python examples/01_ivp_damped_oscillator.py
