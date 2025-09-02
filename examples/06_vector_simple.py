#!/usr/bin/env python3
"""
Simple Vector Variable Example

This is a minimal example demonstrating vector variables in odecast.
It shows the basic syntax and usage patterns for creating and solving
systems with vector variables.

Mathematical model:
    u'' + u = 0    (vector equation)

This represents a 2D harmonic oscillator where u = [x, y].
"""

from odecast import t, var, Eq, solve
import numpy as np


def main():
    print("Simple Vector Variable Example")
    print("=" * 35)

    # Create a 2D vector variable
    u = var("u", shape=2)
    print(f"Created vector variable: {u}")
    print(f"Components: {u[0]}, {u[1]}")
    print()

    # Create a vector equation: u'' + u = 0
    eq = Eq(u.d(2) + u, 0)
    print(f"Vector equation: {eq}")
    print("This automatically expands to component equations:")
    print(f"  {u[0].name}'' + {u[0].name} = 0")
    print(f"  {u[1].name}'' + {u[1].name} = 0")
    print()

    # Set up initial conditions in vector form
    initial_position = [1.0, 0.0]  # u(0) = [1, 0]
    initial_velocity = [0.0, 1.0]  # u'(0) = [0, 1]

    ivp = {u: initial_position, u.d(): initial_velocity}

    print("Initial conditions:")
    print(f"  Position: u(0) = {initial_position}")
    print(f"  Velocity: u'(0) = {initial_velocity}")
    print()

    # Solve the system
    print("Solving...")
    sol = solve(eq, ivp=ivp, tspan=(0, 6), backend="scipy")
    print(f"Solution computed at {len(sol.t)} time points")
    print()

    # Access solutions in different ways
    print("Accessing solution data:")

    # 1. Vector access - get both components at once
    positions = sol[u]  # 2×N array
    print(f"Vector access sol[u] shape: {positions.shape}")

    # 2. Component access - get individual components
    x_vals = sol[u[0]]  # 1D array
    y_vals = sol[u[1]]  # 1D array
    print(f"Component access sol[u[0]] shape: {x_vals.shape}")
    print(f"Component access sol[u[1]] shape: {y_vals.shape}")

    # 3. Derivative access - get velocities
    velocities = sol[u.d()]  # 2×N array
    print(f"Vector derivative sol[u.d()] shape: {velocities.shape}")
    print()

    # Show some values
    print("Sample values:")
    print("Time    x       y       vx      vy")
    print("----  ------  ------  ------  ------")

    times = sol.t
    for i in [0, len(times) // 4, len(times) // 2, 3 * len(times) // 4, -1]:
        t_val = times[i]
        x = positions[0, i]  # x component from vector
        y = positions[1, i]  # y component from vector
        vx = velocities[0, i]  # vx component from vector derivative
        vy = velocities[1, i]  # vy component from vector derivative
        print(f"{t_val:4.1f}  {x:6.3f}  {y:6.3f}  {vx:6.3f}  {vy:6.3f}")

    print()

    # Evaluate at specific times
    print("Evaluation at specific times:")
    eval_times = [0.0, 1.5, 3.0, 4.5, 6.0]

    for t_eval in eval_times:
        pos = sol.eval(u, t_eval)  # Returns [x, y] at time t_eval
        vel = sol.eval(u.d(), t_eval)  # Returns [vx, vy] at time t_eval
        print(f"t={t_eval}: position={pos}, velocity={vel}")

    print()
    print("Key features demonstrated:")
    print("  ✓ Vector variable creation with var('u', shape=2)")
    print("  ✓ Vector equation notation: u.d(2) + u = 0")
    print("  ✓ Vector initial conditions: {u: [...], u.d(): [...]}")
    print("  ✓ Vector solution access: sol[u] returns 2×N array")
    print("  ✓ Component access: sol[u[0]], sol[u[1]] return 1D arrays")
    print("  ✓ Vector derivative access: sol[u.d()] returns 2×N array")
    print("  ✓ Evaluation at specific times with sol.eval()")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
