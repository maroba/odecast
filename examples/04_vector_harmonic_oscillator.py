#!/usr/bin/env python3
"""
2D Harmonic Oscillator using Vector Variables

This example demonstrates solving a 2D harmonic oscillator using vector variables.
The system represents a mass connected to springs in both x and y directions.

Mathematical model:
    u'' + ω²u = 0

Where u = [x, y] is the position vector and ω is the angular frequency.

This expands to the component equations:
    x'' + ω²x = 0
    y'' + ω²y = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from odecast import t, var, Eq, solve


def main():
    print("2D Harmonic Oscillator Example")
    print("=" * 40)

    # System parameters
    omega = 2.0  # Angular frequency (rad/s)

    # Create 2D position vector variable
    u = var("u", shape=2)  # u = [x, y]

    # Define the vector equation: u'' + ω²u = 0
    eq = Eq(u.d(2) + omega**2 * u, 0)

    print(f"Vector equation: u'' + {omega}²u = 0")
    print("This automatically expands to:")
    print(f"  x'' + {omega}²x = 0")
    print(f"  y'' + {omega}²y = 0")
    print()

    # Initial conditions: circular motion
    # Position: u(0) = [1, 0]  (start at x=1, y=0)
    # Velocity: u'(0) = [0, 2] (initial velocity in y direction)
    ivp = {u: [1.0, 0.0], u.d(): [0.0, 2.0]}  # Initial position  # Initial velocity

    print("Initial conditions:")
    print("  Position: u(0) = [1.0, 0.0]")
    print("  Velocity: u'(0) = [0.0, 2.0]")
    print()

    # Solve numerically
    tspan = (0.0, 2 * np.pi / omega)  # One complete period
    print(f"Solving over time span: {tspan}")

    sol = solve(eq, ivp=ivp, tspan=tspan, backend="scipy")

    print(f"Solution computed at {len(sol.t)} time points")
    print()

    # Extract results
    times = sol.t

    # Vector access: get 2×N array
    positions = sol[u]  # Shape: (2, N)
    x_vals = positions[0, :]  # x component
    y_vals = positions[1, :]  # y component

    # Alternative: component access (equivalent)
    x_vals_alt = sol[u[0]]  # Direct access to x component
    y_vals_alt = sol[u[1]]  # Direct access to y component

    # Verify they're the same
    assert np.allclose(x_vals, x_vals_alt)
    assert np.allclose(y_vals, y_vals_alt)

    # Calculate velocities
    velocities = sol[u.d()]  # Get velocity vector
    vx_vals = velocities[0, :]
    vy_vals = velocities[1, :]

    # Energy analysis (should be conserved)
    kinetic_energy = 0.5 * (vx_vals**2 + vy_vals**2)
    potential_energy = 0.5 * omega**2 * (x_vals**2 + y_vals**2)
    total_energy = kinetic_energy + potential_energy

    print("Energy Analysis:")
    print(f"  Initial total energy: {total_energy[0]:.6f}")
    print(f"  Final total energy:   {total_energy[-1]:.6f}")
    print(f"  Energy conservation error: {abs(total_energy[-1] - total_energy[0]):.2e}")
    print()

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Phase space plot (trajectory)
    ax1.plot(x_vals, y_vals, "b-", linewidth=2, label="Trajectory")
    ax1.plot(x_vals[0], y_vals[0], "go", markersize=8, label="Start")
    ax1.plot(x_vals[-1], y_vals[-1], "ro", markersize=8, label="End")
    ax1.set_xlabel("x position")
    ax1.set_ylabel("y position")
    ax1.set_title("2D Trajectory (Phase Space)")
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")
    ax1.legend()

    # Position vs time
    ax2.plot(times, x_vals, "b-", label="x(t)", linewidth=2)
    ax2.plot(times, y_vals, "r-", label="y(t)", linewidth=2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Position")
    ax2.set_title("Position Components vs Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Velocity vs time
    ax3.plot(times, vx_vals, "b-", label="x'(t)", linewidth=2)
    ax3.plot(times, vy_vals, "r-", label="y'(t)", linewidth=2)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Velocity")
    ax3.set_title("Velocity Components vs Time")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Energy vs time
    ax4.plot(times, kinetic_energy, "g-", label="Kinetic", linewidth=2)
    ax4.plot(times, potential_energy, "r-", label="Potential", linewidth=2)
    ax4.plot(times, total_energy, "k--", label="Total", linewidth=2)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Energy")
    ax4.set_title("Energy vs Time")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    # Try symbolic solution for comparison
    try:
        print("Computing symbolic solution...")
        sol_symbolic = solve(eq, backend="sympy")

        print("Symbolic solution expressions:")
        x_expr = sol_symbolic.as_expr(u[0])
        y_expr = sol_symbolic.as_expr(u[1])
        print(f"  x(t) = {x_expr}")
        print(f"  y(t) = {y_expr}")

    except Exception as e:
        print(f"Symbolic solution failed: {e}")

    print("\nExample completed successfully!")
    print("Close the plot window to continue...")
    plt.show()


if __name__ == "__main__":
    main()
