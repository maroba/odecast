#!/usr/bin/env python3
"""
Example 1: Damped Harmonic Oscillator (Numeric IVP)

This example demonstrates solving a second-order ODE representing a damped
harmonic oscillator using the numeric SciPy backend.

The equation: y'' + 0.3*y' + y = 0
with initial conditions: y(0) = 1, y'(0) = 0

This represents a mass-spring-damper system with:
- Natural frequency: ω₀ = 1 rad/s
- Damping ratio: ζ = 0.15 (underdamped)
"""

import numpy as np
import matplotlib.pyplot as plt
from odecast import t, var, Eq, solve


def main():
    print("🎯 Damped Harmonic Oscillator Example")
    print("=" * 50)

    # Define the dependent variable
    y = var("y")

    # Create the differential equation: y'' + 0.3*y' + y = 0
    eq = Eq(y.d(2) + 0.3 * y.d() + y, 0)
    print(f"Equation: {eq}")

    # Set initial conditions: y(0) = 1, y'(0) = 0
    initial_conditions = {y: 1.0, y.d(): 0.0}  # Initial position  # Initial velocity
    print(
        f"Initial conditions: y(0) = {initial_conditions[y]}, y'(0) = {initial_conditions[y.d()]}"
    )

    # Solve over time span [0, 20] seconds
    time_span = (0.0, 20.0)
    print(f"Time span: {time_span}")

    # Solve using the SciPy numeric backend
    print("\n🔄 Solving...")
    sol = solve(eq, ivp=initial_conditions, tspan=time_span, backend="scipy")

    # Extract solution data
    t_vals = sol.t
    y_vals = sol[y]  # Position
    yprime_vals = sol[y.d()]  # Velocity

    print(f"✅ Solution computed with {len(t_vals)} time points")
    print(f"   Final position: y({t_vals[-1]:.1f}) = {y_vals[-1]:.6f}")
    print(f"   Final velocity: y'({t_vals[-1]:.1f}) = {yprime_vals[-1]:.6f}")

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot position vs time
    ax1.plot(t_vals, y_vals, "b-", linewidth=2, label="Position y(t)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position")
    ax1.set_title("Damped Harmonic Oscillator - Position")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot velocity vs time
    ax2.plot(t_vals, yprime_vals, "r-", linewidth=2, label="Velocity y'(t)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity")
    ax2.set_title("Damped Harmonic Oscillator - Velocity")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Demonstrate phase space plot
    plt.figure(figsize=(8, 6))
    plt.plot(y_vals, yprime_vals, "g-", linewidth=2, alpha=0.8)
    plt.plot(y_vals[0], yprime_vals[0], "go", markersize=8, label="Start")
    plt.plot(y_vals[-1], yprime_vals[-1], "rs", markersize=8, label="End")
    plt.xlabel("Position y")
    plt.ylabel("Velocity y'")
    plt.title("Phase Space Plot")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.show()

    # Demonstrate solution evaluation at specific times
    print("\n📊 Solution evaluation at specific times:")
    eval_times = [0, 5, 10, 15, 20]
    for t_eval in eval_times:
        y_eval = sol.eval(y, t_eval)
        yprime_eval = sol.eval(y.d(), t_eval)
        print(f"   t = {t_eval:2.0f}s: y = {y_eval:8.4f}, y' = {yprime_eval:8.4f}")


if __name__ == "__main__":
    main()
