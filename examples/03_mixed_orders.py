#!/usr/bin/env python3
"""
Example 3: Coupled Mixed-Order System

This example demonstrates solving a system of coupled differential equations
with different orders using the numeric backend.

System:
- x'' + 0.1*x' + x - 0.5*y = 0  (second-order in x)
- y' + 2*y + 0.3*x = 0          (first-order in y)

This represents a coupled oscillator system where one variable affects the other.
"""

import numpy as np
import matplotlib.pyplot as plt
from odecast import t, var, Eq, solve


def main():
    print("🎯 Coupled Mixed-Order System Example")
    print("=" * 45)

    # Define the dependent variables
    x = var("x")
    y = var("y")

    # Create the system of coupled equations
    eq1 = Eq(x.d(2) + 0.1 * x.d() + x - 0.5 * y, 0)  # Second-order in x
    eq2 = Eq(y.d() + 2 * y + 0.3 * x, 0)  # First-order in y

    equations = [eq1, eq2]

    print("System of equations:")
    for i, eq in enumerate(equations, 1):
        print(f"  Eq {i}: {eq}")

    print(f"\nVariable orders:")
    print(f"  x: 2nd order (requires x(0) and x'(0))")
    print(f"  y: 1st order (requires y(0))")

    # Set initial conditions
    initial_conditions = {
        x: 1.0,  # x(0) = 1
        x.d(): 0.0,  # x'(0) = 0
        y: 0.5,  # y(0) = 0.5
    }

    print(f"\nInitial conditions:")
    print(f"  x(0) = {initial_conditions[x]}")
    print(f"  x'(0) = {initial_conditions[x.d()]}")
    print(f"  y(0) = {initial_conditions[y]}")

    # Solve over time span [0, 15] seconds
    time_span = (0.0, 15.0)
    print(f"\nTime span: {time_span}")

    # Solve using the SciPy numeric backend
    print("\n🔄 Solving coupled system...")
    sol = solve(equations, ivp=initial_conditions, tspan=time_span, backend="scipy")

    # Extract solution data
    t_vals = sol.t
    x_vals = sol[x]  # Position x
    xprime_vals = sol[x.d()]  # Velocity x'
    y_vals = sol[y]  # Variable y

    print(f"✅ Solution computed with {len(t_vals)} time points")
    print(f"   Final values at t = {t_vals[-1]:.1f}:")
    print(f"     x({t_vals[-1]:.1f}) = {x_vals[-1]:.6f}")
    print(f"     x'({t_vals[-1]:.1f}) = {xprime_vals[-1]:.6f}")
    print(f"     y({t_vals[-1]:.1f}) = {y_vals[-1]:.6f}")

    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot x position vs time
    ax1.plot(t_vals, x_vals, "b-", linewidth=2, label="x(t)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("x Position")
    ax1.set_title("Variable x vs Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot y vs time
    ax2.plot(t_vals, y_vals, "r-", linewidth=2, label="y(t)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("y Value")
    ax2.set_title("Variable y vs Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot x velocity vs time
    ax3.plot(t_vals, xprime_vals, "g-", linewidth=2, label="x'(t)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("x Velocity")
    ax3.set_title("Velocity of x vs Time")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot phase space for x (position vs velocity)
    ax4.plot(x_vals, xprime_vals, "m-", linewidth=2, alpha=0.8)
    ax4.plot(x_vals[0], xprime_vals[0], "go", markersize=8, label="Start")
    ax4.plot(x_vals[-1], xprime_vals[-1], "rs", markersize=8, label="End")
    ax4.set_xlabel("x Position")
    ax4.set_ylabel("x Velocity (x')")
    ax4.set_title("Phase Space of x")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()

    # Show coupling effects
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, x_vals, "b-", linewidth=2, label="x(t)", alpha=0.8)
    plt.plot(t_vals, y_vals, "r-", linewidth=2, label="y(t)", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("Coupled Variables x and y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Demonstrate solution evaluation
    print("\n📊 Solution evaluation at specific times:")
    eval_times = [0, 3, 6, 9, 12, 15]
    print("   Time   x(t)      x'(t)     y(t)")
    print("   " + "-" * 35)
    for t_eval in eval_times:
        x_eval = sol.eval(x, t_eval)
        xprime_eval = sol.eval(x.d(), t_eval)
        y_eval = sol.eval(y, t_eval)
        print(f"   {t_eval:4.0f}   {x_eval:8.4f}  {xprime_eval:8.4f}  {y_eval:8.4f}")

    # Show system properties
    print(f"\n🔧 System Analysis:")
    print(f"This coupled system demonstrates:")
    print(f"  • Mixed orders: x is 2nd order, y is 1st order")
    print(f"  • Coupling: x affects y through the +0.3*x term")
    print(f"  • Coupling: y affects x through the -0.5*y term")
    print(f"  • Damping: x has damping (0.1*x' term), y has decay (2*y term)")

    # Inspect the first-order system
    print(f"\n🔍 First-order system inspection:")
    f, jac, x0, t0, mapping = sol.as_first_order()
    print(f"  State vector dimension: {len(x0)}")
    print(f"  State mapping:")
    print(f"    x    -> index {mapping[(x, 0)]}")
    print(f"    x'   -> index {mapping[(x, 1)]}")
    print(f"    y    -> index {mapping[(y, 0)]}")
    print(f"  Initial state vector: {x0}")


if __name__ == "__main__":
    main()
