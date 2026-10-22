#!/usr/bin/env python3
"""
Vector/Matrix Variables Demo

This script demonstrates all the key vector variable features in odecast.
Run this to see a comprehensive showcase of vector functionality.
"""

from odecast import Eq, solve, var


def demo_vector_features():
    """Demonstrate all vector variable features."""
    print("Odecast Vector/Matrix Variables Demo")
    print("=" * 50)

    # Feature 1: Vector Variable Creation
    print("\n1. Vector Variable Creation")
    print("-" * 30)

    u = var("u", shape=2)
    v = var("v", shape=3)

    print(f"2D vector: {u} with components {u[0]}, {u[1]}")
    print(f"3D vector: {v} with components {v[0]}, {v[1]}, {v[2]}")

    # Feature 2: Vector Derivatives
    print("\n2. Vector Derivatives")
    print("-" * 20)

    u_dot = u.d()
    u_ddot = u.d(2)

    print(f"First derivative:  {u_dot}")
    print(f"Second derivative: {u_ddot}")
    print(f"Component access:  {u_dot.variable[0]}, {u_dot.variable[1]}")

    # Feature 3: Vector Equations
    print("\n3. Vector Equations & Auto-Expansion")
    print("-" * 35)

    eq = Eq(u.d(2) + 2 * u.d() + u, 0)
    print(f"Vector equation: {eq}")

    from odecast.equation import expand_vector_equations

    expanded = expand_vector_equations([eq])
    print("Auto-expands to:")
    for i, comp_eq in enumerate(expanded):
        print(f"  Component {i}: {comp_eq}")

    # Feature 4: Vector IVP Solving
    print("\n4. Vector IVP Solving")
    print("-" * 22)

    # Simple harmonic oscillator: u'' + u = 0
    eq_simple = Eq(u.d(2) + u, 0)

    ivp = {u: [1.0, 0.5], u.d(): [0.0, -0.2]}  # Initial position  # Initial velocity

    print("Solving: u'' + u = 0")
    print(f"IVP: {ivp}")

    sol = solve(eq_simple, ivp=ivp, tspan=(0, 2), backend="scipy")
    print(f"Solved! Got {len(sol.t)} time points")

    # Feature 5: Vector Solution Access
    print("\n5. Vector Solution Access")
    print("-" * 26)

    # Vector access
    u_traj = sol[u]  # 2×N array
    print(f"Vector access sol[u] shape: {u_traj.shape}")

    # Component access
    u0_traj = sol[u[0]]  # 1D array
    u1_traj = sol[u[1]]  # 1D array
    print(f"Component sol[u[0]] shape: {u0_traj.shape}")
    print(f"Component sol[u[1]] shape: {u1_traj.shape}")

    # Derivative access
    u_vel = sol[u.d()]  # 2×N array
    print(f"Velocity sol[u.d()] shape: {u_vel.shape}")

    # Feature 6: Vector Evaluation
    print("\n6. Vector Evaluation at Specific Times")
    print("-" * 40)

    eval_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    for t_val in eval_times:
        pos = sol.eval(u, t_val)
        vel = sol.eval(u.d(), t_val)
        print(f"t={t_val}: pos={pos}, vel={vel}")

    # Feature 7: Mixed Vector/Scalar Systems
    print("\n7. Mixed Vector/Scalar Systems")
    print("-" * 33)

    x = var("x")  # Scalar
    w = var("w", shape=2)  # Vector

    mixed_eqs = [
        Eq(x.d(2) + x, w[0]),  # Scalar driven by vector component
        Eq(w[0].d() + w[1], 0),  # Vector components coupled
        Eq(w[1].d() - w[0], x),  # Vector driven by scalar
    ]

    print("System:")
    for i, eq in enumerate(mixed_eqs):
        print(f"  {eq}")

    mixed_ivp = {
        x: 1.0,
        x.d(): 0.0,
        w: [0.1, 0.2],
        w.d(): [0.0, 0.0],  # Auto-filtered for first-order components
    }

    print(f"Mixed IVP: {mixed_ivp}")

    mixed_sol = solve(mixed_eqs, ivp=mixed_ivp, tspan=(0, 1), backend="scipy")
    print(f"Mixed system solved! {len(mixed_sol.t)} points")

    # Show final values
    x_final = mixed_sol.eval(x, 1.0)
    w_final = mixed_sol.eval(w, 1.0)
    print(f"Final values: x(1)={x_final:.4f}, w(1)={w_final}")

    # Feature 8: SymPy Backend with Vectors
    print("\n8. SymPy Backend with Vector Equations")
    print("-" * 38)

    try:
        # Simple decoupled vector system for SymPy
        simple_eq = Eq(u.d(2) + 4 * u, 0)  # u'' + 4u = 0

        print("Solving symbolically: u'' + 4u = 0")
        sym_sol = solve(simple_eq, backend="sympy")

        # Get symbolic expressions
        u0_expr = sym_sol.as_expr(u[0])
        u1_expr = sym_sol.as_expr(u[1])

        print("Symbolic solution:")
        print(f"  u[0](t) = {u0_expr}")
        print(f"  u[1](t) = {u1_expr}")

    except Exception as e:
        print(f"Symbolic solving failed: {e}")
        print("(This is expected for some complex cases)")


if __name__ == "__main__":
    demo_vector_features()
