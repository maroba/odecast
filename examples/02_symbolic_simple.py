#!/usr/bin/env python3
"""
Example 2: Symbolic Solution (SymPy Backend)

This example demonstrates solving a simple ODE symbolically using the SymPy backend
to obtain a closed-form analytical solution.

The equation: y'' + y = 0 (simple harmonic oscillator)

This is a classic example where symbolic solution provides exact expressions
rather than numerical approximations.
"""

import sympy as sp
from odecast import t, var, Eq, solve


def main():
    print("🎯 Symbolic Solution Example")
    print("=" * 40)

    # Define the dependent variable
    y = var("y")

    # Create the differential equation: y'' + y = 0
    eq = Eq(y.d(2) + y, 0)
    print(f"Equation: {eq}")
    print("This is the simple harmonic oscillator equation.")

    # Solve symbolically using SymPy backend
    print("\n🔄 Solving symbolically...")
    sol = solve(eq, backend="sympy")

    # Get the symbolic expression
    expr = sol.as_expr(y)
    print(f"✅ Symbolic solution: y(t) = {expr}")

    # Verify the solution by substituting back into the ODE
    print("\n🔍 Verification:")
    t_sym = t.symbol  # Get the SymPy symbol for t

    # Compute derivatives
    y_expr = expr
    yprime_expr = sp.diff(y_expr, t_sym)
    ydoubleprime_expr = sp.diff(y_expr, t_sym, 2)

    print(f"y(t)    = {y_expr}")
    print(f"y'(t)   = {yprime_expr}")
    print(f"y''(t)  = {ydoubleprime_expr}")

    # Check if y'' + y = 0
    lhs = ydoubleprime_expr + y_expr
    simplified = sp.simplify(lhs)
    print(f"y''(t) + y(t) = {lhs} = {simplified}")

    if simplified == 0:
        print("✅ Verification successful: the solution satisfies the ODE!")
    else:
        print("❌ Verification failed!")

    # Show general form with constants
    print(f"\n📝 General solution form:")
    print(f"The solution contains arbitrary constants C1 and C2,")
    print(f"which would be determined by initial or boundary conditions.")
    print(f"General form: y(t) = C1*cos(t) + C2*sin(t)")

    # Demonstrate symbolic manipulation
    print(f"\n🔧 Symbolic manipulation examples:")

    # Extract coefficients if possible
    try:
        # Try to collect terms (this depends on SymPy's output format)
        cos_coeff = expr.coeff(sp.cos(t_sym), 1)
        sin_coeff = expr.coeff(sp.sin(t_sym), 1)

        if cos_coeff is not None:
            print(f"Coefficient of cos(t): {cos_coeff}")
        if sin_coeff is not None:
            print(f"Coefficient of sin(t): {sin_coeff}")
    except:
        print("Coefficients depend on the specific form SymPy returns")

    # Show derivative relationships
    print(f"\nDerivative relationship:")
    print(f"If y(t) = A*cos(t) + B*sin(t), then:")
    print(f"y'(t) = -A*sin(t) + B*cos(t)")
    print(f"y''(t) = -A*cos(t) - B*sin(t) = -y(t)")
    print(f"Therefore: y''(t) + y(t) = 0 ✓")

    # Demonstrate evaluation at specific points
    print(f"\n📊 Symbolic evaluation:")
    eval_points = [0, sp.pi / 4, sp.pi / 2, sp.pi]
    for t_val in eval_points:
        y_val = expr.subs(t_sym, t_val)
        y_val_simplified = sp.simplify(y_val)
        print(f"y({t_val}) = {y_val_simplified}")


if __name__ == "__main__":
    main()
