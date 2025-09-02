import sympy as sp
from odecast import t, var, Eq, solve


def test_symbolic_solution_matches_form():
    y = var("y")
    eq = Eq(y.d(2) + y, 0)
    sol = solve(eq, backend="sympy")
    expr = sol.as_expr(y)  # SymPy expression in t
    assert isinstance(expr, sp.Expr)
    # Plug into the ODE: y'' + y -> 0
    tt = t.symbol
    assert sp.simplify(sp.diff(expr, tt, 2) + expr) == 0
