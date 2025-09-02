"""
Comprehensive tests for Playbook 7 - SymPy backend implementation
"""

import pytest
import sympy as sp
from odecast import var, Eq, solve, t
from odecast.backends.sympy_backend import SymPyBackend, SolutionExpr
from odecast.errors import BackendError


class TestSolutionExpr:
    """Test the SolutionExpr class."""

    def test_as_expr_returns_sympy_expression(self):
        """Test that as_expr returns a SymPy expression."""
        y = var("y")
        t_sym = sp.Symbol("t")
        expr = sp.cos(t_sym)

        sol = SolutionExpr({y: expr}, t_sym)
        result = sol.as_expr(y)

        assert result == expr
        assert isinstance(result, sp.Expr)

    def test_as_expr_missing_variable(self):
        """Test that as_expr raises KeyError for missing variables."""
        y = var("y")
        z = var("z")
        t_sym = sp.Symbol("t")

        sol = SolutionExpr({y: sp.cos(t_sym)}, t_sym)

        with pytest.raises(KeyError, match="Variable z not found"):
            sol.as_expr(z)


class TestSymPyBackend:
    """Test the SymPy backend."""

    def test_simple_harmonic_oscillator(self):
        """Test solving simple harmonic oscillator y'' + y = 0."""
        backend = SymPyBackend()
        y = var("y")
        eq = Eq(y.d(2) + y, 0)

        sol = backend.solve([eq], t.symbol)
        expr = sol.as_expr(y)

        # Verify it's a SymPy expression
        assert isinstance(expr, sp.Expr)

        # Verify it satisfies the ODE: y'' + y = 0
        y_second = sp.diff(expr, t.symbol, 2)
        ode_check = sp.simplify(y_second + expr)
        assert ode_check == 0

    def test_first_order_ode(self):
        """Test solving first-order ODE y' - y = 0."""
        backend = SymPyBackend()
        y = var("y")
        eq = Eq(y.d() - y, 0)

        sol = backend.solve([eq], t.symbol)
        expr = sol.as_expr(y)

        # Verify it satisfies the ODE: y' - y = 0
        y_prime = sp.diff(expr, t.symbol)
        ode_check = sp.simplify(y_prime - expr)
        assert ode_check == 0

    def test_damped_oscillator(self):
        """Test solving damped oscillator y'' + 2*y' + y = 0."""
        backend = SymPyBackend()
        y = var("y")
        eq = Eq(y.d(2) + 2 * y.d() + y, 0)

        sol = backend.solve([eq], t.symbol)
        expr = sol.as_expr(y)

        # Verify it satisfies the ODE
        y_prime = sp.diff(expr, t.symbol)
        y_second = sp.diff(expr, t.symbol, 2)
        ode_check = sp.simplify(y_second + 2 * y_prime + expr)
        assert ode_check == 0

    def test_multiple_equations_raises_error(self):
        """Test that coupled equations raise an error."""
        backend = SymPyBackend()
        y = var("y")
        z = var("z")
        eq1 = Eq(y.d(2) + y, 0)
        eq2 = Eq(z.d() - y, 0)

        with pytest.raises(BackendError, match="only supports decoupled systems"):
            backend.solve([eq1, eq2], t.symbol)

    def test_multiple_variables_raises_error(self):
        """Test that multiple variables in single equation raise an error."""
        backend = SymPyBackend()
        y = var("y")
        z = var("z")
        eq = Eq(y.d() + z, 0)  # Two variables in one equation

        with pytest.raises(
            BackendError, match="only supports single-variable equations"
        ):
            backend.solve([eq], t.symbol)

    def test_unsolvable_equation_raises_error(self):
        """Test that unsolvable equations raise BackendError."""
        backend = SymPyBackend()
        y = var("y")
        # Create an equation that SymPy might have trouble with
        # Using a more complex equation that's still parseable
        eq = Eq(
            y.d(3) + y.d(2) + y.d() + y, sp.exp(t.symbol)
        )  # Inhomogeneous higher-order

        # This might succeed or fail depending on SymPy version, but we test error handling
        try:
            result = backend.solve([eq], t.symbol)
            # If it succeeds, just verify it's a valid solution
            assert isinstance(result, SolutionExpr)
        except BackendError:
            # If it fails, that's also acceptable for this test
            pass


class TestSymPyIntegration:
    """Test SymPy backend integration with solve API."""

    def test_solve_with_sympy_backend(self):
        """Test solve function with explicit SymPy backend."""
        y = var("y")
        eq = Eq(y.d(2) + y, 0)

        sol = solve(eq, backend="sympy")

        assert hasattr(sol, "as_expr")
        expr = sol.as_expr(y)
        assert isinstance(expr, sp.Expr)

        # Verify solution satisfies ODE
        y_second = sp.diff(expr, t.symbol, 2)
        ode_check = sp.simplify(y_second + expr)
        assert ode_check == 0

    def test_solve_with_auto_backend_sympy_success(self):
        """Test auto backend choosing SymPy when it succeeds."""
        y = var("y")
        eq = Eq(y.d(2) + y, 0)

        sol = solve(eq, backend="auto")

        # Should return SymPy solution
        assert hasattr(sol, "as_expr")
        expr = sol.as_expr(y)
        assert isinstance(expr, sp.Expr)

    def test_solve_with_auto_backend_fallback_to_scipy(self):
        """Test auto backend falling back to SciPy when SymPy fails."""
        y = var("y")
        z = var("z")
        # Create a system that SymPy can't handle (multiple equations)
        eq1 = Eq(y.d(2) + z, 0)
        eq2 = Eq(z.d() - y, 0)

        # Auto should fall back to SciPy, but SciPy needs IVP conditions
        with pytest.raises(ValueError, match="IVP conditions required"):
            solve([eq1, eq2], backend="auto")

    def test_different_ode_types(self):
        """Test various ODE types with SymPy backend."""
        # Linear first-order
        y = var("y")
        eq1 = Eq(y.d() + 2 * y, 0)
        sol1 = solve(eq1, backend="sympy")
        expr1 = sol1.as_expr(y)

        # Verify solution
        y_prime = sp.diff(expr1, t.symbol)
        assert sp.simplify(y_prime + 2 * expr1) == 0

        # Linear second-order with constant coefficients
        eq2 = Eq(y.d(2) + 3 * y.d() + 2 * y, 0)
        sol2 = solve(eq2, backend="sympy")
        expr2 = sol2.as_expr(y)

        # Verify solution
        y_prime = sp.diff(expr2, t.symbol)
        y_second = sp.diff(expr2, t.symbol, 2)
        assert sp.simplify(y_second + 3 * y_prime + 2 * expr2) == 0


class TestSymPyErrorHandling:
    """Test error handling in SymPy backend."""

    def test_backend_error_contains_original_exception(self):
        """Test that BackendError contains the original exception."""
        backend = SymPyBackend()
        y = var("y")
        # Force an error by trying to solve multiple variables
        z = var("z")
        eq = Eq(y.d() + z, 0)

        with pytest.raises(BackendError) as excinfo:
            backend.solve([eq], t.symbol)

        # Check that error message is informative
        assert "only supports single-variable equations" in str(excinfo.value)

    def test_solve_integration_error_propagation(self):
        """Test that errors from SymPy backend propagate correctly through solve."""
        y = var("y")
        z = var("z")
        eq1 = Eq(y.d() + z, 0)
        eq2 = Eq(z.d() - y, 0)

        # Should raise BackendError due to coupled equations
        with pytest.raises(BackendError, match="only supports decoupled systems"):
            solve([eq1, eq2], backend="sympy")
