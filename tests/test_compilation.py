"""
Unit tests for Playbook 5 functionality - compilation and SciPy backend
"""

import pytest
import numpy as np
import sympy as sp
from odecast import var, Eq, solve, t
from odecast.compile import lambdify_rhs, lambdify_jac
from odecast.backends.scipy_ivp import ScipyIVPBackend, convert_ivp_to_state_vector
from odecast.solution import SolutionIVP
from odecast.reduce import build_state_map
from odecast.errors import MissingInitialConditionError


class TestCompilation:
    """Test the compilation functions."""

    def test_lambdify_rhs_simple(self):
        """Test lambdify_rhs with simple expressions."""
        # Create a simple RHS: [x1, -x0]
        x0, x1 = sp.symbols("x0 x1")
        f_sym_vec = sp.Matrix([x1, -x0])

        f_compiled = lambdify_rhs(f_sym_vec, t.symbol, [x0, x1])

        # Test the compiled function
        result = f_compiled(0.0, np.array([1.0, 2.0]))
        expected = np.array([2.0, -1.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_lambdify_rhs_with_time(self):
        """Test lambdify_rhs with time dependence."""
        x0, x1 = sp.symbols("x0 x1")
        # RHS with time dependence: [x1, -x0 + sin(t)]
        f_sym_vec = sp.Matrix([x1, -x0 + sp.sin(t.symbol)])

        f_compiled = lambdify_rhs(f_sym_vec, t.symbol, [x0, x1])

        # Test at t=0
        result = f_compiled(0.0, np.array([1.0, 2.0]))
        expected = np.array([2.0, -1.0])  # sin(0) = 0

        np.testing.assert_array_almost_equal(result, expected)

        # Test at t=π/2
        result = f_compiled(np.pi / 2, np.array([1.0, 2.0]))
        expected = np.array([2.0, 0.0])  # sin(π/2) = 1, so -1 + 1 = 0

        np.testing.assert_array_almost_equal(result, expected)


class TestSolutionIVP:
    """Test the SolutionIVP class."""

    def test_solution_indexing(self):
        """Test SolutionIVP indexing with variables and derivatives."""
        # Create a simple solution
        y = var("y")
        t_vals = np.array([0.0, 0.1, 0.2])
        y_vals = np.array([[1.0, 0.9, 0.8], [0.0, -0.1, -0.2]])  # y values  # y' values

        mapping = {y: [0, 1], (y, 0): 0, (y, 1): 1}

        sol = SolutionIVP(t_vals, y_vals, mapping, 0.0)

        # Test variable indexing
        y_result = sol[y]
        np.testing.assert_array_equal(y_result, y_vals[0, :])

        # Test derivative indexing
        y_prime_result = sol[y.d()]
        np.testing.assert_array_equal(y_prime_result, y_vals[1, :])

    def test_solution_eval(self):
        """Test SolutionIVP interpolation."""
        y = var("y")
        t_vals = np.array([0.0, 1.0, 2.0])
        y_vals = np.array(
            [[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]  # Linear function y = t
        )  # Constant derivative y' = 1

        mapping = {y: [0, 1], (y, 0): 0, (y, 1): 1}

        sol = SolutionIVP(t_vals, y_vals, mapping, 0.0)

        # Test interpolation at intermediate point
        result = sol.eval(y, 0.5)
        expected = 0.5  # Linear interpolation

        assert abs(result - expected) < 1e-10


class TestScipyBackend:
    """Test the SciPy IVP backend."""

    def test_convert_ivp_to_state_vector(self):
        """Test conversion of IVP dict to state vector."""
        y = var("y")
        z = var("z")

        # Create mapping for two variables
        mapping = {y: [0, 1], (y, 0): 0, (y, 1): 1, z: [2], (z, 0): 2}

        # Create IVP dict
        ivp = {y: 1.0, y.d(): 2.0, z: 3.0}

        x0 = convert_ivp_to_state_vector(ivp, mapping)

        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(x0, expected)


class TestIntegratedWorkflow:
    """Test the complete integrated workflow."""

    def test_simple_harmonic_oscillator(self):
        """Test complete solution of y'' + y = 0."""
        y = var("y")
        eq = Eq(y.d(2) + y, 0)

        # Solve with initial conditions y(0) = 1, y'(0) = 0
        sol = solve(
            eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0.0, 2 * np.pi), backend="scipy"
        )

        # Check that we have a solution
        assert hasattr(sol, "t")
        assert hasattr(sol, "y")
        assert len(sol.t) > 0

        # Check that solution shape is correct
        y_vals = sol[y]
        y_prime_vals = sol[y.d()]
        assert y_vals.shape == sol.t.shape
        assert y_prime_vals.shape == sol.t.shape

        # Check initial conditions
        assert abs(y_vals[0] - 1.0) < 1e-6
        assert abs(y_prime_vals[0] - 0.0) < 1e-6

        # Check that it's approximately periodic (y(2π) ≈ y(0))
        # Note: numerical errors might accumulate, so we use a reasonable tolerance
        if len(sol.t) > 1:
            assert (
                abs(y_vals[-1] - 1.0) < 0.1
            )  # Reasonable tolerance for numerical solution

    def test_damped_oscillator(self):
        """Test damped oscillator from API tests: y'' + 0.3*y' + y = 0."""
        y = var("y")
        eq = Eq(y.d(2) + 0.3 * y.d() + y, 0)

        sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0.0, 5.0), backend="scipy")

        # Check basic properties
        assert hasattr(sol, "t")
        y_vals = sol[y]
        y_prime_vals = sol[y.d()]

        # Check initial conditions
        assert abs(y_vals[0] - 1.0) < 1e-6
        assert abs(y_prime_vals[0] - 0.0) < 1e-6

        # For a damped oscillator, amplitude should decrease
        # (this is a basic sanity check)
        if len(y_vals) > 10:
            assert abs(y_vals[-1]) < abs(
                y_vals[0]
            )  # Final amplitude < initial amplitude

    def test_as_first_order_contract(self):
        """Test the as_first_order method contract."""
        y = var("y")
        eq = Eq(y.d(2) + y, 0)
        sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0.0, 1.0), backend="scipy")

        f, jac, x0, t0, mapping = sol.as_first_order()

        # Check types and shapes
        assert callable(f)
        assert isinstance(x0, np.ndarray)
        assert isinstance(mapping, dict)
        assert isinstance(t0, float)

        # Check mapping contract
        assert mapping[y][0] == mapping[(y, 0)]
        assert mapping[y][1] == mapping[(y, 1)]

        # Check that f can be called
        result = f(0.0, x0)
        assert isinstance(result, np.ndarray)
        assert result.shape == x0.shape


class TestValidation:
    """Test validation functionality."""

    def test_missing_initial_condition_error(self):
        """Test that missing initial conditions raise proper errors."""
        y = var("y")
        eq = Eq(y.d(2) + y, 0)

        # Missing y' initial condition
        with pytest.raises(
            MissingInitialConditionError, match="Missing initial condition"
        ):
            solve(eq, ivp={y: 1.0}, tspan=(0, 1), backend="scipy")

    def test_order_mismatch_error(self):
        """Test that order mismatches raise proper errors."""
        from odecast.errors import OrderMismatchError

        y = var("y", order=1)  # Declare order 1
        eq = Eq(y.d(2) + y, 0)  # But use order 2

        with pytest.raises(OrderMismatchError, match="declared with order 1"):
            solve(eq, ivp={y: 0.0, y.d(): 0.0}, tspan=(0, 1), backend="scipy")
