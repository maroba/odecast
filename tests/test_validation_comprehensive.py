"""
Test validation functions thoroughly for Playbook 6
"""

import pytest
from odecast import var, Eq, solve, t
from odecast.validate import normalize_ivp, validate_ivp, validate_variable_orders
from odecast.errors import (
    MissingInitialConditionError,
    OverdeterminedConditionsError,
    OrderMismatchError,
)


class TestNormalizeIVP:
    """Test the normalize_ivp function."""

    def test_variable_only(self):
        """Test normalization with Variable keys only."""
        y = var("y")
        z = var("z")

        ivp = {y: 1.0, z: 2.0}
        normalized = normalize_ivp(ivp)

        expected = {(y, 0): 1.0, (z, 0): 2.0}
        assert normalized == expected

    def test_derivative_only(self):
        """Test normalization with Derivative keys only."""
        y = var("y")

        ivp = {y.d(): 1.0, y.d(2): 2.0}
        normalized = normalize_ivp(ivp)

        expected = {(y, 1): 1.0, (y, 2): 2.0}
        assert normalized == expected

    def test_mixed_variable_derivative(self):
        """Test normalization with mixed Variable and Derivative keys."""
        y = var("y")
        z = var("z")

        ivp = {y: 1.0, y.d(): 2.0, z: 3.0, z.d(2): 4.0}
        normalized = normalize_ivp(ivp)

        expected = {(y, 0): 1.0, (y, 1): 2.0, (z, 0): 3.0, (z, 2): 4.0}
        assert normalized == expected

    def test_invalid_key_type(self):
        """Test that invalid key types raise TypeError."""
        ivp = {"invalid": 1.0}

        with pytest.raises(
            TypeError, match="IVP key must be Variable, Derivative, or VectorDerivative"
        ):
            normalize_ivp(ivp)


class TestValidateIVP:
    """Test the validate_ivp function."""

    def test_correct_conditions_single_variable(self):
        """Test validation with correct conditions for single variable."""
        y = var("y")
        orders = {y: 2}
        ivp = {y: 1.0, y.d(): 2.0}

        # Should not raise any exception
        validate_ivp(orders, ivp, 0.0)

    def test_correct_conditions_multiple_variables(self):
        """Test validation with correct conditions for multiple variables."""
        y = var("y")
        z = var("z")
        orders = {y: 2, z: 1}
        ivp = {y: 1.0, y.d(): 2.0, z: 3.0}

        # Should not raise any exception
        validate_ivp(orders, ivp, 0.0)

    def test_missing_initial_condition_level_0(self):
        """Test missing level 0 condition (the variable itself)."""
        y = var("y")
        orders = {y: 2}
        ivp = {y.d(): 2.0}  # Missing y itself

        with pytest.raises(MissingInitialConditionError) as excinfo:
            validate_ivp(orders, ivp, 0.0)

        error_msg = str(excinfo.value)
        assert "y^(0)" in error_msg
        assert "order 2" in error_msg

    def test_missing_initial_condition_level_1(self):
        """Test missing level 1 condition (first derivative)."""
        y = var("y")
        orders = {y: 2}
        ivp = {y: 1.0}  # Missing y'

        with pytest.raises(MissingInitialConditionError) as excinfo:
            validate_ivp(orders, ivp, 0.0)

        error_msg = str(excinfo.value)
        assert "y^(1)" in error_msg
        assert "order 2" in error_msg

    def test_overdetermined_conditions(self):
        """Test too many initial conditions."""
        y = var("y")
        orders = {y: 2}  # Order 2: needs levels 0, 1
        ivp = {y: 1.0, y.d(): 2.0, y.d(2): 3.0}  # Provided levels 0, 1, 2

        with pytest.raises(OverdeterminedConditionsError) as excinfo:
            validate_ivp(orders, ivp, 0.0)

        error_msg = str(excinfo.value)
        assert "variable y" in error_msg
        assert "order 2" in error_msg

    def test_zero_order_variable(self):
        """Test validation for zero-order variable (algebraic variable)."""
        y = var("y")
        orders = {y: 0}
        ivp = {}  # No conditions needed for zero-order variable

        # Should not raise any exception
        validate_ivp(orders, ivp, 0.0)

    def test_zero_order_with_extra_condition(self):
        """Test zero-order variable with unnecessary condition."""
        y = var("y")
        orders = {y: 0}
        ivp = {y: 1.0}  # Extra condition for zero-order variable

        with pytest.raises(OverdeterminedConditionsError):
            validate_ivp(orders, ivp, 0.0)

    def test_first_order_system(self):
        """Test first-order system validation."""
        y = var("y")
        z = var("z")
        orders = {y: 1, z: 1}
        ivp = {y: 1.0, z: 2.0}  # Only need level 0 for order 1

        # Should not raise any exception
        validate_ivp(orders, ivp, 0.0)

    def test_high_order_system(self):
        """Test high-order system validation."""
        y = var("y")
        orders = {y: 4}  # Fourth-order system
        ivp = {y: 1.0, y.d(): 2.0, y.d(2): 3.0, y.d(3): 4.0}

        # Should work correctly
        validate_ivp(orders, ivp, 0.0)


class TestValidateVariableOrders:
    """Test the validate_variable_orders function."""

    def test_correct_declared_orders(self):
        """Test validation with correct declared orders."""
        y = var("y", order=2)
        z = var("z", order=1)

        eq1 = Eq(y.d(2) + y, 0)
        eq2 = Eq(z.d() - y, 0)

        declared_orders = {y: 2, z: 1}

        # Should not raise any exception
        validate_variable_orders([eq1, eq2], declared_orders)

    def test_order_mismatch_too_low(self):
        """Test declared order is too low for equation usage."""
        y = var("y", order=1)  # Declared order 1

        eq = Eq(y.d(2) + y, 0)  # But uses order 2

        declared_orders = {y: 1}

        with pytest.raises(OrderMismatchError) as excinfo:
            validate_variable_orders([eq], declared_orders)

        error_msg = str(excinfo.value)
        assert "declared with order 1" in error_msg
        assert "equations require order 2" in error_msg

    def test_no_declared_orders(self):
        """Test validation with no declared orders (all inferred)."""
        y = var("y")  # No explicit order

        eq = Eq(y.d(2) + y, 0)

        declared_orders = {}  # No explicit declarations

        # Should not raise any exception
        validate_variable_orders([eq], declared_orders)


class TestIntegrationWithSolve:
    """Test validation integration with the solve function."""

    def test_solve_with_missing_condition(self):
        """Test that solve raises validation errors appropriately."""
        y = var("y")
        eq = Eq(y.d(2) + y, 0)

        with pytest.raises(MissingInitialConditionError):
            solve(eq, ivp={y: 1.0}, tspan=(0, 1), backend="scipy")

    def test_solve_with_order_mismatch(self):
        """Test that solve raises order mismatch errors."""
        y = var("y", order=1)  # Declared order 1
        eq = Eq(y.d(2) + y, 0)  # But uses order 2

        with pytest.raises(OrderMismatchError):
            solve(eq, ivp={y: 0.0, y.d(): 0.0}, tspan=(0, 1), backend="scipy")

    def test_solve_with_correct_conditions(self):
        """Test that solve works when conditions are correct."""
        y = var("y")
        eq = Eq(y.d(2) + y, 0)

        # This should work without errors
        sol = solve(eq, ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 1), backend="scipy")
        assert hasattr(sol, "t")
        assert hasattr(sol, "y")


class TestEdgeCases:
    """Test edge cases in validation."""

    def test_empty_ivp_dict(self):
        """Test validation with empty IVP dictionary."""
        y = var("y")
        orders = {y: 1}
        ivp = {}

        with pytest.raises(MissingInitialConditionError):
            validate_ivp(orders, ivp, 0.0)

    def test_empty_orders_dict(self):
        """Test validation with empty orders dictionary."""
        y = var("y")
        orders = {}
        ivp = {y: 1.0}

        # Should not raise error - no variables to validate
        validate_ivp(orders, ivp, 0.0)

    def test_variable_identity_preservation(self):
        """Test that validation respects variable identity."""
        y1 = var("y")
        y2 = var("y")  # Same name, different object

        orders = {y1: 1}
        ivp = {y2: 1.0}  # Different variable object with same name

        # Should fail because y2 is not the same object as y1
        with pytest.raises(MissingInitialConditionError):
            validate_ivp(orders, ivp, 0.0)
