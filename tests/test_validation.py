"""
Comprehensive tests for Playbook 6 - Validation and precise errors
"""

import pytest
import numpy as np
from odecast import var, Eq, solve, t
from odecast.validate import normalize_ivp, validate_ivp
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

        with pytest.raises(
            MissingInitialConditionError,
            match="Missing initial condition for y\\^\\(0\\)",
        ):
            validate_ivp(orders, ivp, 0.0)

    def test_missing_initial_condition_level_1(self):
        """Test missing level 1 condition (first derivative)."""
        y = var("y")
        orders = {y: 2}
        ivp = {y: 1.0}  # Missing y'

        with pytest.raises(
            MissingInitialConditionError,
            match="Missing initial condition for y\\^\\(1\\)",
        ):
            validate_ivp(orders, ivp, 0.0)

    def test_missing_multiple_conditions(self):
        """Test missing multiple conditions (should report lowest level)."""
        y = var("y")
        orders = {y: 3}
        ivp = {}  # Missing all conditions

        with pytest.raises(
            MissingInitialConditionError,
            match="Missing initial condition for y\\^\\(0\\)",
        ):
            validate_ivp(orders, ivp, 0.0)

    def test_overdetermined_conditions(self):
        """Test too many initial conditions."""
        y = var("y")
        orders = {y: 2}  # Order 2: needs levels 0, 1
        ivp = {y: 1.0, y.d(): 2.0, y.d(2): 3.0}  # Provided levels 0, 1, 2

        with pytest.raises(
            OverdeterminedConditionsError,
            match="Too many initial conditions for variable y",
        ):
            validate_ivp(orders, ivp, 0.0)

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

        with pytest.raises(
            OverdeterminedConditionsError,
            match="Too many initial conditions for variable y",
        ):
            validate_ivp(orders, ivp, 0.0)


class TestErrorMessages:
    """Test that error messages are precise and helpful."""

    def test_missing_condition_error_message_details(self):
        """Test that missing condition errors have detailed messages."""
        y = var("y")
        orders = {y: 3}
        ivp = {y: 1.0, y.d(): 2.0}  # Missing y''

        with pytest.raises(MissingInitialConditionError) as excinfo:
            validate_ivp(orders, ivp, 0.0)

        error_msg = str(excinfo.value)
        assert "y^(2)" in error_msg
        assert "order 3" in error_msg
        assert "levels 0 to 2" in error_msg

    def test_overdetermined_error_message_details(self):
        """Test that overdetermined condition errors have detailed messages."""
        y = var("y")
        orders = {y: 1}  # Order 1: needs only level 0
        ivp = {y: 1.0, y.d(): 2.0}  # Provided levels 0, 1

        with pytest.raises(OverdeterminedConditionsError) as excinfo:
            validate_ivp(orders, ivp, 0.0)

        error_msg = str(excinfo.value)
        assert "variable y" in error_msg
        assert "has order 1" in error_msg
        assert "level 0 only" in error_msg or "levels 0 to 0" in error_msg


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


class TestAdvancedValidationScenarios:
    """Test advanced validation scenarios."""

    def test_multiple_variables_mixed_errors(self):
        """Test validation with multiple variables having different error types."""
        y = var("y")
        z = var("z")
        orders = {y: 2, z: 1}

        # y is correct, z is missing condition
        ivp = {y: 1.0, y.d(): 2.0}  # Missing z

        with pytest.raises(MissingInitialConditionError, match="z\\^\\(0\\)"):
            validate_ivp(orders, ivp, 0.0)

    def test_high_order_system(self):
        """Test validation for high-order system."""
        y = var("y")
        orders = {y: 5}  # Fifth-order system
        ivp = {y: 1.0, y.d(): 2.0, y.d(2): 3.0, y.d(3): 4.0, y.d(4): 5.0}

        # Should work correctly
        validate_ivp(orders, ivp, 0.0)

        # Test missing high-order derivative
        ivp_incomplete = {y: 1.0, y.d(): 2.0, y.d(2): 3.0, y.d(3): 4.0}  # Missing y''''

        with pytest.raises(MissingInitialConditionError, match="y\\^\\(4\\)"):
            validate_ivp(orders, ivp_incomplete, 0.0)

    def test_coupled_system_validation(self):
        """Test validation for coupled system from API tests."""
        y = var("y")
        z = var("z")

        eq1 = Eq(y.d(2) + z, 0)
        eq2 = Eq(z.d() - y, 0)

        # Correct conditions
        sol = solve(
            [eq1, eq2], ivp={y: 1.0, y.d(): 0.0, z: 0.0}, tspan=(0, 1), backend="scipy"
        )
        assert hasattr(sol, "t")

        # Missing condition for z
        with pytest.raises(MissingInitialConditionError):
            solve([eq1, eq2], ivp={y: 1.0, y.d(): 0.0}, tspan=(0, 1), backend="scipy")


class TestValidationEdgeCases:
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
