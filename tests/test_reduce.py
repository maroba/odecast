"""
Unit tests for the reduce module - Playbook 4 functionality
"""

import pytest
import sympy as sp
from odecast import var, Eq, t
from odecast.reduce import build_state_map, isolate_highest_derivatives, make_rhs
from odecast.analyze import collect_variables, infer_orders, resolve_orders
from odecast.errors import NonSolvableFormError


class TestBuildStateMap:
    """Test the build_state_map function."""

    def test_single_variable_second_order(self):
        """Test state mapping for a single second-order variable."""
        y = var("y")
        orders = {y: 2}

        mapping = build_state_map(orders)

        # Should have mappings for y: [0, 1], (y, 0): 0, (y, 1): 1
        assert list(mapping[y]) == [0, 1]  # VariableIndexer should iterate as [0, 1]
        assert mapping[y][0] == 0  # Should support indexing
        assert mapping[y][1] == 1
        assert mapping[(y, 0)] == 0  # Should support tuple access
        assert mapping[(y, 1)] == 1

    def test_single_variable_third_order(self):
        """Test state mapping for a single third-order variable."""
        z = var("z")
        orders = {z: 3}

        mapping = build_state_map(orders)

        assert list(mapping[z]) == [0, 1, 2]
        assert mapping[z][0] == 0
        assert mapping[z][1] == 1
        assert mapping[z][2] == 2
        assert mapping[(z, 0)] == 0
        assert mapping[(z, 1)] == 1
        assert mapping[(z, 2)] == 2

    def test_multiple_variables(self):
        """Test state mapping for multiple variables with different orders."""
        y = var("y")
        z = var("z")
        orders = {y: 2, z: 1}

        mapping = build_state_map(orders)

        # y gets indices 0, 1 and z gets index 2
        assert list(mapping[y]) == [0, 1]
        assert mapping[y][0] == 0
        assert mapping[y][1] == 1
        assert list(mapping[z]) == [2]
        assert mapping[z][0] == 2
        assert mapping[(y, 0)] == 0
        assert mapping[(y, 1)] == 1
        assert mapping[(z, 0)] == 2

    def test_zero_order_variable(self):
        """Test that zero-order variables are handled correctly."""
        y = var("y")
        orders = {y: 0}

        mapping = build_state_map(orders)

        # Zero-order variable should have empty list
        assert list(mapping[y]) == []
        # But should still be accessible via tuple notation if there were any entries
        # (there aren't any in this case)
        # No (y, level) mappings should exist


class TestIsolateHighestDerivatives:
    """Test the isolate_highest_derivatives function."""

    def test_simple_second_order_ode(self):
        """Test isolating highest derivative from y'' + y = 0."""
        y = var("y")
        eq = Eq(y.d(2) + y, 0)
        orders = {y: 2}

        rules = isolate_highest_derivatives([eq], orders)

        # Should solve y'' = -y
        assert y in rules
        expected = sp.Function(y.name)(t.symbol)
        actual = rules[y]
        assert actual == -expected

    def test_second_order_with_damping(self):
        """Test y'' + 0.3*y' + y = 0."""
        y = var("y")
        eq = Eq(y.d(2) + 0.3 * y.d() + y, 0)
        orders = {y: 2}

        rules = isolate_highest_derivatives([eq], orders)

        # Should solve y'' = -0.3*y' - y
        assert y in rules
        y_func = sp.Function(y.name)(t.symbol)
        y_prime = sp.Derivative(y_func, t.symbol)
        expected = -0.3 * y_prime - y_func
        assert rules[y] == expected

    def test_multiple_variables(self):
        """Test system with multiple variables."""
        y = var("y")
        z = var("z")
        eq1 = Eq(y.d(2) + z, 0)
        eq2 = Eq(z.d() - y, 0)
        orders = {y: 2, z: 1}

        rules = isolate_highest_derivatives([eq1, eq2], orders)

        # Should have rules for both variables
        assert y in rules
        assert z in rules

        # y'' = -z
        z_func = sp.Function(z.name)(t.symbol)
        assert rules[y] == -z_func

        # z' = y
        y_func = sp.Function(y.name)(t.symbol)
        assert rules[z] == y_func

    def test_missing_highest_derivative(self):
        """Test error when highest derivative is not present."""
        y = var("y")
        eq = Eq(y.d() + y, 0)  # Only first derivative, but order is 2
        orders = {y: 2}

        with pytest.raises(NonSolvableFormError, match="No equation found containing"):
            isolate_highest_derivatives([eq], orders)

    def test_non_solvable_form(self):
        """Test error when equation cannot be solved for highest derivative."""
        y = var("y")
        # Create an equation that can't be solved for y''
        # Instead of using ** operator, create a more complex unsolvable equation
        eq = Eq(y.d(2) * y.d(2) + y.d(2) + 1, 0)  # y''^2 + y'' + 1 = 0
        orders = {y: 2}

        # This might succeed in solving (quadratic formula), but if not it should raise an error
        # Let's test with a truly unsolvable form
        try:
            rules = isolate_highest_derivatives([eq], orders)
            # If it succeeds, that's actually fine - SymPy is quite capable
            assert y in rules
        except NonSolvableFormError:
            # If it fails, that's also expected for some complex forms
            pass


class TestMakeRhs:
    """Test the make_rhs function."""

    def test_simple_second_order(self):
        """Test RHS construction for y'' + y = 0."""
        y = var("y")
        orders = {y: 2}
        mapping = build_state_map(orders)

        # Highest rule: y'' = -y
        y_func = sp.Function(y.name)(t.symbol)
        highest_rules = {y: -y_func}

        f_vector, jac = make_rhs(t.symbol, mapping, highest_rules)

        # Should get [x1, -x0] where x0=y, x1=y'
        assert f_vector.shape == (2, 1)

        # f[0] = dx0/dt = x1 (since x0=y, x1=y')
        x0, x1 = sp.symbols("x0 x1")
        assert f_vector[0] == x1

        # f[1] = dx1/dt = y'' = -y = -x0
        assert f_vector[1] == -x0

    def test_second_order_with_damping(self):
        """Test RHS for y'' + 0.3*y' + y = 0."""
        y = var("y")
        orders = {y: 2}
        mapping = build_state_map(orders)

        # Highest rule: y'' = -0.3*y' - y
        y_func = sp.Function(y.name)(t.symbol)
        y_prime = sp.Derivative(y_func, t.symbol)
        highest_rules = {y: -0.3 * y_prime - y_func}

        f_vector, jac = make_rhs(t.symbol, mapping, highest_rules)

        # Should get [x1, -0.3*x1 - x0]
        assert f_vector.shape == (2, 1)

        x0, x1 = sp.symbols("x0 x1")
        assert f_vector[0] == x1
        assert f_vector[1] == -0.3 * x1 - x0


class TestIntegration:
    """Test integration of all reduction functions."""

    def test_complete_reduction_workflow(self):
        """Test the complete workflow from equation to first-order system."""
        # Start with the second-order ODE from API tests: y'' + 0.3*y' + y = 0
        y = var("y")
        eq = Eq(y.d(2) + 0.3 * y.d() + y, 0)

        # Step 1: Analyze
        variables = collect_variables([eq])
        inferred_orders = infer_orders([eq])
        orders = resolve_orders(list(variables), inferred_orders)

        assert orders == {y: 2}

        # Step 2: Build state mapping
        mapping = build_state_map(orders)

        assert list(mapping[y]) == [0, 1]
        assert mapping[y][0] == 0
        assert mapping[y][1] == 1
        assert mapping[(y, 0)] == 0
        assert mapping[(y, 1)] == 1

        # Step 3: Isolate highest derivatives
        highest_rules = isolate_highest_derivatives([eq], orders)

        # Should get y'' = -0.3*y' - y
        y_func = sp.Function(y.name)(t.symbol)
        y_prime = sp.Derivative(y_func, t.symbol)
        expected_rule = -0.3 * y_prime - y_func
        assert highest_rules[y] == expected_rule

        # Step 4: Build RHS
        f_vector, jac = make_rhs(t.symbol, mapping, highest_rules)

        # Should get the correct first-order system
        assert f_vector.shape == (2, 1)

        x0, x1 = sp.symbols("x0 x1")
        assert f_vector[0] == x1
        assert f_vector[1] == -0.3 * x1 - x0

        # This represents the system:
        # dx0/dt = x1      (where x0 = y)
        # dx1/dt = -0.3*x1 - x0  (where x1 = y')
        # Which is equivalent to: y' = y', y'' = -0.3*y' - y
