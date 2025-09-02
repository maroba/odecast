"""
Unit tests for the analyze module.
"""

import pytest
from odecast import var, Eq
from odecast.analyze import collect_variables, infer_orders, resolve_orders
from odecast.errors import OrderMismatchError


def test_collect_variables_single_equation():
    """Test collecting variables from a single equation."""
    y = var("y")
    eq = Eq(y.d(2) + y, 0)

    variables = collect_variables([eq])

    assert len(variables) == 1
    assert y in variables


def test_collect_variables_multiple_equations():
    """Test collecting variables from multiple equations."""
    y = var("y")
    z = var("z")
    eq1 = Eq(y.d(2) + z, 0)
    eq2 = Eq(z.d() - y, 0)

    variables = collect_variables([eq1, eq2])

    assert len(variables) == 2
    assert y in variables
    assert z in variables


def test_infer_orders_basic():
    """Test basic order inference."""
    y = var("y")
    eq = Eq(y.d(2) + 0.3 * y.d() + y, 0)

    orders = infer_orders([eq])

    assert orders[y] == 2


def test_infer_orders_variable_only():
    """Test order inference for variable without derivatives."""
    y = var("y")
    eq = Eq(y, 5)

    orders = infer_orders([eq])

    assert orders[y] == 0


def test_infer_orders_multiple_variables():
    """Test order inference with multiple variables."""
    x = var("x")
    y = var("y")
    eq1 = Eq(x.d(3) + y.d(), 0)
    eq2 = Eq(y.d(2) + x, 0)

    orders = infer_orders([eq1, eq2])

    assert orders[x] == 3
    assert orders[y] == 2


def test_resolve_orders_no_intent():
    """Test order resolution without intent_order."""
    y = var("y")
    eq = Eq(y.d(2) + y, 0)

    variables = collect_variables([eq])
    inferred = infer_orders([eq])
    resolved = resolve_orders(list(variables), inferred)

    assert resolved[y] == 2


def test_resolve_orders_intent_higher():
    """Test order resolution with intent_order higher than inferred."""
    y = var("y", order=3)
    eq = Eq(y.d() + y, 0)

    variables = collect_variables([eq])
    inferred = infer_orders([eq])
    resolved = resolve_orders(list(variables), inferred)

    assert resolved[y] == 3  # Should use intent_order


def test_resolve_orders_intent_equal():
    """Test order resolution with intent_order equal to inferred."""
    y = var("y", order=2)
    eq = Eq(y.d(2) + y, 0)

    variables = collect_variables([eq])
    inferred = infer_orders([eq])
    resolved = resolve_orders(list(variables), inferred)

    assert resolved[y] == 2


def test_resolve_orders_mismatch():
    """Test OrderMismatchError when intent_order is too low."""
    y = var("y", order=1)
    eq = Eq(y.d(2) + y, 0)

    variables = collect_variables([eq])
    inferred = infer_orders([eq])

    with pytest.raises(OrderMismatchError) as exc_info:
        resolve_orders(list(variables), inferred)

    assert "declared with order 1" in str(exc_info.value)
    assert "derivatives up to order 2" in str(exc_info.value)
