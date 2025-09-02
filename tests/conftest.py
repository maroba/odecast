"""
Pytest configuration and fixtures
"""

import pytest


@pytest.fixture
def sample_variable():
    """Create a sample variable for testing."""
    from odecast import var

    return var("y")


@pytest.fixture
def sample_equation():
    """Create a sample equation for testing."""
    from odecast import var, Eq

    y = var("y")
    return Eq(y.d(2) + y, 0)
