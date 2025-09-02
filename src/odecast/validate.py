"""
Validation functions for ODEs and initial/boundary conditions
"""

from typing import Dict, Union, Tuple
from .symbols import Variable, Derivative
from .errors import MissingInitialConditionError, OverdeterminedConditionsError


def normalize_ivp(
    ivp_dict: Dict[Union[Variable, Derivative], float],
) -> Dict[Tuple[Variable, int], float]:
    """
    Normalize IVP dictionary to use (Variable, level) keys.

    Args:
        ivp_dict: Dictionary with Variable (level 0) or Derivative (level n≥1) keys

    Returns:
        Dictionary with (Variable, level) keys
    """
    normalized = {}

    for key, value in ivp_dict.items():
        if isinstance(key, Variable):
            # Variable itself is level 0
            normalized[(key, 0)] = value
        elif isinstance(key, Derivative):
            # Derivative has specified level
            normalized[(key.variable, key.order)] = value
        else:
            raise TypeError(f"IVP key must be Variable or Derivative, got {type(key)}")

    return normalized


def validate_ivp(
    orders: Dict[Variable, int],
    ivp: Dict[Union[Variable, Derivative], float],
    t0: float,
) -> None:
    """
    Validate that IVP has correct number of initial conditions.

    Args:
        orders: Dictionary mapping variables to their orders
        ivp: IVP dictionary with initial conditions
        t0: Initial time

    Raises:
        MissingInitialConditionError: When required initial conditions are missing
        OverdeterminedConditionsError: When too many conditions are provided
    """
    # Normalize the IVP dictionary
    normalized_ivp = normalize_ivp(ivp)

    # Check each variable
    for var, order in orders.items():
        # Collect all conditions for this variable
        var_conditions = {}
        for (var_key, level), value in normalized_ivp.items():
            if var_key is var:
                var_conditions[level] = value

        # For a variable of order k, we need exactly k initial conditions:
        # levels 0, 1, 2, ..., k-1
        required_levels = set(range(order))
        provided_levels = set(var_conditions.keys())

        # Check for missing conditions
        missing_levels = required_levels - provided_levels
        if missing_levels:
            missing_level = min(missing_levels)  # Report the lowest missing level
            raise MissingInitialConditionError(
                f"Missing initial condition for {var.name}^({missing_level}) "
                f"(variable has order {order}, requires conditions for levels 0 to {order-1})"
            )

        # Check for extra conditions
        extra_levels = provided_levels - required_levels
        if extra_levels:
            raise OverdeterminedConditionsError(
                f"Too many initial conditions for variable {var.name}: "
                f"provided levels {sorted(provided_levels)}, but order is {order} "
                f"(only need levels 0 to {order-1})"
            )


def validate_initial_conditions(equation, conditions):
    """
    Validate that initial conditions match the equation requirements.

    This is a placeholder that will be removed once the above functions are integrated.
    """
    raise NotImplementedError(
        "validate_initial_conditions will be implemented in later playbooks"
    )
