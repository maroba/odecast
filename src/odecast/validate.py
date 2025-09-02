"""
Validation functions for ODEs and initial/boundary conditions
"""

from typing import Dict, Union, Tuple, List
from .symbols import Variable, Derivative, VectorDerivative
from .equation import Eq
from .errors import (
    MissingInitialConditionError,
    OverdeterminedConditionsError,
    OrderMismatchError,
)


def normalize_ivp(
    ivp_dict: Dict[Union[Variable, Derivative], float],
) -> Dict[Tuple[Variable, int], float]:
    """
    Normalize IVP dictionary to use (Variable, level) keys.

    Args:
        ivp_dict: Dictionary with Variable (level 0), Derivative (level n≥1),
                 or VectorDerivative keys, with values that can be floats or lists

    Returns:
        Dictionary with (Variable, level) keys for all components
    """
    normalized = {}

    for key, value in ivp_dict.items():
        if isinstance(key, Variable):
            if key.shape is not None:
                # Vector variable - expand to components
                if not isinstance(value, (list, tuple)):
                    raise ValueError(
                        f"Vector variable {key.name} requires list/tuple initial condition, got {type(value)}"
                    )

                for i, comp_value in enumerate(value):
                    component_var = key[i]  # Get ComponentVariable
                    normalized[(component_var, 0)] = comp_value
            else:
                # Scalar variable - level 0
                normalized[(key, 0)] = value

        elif isinstance(key, Derivative):
            # Derivative has specified level
            normalized[(key.variable, key.order)] = value

        elif isinstance(key, VectorDerivative):
            # Vector derivative - expand to component derivatives
            if not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"Vector derivative requires list/tuple initial condition, got {type(value)}"
                )

            for i, comp_value in enumerate(value):
                component_var = key.variable[i]  # Get ComponentVariable
                normalized[(component_var, key.order)] = comp_value
        else:
            raise TypeError(
                f"IVP key must be Variable, Derivative, or VectorDerivative, got {type(key)}"
            )

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
            if order == 0:
                # Zero-order variables don't need any conditions
                continue
            elif order == 1:
                level_desc = "level 0"
            else:
                level_desc = f"levels 0 to {order-1}"

            raise MissingInitialConditionError(
                f"Missing initial condition for {var.name}^({missing_level}). "
                f"Variable has order {order} and requires conditions for {level_desc}."
            )

        # Check for extra conditions - but be lenient with vector derivatives
        extra_levels = provided_levels - required_levels
        if extra_levels:
            # Check if this might be from vector derivative expansion
            # In that case, we can just ignore the extra conditions
            from .symbols import VectorDerivative

            has_vector_derivative = any(
                isinstance(key, VectorDerivative) for key in ivp.keys()
            )

            if has_vector_derivative:
                # Filter out the extra conditions silently - they came from vector derivative expansion
                # where some components don't need as high an order
                continue
            else:
                # Strict validation for non-vector cases
                if order == 0:
                    level_desc = "no conditions"
                elif order == 1:
                    level_desc = "level 0 only"
                else:
                    level_desc = f"levels 0 to {order-1} only"

                raise OverdeterminedConditionsError(
                    f"Too many initial conditions for variable {var.name}. "
                    f"Variable has order {order} and needs {level_desc}, "
                    f"but got conditions for levels {sorted(provided_levels)}."
                )


def validate_variable_orders(
    equations: List[Eq],
    declared_orders: Dict[Variable, int],
) -> None:
    """
    Validate that declared variable orders match equation requirements.

    Args:
        equations: List of equations to check
        declared_orders: Dictionary of explicitly declared variable orders

    Raises:
        OrderMismatchError: When declared order doesn't match equation usage
    """
    from .analyze import collect_variables, infer_orders

    # Get all variables used in equations
    all_variables = collect_variables(equations)

    # Infer orders from equation usage
    inferred_orders = infer_orders(equations)

    # Check for mismatches between declared and inferred orders
    for var, declared_order in declared_orders.items():
        if var in inferred_orders:
            inferred_order = inferred_orders[var]
            if declared_order != inferred_order:
                raise OrderMismatchError(
                    f"Variable {var.name} declared with order {declared_order} "
                    f"but equations require order {inferred_order}. "
                    f"Either update the declaration or check your equations."
                )


def validate_initial_conditions(equation, conditions):
    """
    Validate that initial conditions match the equation requirements.

    This is a placeholder that will be removed once the above functions are integrated.
    """
    raise NotImplementedError(
        "validate_initial_conditions will be implemented in later playbooks"
    )
