"""
Validation functions for ODEs and initial/boundary conditions
"""

import numpy as np

from .equation import Eq
from .errors import (
    MissingInitialConditionError,
    OrderMismatchError,
    OverdeterminedConditionsError,
)
from .symbols import Derivative, Variable, VectorDerivative


def _component_values(name: str, shape: tuple[int, ...], value) -> np.ndarray:
    """Check a vector/matrix condition against the variable's shape."""
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(
            f"Condition for vector variable {name} must be a list, tuple or "
            f"array, got {type(value).__name__}"
        )
    values = np.empty(shape, dtype=object)
    try:
        values[...] = value if isinstance(value, np.ndarray) else _nested(value)
        ok = np.shape(value) == shape
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(
            f"Initial condition for {name} has shape {np.shape(value)}, "
            f"but the variable has shape {shape}."
        )
    return values


def _nested(value):
    """Convert nested sequences to a nested list (callables are kept)."""
    if isinstance(value, (list, tuple)):
        return [_nested(v) for v in value]
    return value


def normalize_ivp(
    ivp_dict: dict[Variable | Derivative, float],
) -> dict[tuple[Variable, int], float]:
    """
    Normalize IVP dictionary to use (Variable, level) keys.

    Args:
        ivp_dict: Dictionary with Variable (level 0), Derivative (level n≥1),
                 or VectorDerivative keys. Values for vector and matrix
                 variables are nested sequences or arrays of matching shape.

    Returns:
        Dictionary with (Variable, level) keys for all components
    """
    normalized = {}

    for key, value in ivp_dict.items():
        if isinstance(key, Variable) and key.shape is None:
            normalized[(key, 0)] = value
        elif isinstance(key, Derivative):
            normalized[(key.variable, key.order)] = value
        elif isinstance(key, (Variable, VectorDerivative)):
            variable, level = (
                (key, 0) if isinstance(key, Variable) else (key.variable, key.order)
            )
            values = _component_values(repr(key), variable.shape, value)
            for index in np.ndindex(*variable.shape):
                normalized[(variable[index], level)] = values[index]
        else:
            raise TypeError(
                f"IVP key must be Variable, Derivative, or VectorDerivative, got {type(key)}"
            )

    return normalized


def normalize_conditions(
    ivp: dict | None = None, bvp: list | None = None, t0: float = 0.0
) -> list[tuple[Variable, int, float, object]]:
    """
    Convert initial or boundary conditions to (variable, level, t, value) tuples.

    Args:
        ivp: Initial conditions as passed to solve; they apply at ``t0``
        bvp: List of BC objects
        t0: Initial time for ``ivp``

    Returns:
        List of (Variable, derivative level, time, value) for scalar components
    """
    if ivp is not None:
        return [
            (v, level, t0, value) for (v, level), value in normalize_ivp(ivp).items()
        ]

    conditions = []
    for bc in bvp or []:
        if bc.t is None or bc.value is None:
            raise ValueError(f"Boundary condition {bc} needs both t and value")
        for (v, level), value in normalize_ivp({bc.variable: bc.value}).items():
            conditions.append((v, level, bc.t, value))
    return conditions


def validate_condition_levels(
    orders: dict[Variable, int], conditions: list[tuple[Variable, int, float, object]]
) -> None:
    """
    Check that conditions refer to variables of the system and only to
    derivatives below each variable's order.

    Raises:
        ValueError: For a variable that does not occur in the equations
        OverdeterminedConditionsError: For a condition on the highest derivative
    """
    for v, level, t_value, _ in conditions:
        if v not in orders:
            raise ValueError(
                f"Condition on {v.name}, which does not occur in the equations"
            )
        if orders[v] == 0:
            raise OverdeterminedConditionsError(
                f"Condition on {v.name} at t={t_value} is not allowed: {v.name} is "
                "an algebraic variable (it has no derivatives) and is determined by "
                "the algebraic equations."
            )
        if level >= orders[v]:
            raise OverdeterminedConditionsError(
                f"Condition on {v.name}^({level}) at t={t_value} is not allowed: "
                f"{v.name} has order {orders[v]}, so conditions may only involve "
                f"derivatives up to order {orders[v] - 1}."
            )


def validate_bvp(
    orders: dict[Variable, int],
    conditions: list[tuple[Variable, int, float, object]],
    tspan: tuple[float, float],
    n_unknowns: int = 0,
) -> None:
    """
    Validate boundary conditions for a two-point boundary value problem.

    Each unknown parameter requires one condition in addition to one per
    first-order state.

    Raises:
        ValueError: If a condition is not at an end point of ``tspan``
        MissingInitialConditionError: If there are too few conditions
        OverdeterminedConditionsError: If there are too many or duplicate conditions
    """
    validate_condition_levels(orders, conditions)

    a, b = tspan
    for v, level, t_value, _ in conditions:
        if not (np.isclose(t_value, a) or np.isclose(t_value, b)):
            raise ValueError(
                f"Boundary condition on {v.name}^({level}) at t={t_value} is not at "
                f"an end point of tspan={tspan}. Only two-point problems are supported."
            )

    keys = [(v, level, np.isclose(t_value, a)) for v, level, t_value, _ in conditions]
    if len(set(keys)) != len(keys):
        raise OverdeterminedConditionsError("Duplicate boundary conditions")

    n_states = sum(orders.values())
    n_required = n_states + n_unknowns
    what = f"{n_states} first-order states" + (
        f" and {n_unknowns} unknown parameter(s)" if n_unknowns else ""
    )
    message = (
        f"The system has {what} and needs {n_required} boundary conditions, "
        f"got {len(conditions)}."
    )
    if len(conditions) < n_required:
        raise MissingInitialConditionError(message)
    if len(conditions) > n_required:
        raise OverdeterminedConditionsError(message)


def validate_ivp(
    orders: dict[Variable, int],
    ivp: dict[Variable | Derivative, float],
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
        # Algebraic variables need no condition; a value is a starting guess
        if order == 0:
            continue

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
                level_desc = f"levels 0 to {order - 1}"

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
                isinstance(key, VectorDerivative) for key in ivp
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
                    level_desc = f"levels 0 to {order - 1} only"

                raise OverdeterminedConditionsError(
                    f"Too many initial conditions for variable {var.name}. "
                    f"Variable has order {order} and needs {level_desc}, "
                    f"but got conditions for levels {sorted(provided_levels)}."
                )


def validate_variable_orders(
    equations: list[Eq],
    declared_orders: dict[Variable, int],
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
    collect_variables(equations)

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
