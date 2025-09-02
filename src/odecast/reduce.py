"""
Reduction of higher-order ODEs to first-order systems
"""

from typing import Dict, List, Optional, Union, Any
import sympy as sp
from .symbols import Variable, Derivative, as_sympy, t
from .equation import Eq
from .errors import NonSolvableFormError


class StateMapping(dict):
    """
    Enhanced state mapping that supports both Variable and (Variable, level) access patterns.

    This class ensures that mapping[var][level] == mapping[(var, level)] for consistency
    across the API as required by Playbook 8.
    """

    def __getitem__(self, key):
        if isinstance(key, Variable):
            # Return a list-like object that supports indexing
            return VariableIndexer(self, key)
        else:
            # Use normal dict access for tuples and other keys
            return super().__getitem__(key)


class VariableIndexer:
    """
    Helper class that makes mapping[var][level] equivalent to mapping[(var, level)].
    """

    def __init__(self, mapping: StateMapping, variable: Variable):
        self.mapping = mapping
        self.variable = variable

    def __getitem__(self, level: int) -> int:
        """Get state index for variable at given derivative level."""
        return self.mapping[(self.variable, level)]

    def __iter__(self):
        """Iterate over all derivative levels for this variable."""
        # Find all levels for this variable
        levels = []
        for key in self.mapping.keys():
            if isinstance(key, tuple) and len(key) == 2 and key[0] is self.variable:
                levels.append(key[1])

        # Sort and return indices in order
        for level in sorted(levels):
            yield self.mapping[(self.variable, level)]

    def __len__(self):
        """Return number of derivative levels for this variable."""
        count = 0
        for key in self.mapping.keys():
            if isinstance(key, tuple) and len(key) == 2 and key[0] is self.variable:
                count += 1
        return count


def build_state_map(
    orders: Dict[Variable, int],
) -> StateMapping:
    """
    Build a state mapping for converting higher-order ODEs to first-order systems.

    Args:
        orders: Dictionary mapping each Variable to its maximum derivative order

    Returns:
        StateMapping with consistent access patterns:
        - mapping[var][level] returns state index for derivative level
        - mapping[(var, level)] returns same state index
    """
    mapping = StateMapping()
    state_index = 0

    for var, order in orders.items():
        # For each variable of order k, we need k state variables:
        # x[i0] = y, x[i1] = y', ..., x[i_{k-1}] = y^(k-1)
        for level in range(order):
            mapping[(var, level)] = state_index
            state_index += 1

    return mapping


def isolate_highest_derivatives(
    eqs: List[Eq], orders: Dict[Variable, int]
) -> Dict[Variable, sp.Expr]:
    """
    Isolate the highest derivatives in the equations.

    For each variable with order k>0, find an equation containing the k-th derivative
    and solve for that derivative in terms of t and lower derivatives.

    Args:
        eqs: List of equations
        orders: Dictionary mapping variables to their orders

    Returns:
        Dictionary mapping each variable to a SymPy expression for its highest derivative

    Raises:
        NonSolvableFormError: If a highest derivative cannot be isolated
    """
    highest_rules = {}

    for var, order in orders.items():
        if order == 0:
            # No derivatives for this variable, skip
            continue

        # Find an equation containing the highest derivative D(var, (t, order))
        target_deriv = var.d(order)
        target_sympy = target_deriv.sympy()

        found_equation = None
        for eq in eqs:
            sympy_eq = eq.sympy()
            if sympy_eq.has(target_sympy):
                found_equation = eq
                break

        if found_equation is None:
            raise NonSolvableFormError(
                f"No equation found containing the highest derivative {var.name}^({order}) "
                f"for variable '{var.name}'"
            )

        # Convert equation to SymPy and solve for the highest derivative
        sympy_eq = found_equation.sympy()

        try:
            # Try to solve for the highest derivative
            solutions = sp.solve(sympy_eq, target_sympy)

            if not solutions:
                raise NonSolvableFormError(
                    f"Cannot isolate highest derivative {var.name}^({order}) "
                    f"from equation: {found_equation}"
                )

            if len(solutions) > 1:
                # Multiple solutions - this might be a nonlinear equation
                # For now, take the first solution but this could be improved
                solution = solutions[0]
            else:
                solution = solutions[0]

            highest_rules[var] = solution

        except Exception as e:
            raise NonSolvableFormError(
                f"Failed to isolate highest derivative {var.name}^({order}) "
                f"from equation: {found_equation}. Error: {e}"
            )

    return highest_rules


def make_rhs(
    t_symbol: sp.Symbol,
    mapping: Dict[Union[Variable, tuple], Union[List[int], int]],
    highest_rules: Dict[Variable, sp.Expr],
    params: Optional[Dict[str, float]] = None,
) -> tuple[sp.Matrix, Optional[sp.Matrix]]:
    """
    Build the right-hand side vector field f(t, x) for the first-order system.

    Args:
        t_symbol: The SymPy symbol for the independent variable
        mapping: State mapping from build_state_map
        highest_rules: Expressions for highest derivatives from isolate_highest_derivatives
        params: Optional parameter values to substitute

    Returns:
        Tuple of (f_sympy_vector, jac_sympy_matrix)
        - f_sympy_vector: SymPy Matrix representing dx/dt = f(t, x)
        - jac_sympy_matrix: SymPy Matrix representing the Jacobian df/dx (optional for MVP)
    """
    # Determine the total number of state variables
    max_index = 0
    for key, value in mapping.items():
        if isinstance(key, tuple):  # (var, level) keys
            max_index = max(max_index, value)
        elif isinstance(key, Variable):  # var keys with lists
            if isinstance(value, list):
                max_index = max(max_index, max(value) if value else -1)

    n_states = max_index + 1

    # Create state symbols x0, x1, x2, ...
    state_symbols = [sp.Symbol(f"x{i}") for i in range(n_states)]

    # Build the RHS vector
    rhs_exprs = []

    for i in range(n_states):
        # Find which variable and derivative level this state index corresponds to
        corresponding_var = None
        corresponding_level = None

        for key, value in mapping.items():
            if isinstance(key, tuple) and len(key) == 2:  # (var, level)
                var, level = key
                if value == i:
                    corresponding_var = var
                    corresponding_level = level
                    break

        if corresponding_var is None:
            raise ValueError(f"No variable found for state index {i}")

        # Build the derivative rule for this state
        var = corresponding_var
        level = corresponding_level

        if level < len(mapping[var]) - 1:
            # For x[i] = y^(level) where level < order-1:
            # dx[i]/dt = x[i+1] = y^(level+1)
            next_index = mapping[(var, level + 1)]
            rhs_exprs.append(state_symbols[next_index])
        else:
            # For x[i] = y^(order-1), the highest derivative:
            # dx[i]/dt = y^(order) = highest_rules[var]
            if var not in highest_rules:
                raise ValueError(
                    f"No highest derivative rule found for variable {var.name}"
                )

            # Substitute all derivatives with state variables in the highest rule
            expr = highest_rules[var]

            # Replace all Variable and Derivative instances with corresponding state variables
            expr = _substitute_derivatives_with_states(expr, mapping, state_symbols)

            # Substitute parameters if provided
            if params:
                for param_name, param_value in params.items():
                    param_symbol = sp.Symbol(param_name)
                    expr = expr.subs(param_symbol, param_value)

            rhs_exprs.append(expr)

    # Create the RHS vector
    f_vector = sp.Matrix(rhs_exprs)

    # Optionally compute Jacobian (for now, return None as specified "optional for MVP")
    jac_matrix = None

    return f_vector, jac_matrix


def _substitute_derivatives_with_states(
    expr: sp.Expr,
    mapping: Dict[Union[Variable, tuple], Union[List[int], int]],
    state_symbols: List[sp.Symbol],
) -> sp.Expr:
    """
    Helper function to substitute derivatives with state variables in a SymPy expression.
    """
    # We need to substitute derivatives first, then functions
    # to avoid SymPy automatically updating derivatives when we substitute the function

    # Get all atoms and separate derivatives from functions
    all_atoms = expr.atoms(sp.Function, sp.Derivative)
    derivatives = [atom for atom in all_atoms if isinstance(atom, sp.Derivative)]
    functions = [
        atom
        for atom in all_atoms
        if isinstance(atom, sp.Function) and atom not in derivatives
    ]

    substitutions = {}

    # First, handle derivatives (higher order first)
    # Sort by derivative order (descending) to handle higher order derivatives first
    derivatives_with_order = []
    for deriv in derivatives:
        func = deriv.args[0]
        if not isinstance(func, sp.Function):
            continue

        func_name = str(func.func)

        # Extract the order of differentiation
        deriv_order = 0
        for arg in deriv.args[1:]:
            if (
                isinstance(arg, (tuple, sp.Tuple))
                and len(arg) == 2
                and arg[0] == t.symbol
            ):
                deriv_order = arg[1]
                break

        derivatives_with_order.append((deriv, func_name, deriv_order))

    # Sort by order (descending) to handle higher derivatives first
    derivatives_with_order.sort(key=lambda x: x[2], reverse=True)

    for deriv, func_name, deriv_order in derivatives_with_order:
        # Find the corresponding Variable by name
        corresponding_var = None
        for key in mapping.keys():
            if isinstance(key, tuple) and len(key) == 2:
                var, order = key
                if isinstance(var, Variable) and var.name == func_name:
                    corresponding_var = var
                    break
            elif isinstance(key, Variable) and key.name == func_name:
                corresponding_var = key
                break

        if corresponding_var and (corresponding_var, deriv_order) in mapping:
            state_index = mapping[(corresponding_var, deriv_order)]
            substitutions[deriv] = state_symbols[state_index]

    # Then handle functions (0th derivatives)
    for func in functions:
        func_name = str(func.func)

        # Make sure this is actually just the function y(t), not already handled
        if len(func.args) == 1 and func.args[0] == t.symbol:
            # Find corresponding variable
            corresponding_var = None
            for key in mapping.keys():
                if isinstance(key, tuple) and len(key) == 2:
                    var, order = key
                    if isinstance(var, Variable) and var.name == func_name:
                        corresponding_var = var
                        break
                elif isinstance(key, Variable) and key.name == func_name:
                    corresponding_var = key
                    break

            if corresponding_var and (corresponding_var, 0) in mapping:
                state_index = mapping[(corresponding_var, 0)]
                substitutions[func] = state_symbols[state_index]

    # Apply all substitutions at once
    result = expr.subs(substitutions)

    return result


def reduce_to_first_order(equation, initial_conditions):
    """
    Reduce a higher-order ODE to a system of first-order ODEs.

    This is a placeholder for future implementation that will integrate
    the above functions.
    """
    raise NotImplementedError(
        "reduce_to_first_order will be implemented in later playbooks"
    )
