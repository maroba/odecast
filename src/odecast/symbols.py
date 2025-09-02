"""
Symbolic variables and the distinguished independent variable t
"""

from typing import Optional, Union, Any, Tuple
import sympy as sp
import numpy as np


class Expression:
    """
    Represents a mathematical expression involving variables and derivatives.
    This is a compatibility layer that delegates to SymPy for actual computation.
    """

    def __init__(self, operator: str, left, right):
        self.operator = operator
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} {self.operator} {self.right})"

    def sympy(self) -> sp.Expr:
        """Convert this expression to a SymPy expression."""

        # Avoid circular dependency by implementing conversion directly
        def _to_sympy(expr):
            if isinstance(expr, (int, float)):
                return sp.sympify(expr)
            elif isinstance(expr, sp.Basic):
                return expr
            elif isinstance(expr, Variable):
                return sp.Function(expr.name)(t.symbol)
            elif isinstance(expr, Derivative):
                return expr.sympy()
            elif isinstance(expr, Expression):
                return expr.sympy()
            else:
                return sp.sympify(expr)

        left_sp = _to_sympy(self.left)
        right_sp = _to_sympy(self.right)

        if self.operator == "+":
            return left_sp + right_sp
        elif self.operator == "-":
            return left_sp - right_sp
        elif self.operator == "*":
            return left_sp * right_sp
        elif self.operator == "/":
            return left_sp / right_sp
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

    # Arithmetic operations for building larger expressions
    def __add__(self, other):
        return Expression("+", self, other)

    def __radd__(self, other):
        return Expression("+", other, self)

    def __sub__(self, other):
        return Expression("-", self, other)

    def __rsub__(self, other):
        return Expression("-", other, self)

    def __mul__(self, other):
        return Expression("*", self, other)

    def __rmul__(self, other):
        return Expression("*", other, self)

    def __truediv__(self, other):
        return Expression("/", self, other)

    def __rtruediv__(self, other):
        return Expression("/", other, self)


from typing import Optional, Union


class IndependentVariable:
    """
    The distinguished independent variable, typically time.
    """

    def __init__(self, name: str = "t"):
        self.name = name
        self._symbol = sp.Symbol(name, real=True)

    @property
    def symbol(self) -> sp.Symbol:
        """Get the SymPy symbol representation."""
        return self._symbol

    def __repr__(self):
        return self.name


class Variable:
    """
    A dependent variable that can appear in differential equations.

    Can be scalar (default) or vector/matrix-valued by specifying shape.
    """

    def __init__(
        self,
        name: str,
        order: Optional[int] = None,
        shape: Optional[Union[int, Tuple[int, ...]]] = None,
    ):
        self.name = name
        self.intent_order = order  # Use intent_order as specified in Playbook 2
        self._derivatives = {}
        self._component_cache = {}  # Cache component variables for consistency

        # Handle shape parameter
        if shape is None:
            self.shape = None  # Scalar variable
        elif isinstance(shape, int):
            self.shape = (shape,)  # Vector variable
        else:
            self.shape = tuple(shape)  # Matrix/tensor variable

    @property
    def order(self) -> Optional[int]:
        """Backward compatibility property."""
        return self.intent_order

    def __bool__(self):
        """Variable always evaluates to True in boolean context."""
        return True

    def __len__(self):
        """Return length for vector variables."""
        if self.shape is None:
            raise TypeError(f"Scalar variable '{self.name}' has no length")
        return self.shape[0]

    def __getitem__(self, index):
        """Component access for vector/matrix variables."""
        if self.shape is None:
            raise TypeError(f"Scalar variable '{self.name}' does not support indexing")

        # Use cache to ensure same ComponentVariable objects
        if index not in self._component_cache:
            self._component_cache[index] = ComponentVariable(self, index)

        return self._component_cache[index]

    def d(
        self, order: int = 1
    ) -> Union["Derivative", "Expression", "VectorDerivative"]:
        """
        Create a derivative of this variable.

        Args:
            order: Order of the derivative (default 1)

        Returns:
            Derivative, Expression for higher orders, or VectorDerivative for vectors
        """
        if self.shape is not None:
            # Vector/matrix variable - return VectorDerivative
            return VectorDerivative(self, order)
        else:
            # Scalar variable - return Derivative directly for any order
            return Derivative(self, order)

    def __repr__(self):
        return self.name

    def __hash__(self):
        """Hash by identity for use as dict keys."""
        return id(self)

    def __eq__(self, other):
        """Equality by identity as specified in Playbook 2."""
        return self is other

    # Arithmetic operations for building expressions
    def __add__(self, other):
        return Expression("+", self, other)

    def __radd__(self, other):
        return Expression("+", other, self)

    def __sub__(self, other):
        return Expression("-", self, other)

    def __rsub__(self, other):
        return Expression("-", other, self)

    def __mul__(self, other):
        return Expression("*", self, other)

    def __rmul__(self, other):
        return Expression("*", other, self)

    def __truediv__(self, other):
        return Expression("/", self, other)

    def __rtruediv__(self, other):
        return Expression("/", other, self)


class ComponentVariable(Variable):
    """
    Represents a component of a vector/matrix variable (e.g., u[0]).
    Behaves like a scalar variable for most purposes.
    """

    def __init__(self, parent: Variable, index: Union[int, Tuple[int, ...]]):
        self.parent = parent
        self.index = index

        # Generate component name
        if isinstance(index, int):
            component_name = f"{parent.name}[{index}]"
        else:
            index_str = ",".join(str(i) for i in index)
            component_name = f"{parent.name}[{index_str}]"

        # Initialize as scalar variable
        super().__init__(component_name, parent.intent_order, shape=None)

    def d(self, order: int = 1) -> "Derivative":
        """Override to use scalar derivative logic."""
        return Derivative(self, order)

    def __bool__(self):
        """ComponentVariable always evaluates to True."""
        return True

    def __repr__(self):
        return self.name

    def sympy(self):
        """Convert to SymPy expression."""
        return sp.Function(self.name)(t.symbol)


class VectorDerivative:
    """
    Represents the derivative of a vector variable.
    Supports component access like u.d()[0] and vectorized operations.
    """

    def __init__(self, variable: Variable, order: int):
        self.variable = variable
        self.order = order
        self._component_derivatives = {}

    def __getitem__(self, index):
        """Access derivative of a specific component."""
        if index not in self._component_derivatives:
            component = self.variable[index]  # Get ComponentVariable
            self._component_derivatives[index] = component.d(self.order)
        return self._component_derivatives[index]

    def __repr__(self):
        if self.order == 1:
            return f"{self.variable.name}'"
        else:
            return f"{self.variable.name}^({self.order})"

    def sympy(self):
        """Convert to SymPy expression."""
        # For now, vectorized operations not supported in SymPy mode
        # This should ideally expand to component derivatives
        raise NotImplementedError(
            "VectorDerivative SymPy conversion requires expanding to component derivatives"
        )

    # Arithmetic operations for vectorized operations
    def __add__(self, other):
        return VectorExpression("+", self, other)

    def __radd__(self, other):
        return VectorExpression("+", other, self)

    def __sub__(self, other):
        return VectorExpression("-", self, other)

    def __rsub__(self, other):
        return VectorExpression("-", other, self)

    def __mul__(self, other):
        return VectorExpression("*", self, other)

    def __rmul__(self, other):
        return VectorExpression("*", other, self)


class VectorExpression:
    """
    Represents arithmetic expressions involving vector variables.
    Will be expanded to component equations internally.
    """

    def __init__(self, operator: str, left, right):
        self.operator = operator
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} {self.operator} {self.right})"

    def sympy(self):
        """Convert to SymPy expression."""
        # Vector expressions need to be expanded to component equations
        # For now, raise error to indicate this needs special handling
        raise NotImplementedError(
            "VectorExpression SymPy conversion requires expanding to component equations"
        )

    def expand_to_components(self, shape):
        """Expand vector expression to component equations."""
        component_exprs = []

        for i in range(shape[0]):  # Handle 1D vectors for now
            left_comp = self._get_component(self.left, i)
            right_comp = self._get_component(self.right, i)

            if self.operator == "+":
                component_exprs.append(left_comp + right_comp)
            elif self.operator == "-":
                component_exprs.append(left_comp - right_comp)
            elif self.operator == "*":
                component_exprs.append(left_comp * right_comp)
            else:
                raise NotImplementedError(
                    f"Vector operator {self.operator} not implemented"
                )

        return component_exprs

    def _get_component(self, expr, index):
        """Get component of an expression."""
        if isinstance(expr, Variable) and expr.shape is not None:
            return expr[index]
        elif isinstance(expr, VectorDerivative):
            return expr[index]
        elif isinstance(expr, VectorExpression):
            return expr._get_component_recursive(index)
        else:
            # Scalar or constant - same for all components
            return expr

    def _get_component_recursive(self, index):
        """Recursively get component for nested vector expressions."""
        left_comp = self._get_component(self.left, index)
        right_comp = self._get_component(self.right, index)

        if self.operator == "+":
            return left_comp + right_comp
        elif self.operator == "-":
            return left_comp - right_comp
        elif self.operator == "*":
            return left_comp * right_comp
        else:
            raise NotImplementedError(
                f"Vector operator {self.operator} not implemented"
            )


class Derivative:
    """
    Represents a derivative of a variable.
    """

    def __init__(self, variable: Variable, order: int):
        self.variable = variable
        self.order = order

    def __repr__(self):
        if self.order == 1:
            return f"{self.variable.name}'"
        return f"{self.variable.name}^({self.order})"

    def sympy(self) -> sp.Expr:
        """
        Convert this derivative to a SymPy expression.

        Returns:
            SymPy Derivative expression
        """
        func = sp.Function(self.variable.name)(t.symbol)
        return sp.Derivative(func, t.symbol, self.order)

    def __hash__(self):
        """Hash by variable identity and order."""
        return hash((id(self.variable), self.order))

    def __eq__(self, other):
        """Equality by variable identity and order."""
        return (
            isinstance(other, Derivative)
            and self.variable is other.variable
            and self.order == other.order
        )

    # Arithmetic operations for building expressions
    def __add__(self, other):
        return Expression("+", self, other)

    def __radd__(self, other):
        return Expression("+", other, self)

    def __sub__(self, other):
        return Expression("-", self, other)

    def __rsub__(self, other):
        return Expression("-", other, self)

    def __mul__(self, other):
        return Expression("*", self, other)

    def __rmul__(self, other):
        return Expression("*", other, self)

    def __truediv__(self, other):
        return Expression("/", self, other)

    def __rtruediv__(self, other):
        return Expression("/", other, self)


# The global independent variable instance
t = IndependentVariable("t")


def var(name: str, order: Optional[int] = None) -> Variable:
    """
    Create a new dependent variable.

    Args:
        name: Name of the variable
        order: Maximum order of derivatives expected (for validation)

    Returns:
        Variable object that supports .d(n) for derivatives
    """
    return Variable(name, order)


def as_sympy(expr) -> sp.Expr:
    """
    Convert various expression types to SymPy expressions.

    Args:
        expr: Variable, Derivative, Expression, number, or SymPy expression

    Returns:
        SymPy expression
    """
    if isinstance(expr, (int, float)):
        return sp.sympify(expr)
    elif isinstance(expr, sp.Basic):
        return expr
    elif isinstance(expr, ComponentVariable):
        return expr.sympy()
    elif isinstance(expr, Variable):
        return sp.Function(expr.name)(t.symbol)
    elif isinstance(expr, Derivative):
        return expr.sympy()
    elif isinstance(expr, Expression):
        return expr.sympy()
    elif isinstance(expr, (VectorExpression, VectorDerivative)):
        # Vector operations need special handling
        raise NotImplementedError(
            f"SymPy conversion for {type(expr).__name__} requires component expansion"
        )
    elif hasattr(expr, "sympy"):  # Custom expression types
        return expr.sympy()
    else:
        # Try to convert using SymPy's sympify
        return sp.sympify(expr)
