"""
Symbolic variables and the distinguished independent variable t
"""

from typing import Optional, Union, Any
import sympy as sp


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
    """

    def __init__(self, name: str, order: Optional[int] = None):
        self.name = name
        self.intent_order = order  # Use intent_order as specified in Playbook 2
        self._derivatives = {}

    @property
    def order(self) -> Optional[int]:
        """Backward compatibility property."""
        return self.intent_order

    def d(self, n: int = 1) -> "Derivative":
        """
        Return the nth derivative of this variable.

        Args:
            n: Order of derivative (default 1)

        Returns:
            Derivative object that can be used in equations
        """
        if n not in self._derivatives:
            self._derivatives[n] = Derivative(self, n)
        return self._derivatives[n]

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

    def d(self, n: int = 1) -> "Derivative":
        """Get higher derivatives."""
        return self.variable.d(self.order + n)

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
    elif isinstance(expr, Variable):
        return sp.Function(expr.name)(t.symbol)
    elif isinstance(expr, Derivative):
        return expr.sympy()
    elif isinstance(expr, Expression):
        return expr.sympy()
    elif hasattr(expr, "sympy"):  # Custom expression types
        return expr.sympy()
    else:
        # Try to convert using SymPy's sympify
        return sp.sympify(expr)
