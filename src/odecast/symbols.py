"""
Symbolic variables and the distinguished independent variable t
"""

import operator
import weakref
from typing import Union

import numpy as np
import sympy as sp

# Most recently created variable for each name. Used to map SymPy functions
# such as y(t) back to Variable objects when a variable only occurs inside a
# SymPy expression, e.g. sp.sin(y).
_registry: "weakref.WeakValueDictionary[str, Variable]" = weakref.WeakValueDictionary()

_BINARY_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
}

_UFUNC_OPERATORS = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
    "true_divide": operator.truediv,
    "power": operator.pow,
    "negative": operator.neg,
    "positive": operator.pos,
}

_UFUNC_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "arctan2": sp.atan2,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "arcsinh": sp.asinh,
    "arccosh": sp.acosh,
    "arctanh": sp.atanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "absolute": sp.Abs,
    "sign": sp.sign,
}


def lookup_variable(name: str) -> "Variable | None":
    """Return the most recently created variable with the given name."""
    return _registry.get(name)


class _ScalarOps:
    """
    Arithmetic shared by scalar expression types.

    Operations build an Expression tree. Conversion to SymPy goes through
    ``_sympy_``, which SymPy calls automatically, so ``sp.sin(y)`` works.
    NumPy ufuncs such as ``np.sin(y)`` are translated to their SymPy
    counterparts.
    """

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

    def __pow__(self, other):
        return Expression("**", self, other)

    def __rpow__(self, other):
        return Expression("**", other, self)

    def __neg__(self):
        return Expression("*", -1, self)

    def __pos__(self):
        return self

    def _sympy_(self):
        return self.sympy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _apply_ufunc(ufunc, method, inputs, kwargs)


class _VectorOps:
    """Element-wise arithmetic for vector and matrix expressions."""

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

    def __truediv__(self, other):
        return VectorExpression("/", self, other)

    def __rtruediv__(self, other):
        return VectorExpression("/", other, self)

    def __pow__(self, other):
        return VectorExpression("**", self, other)

    def __rpow__(self, other):
        return VectorExpression("**", other, self)

    def __neg__(self):
        return VectorExpression("*", -1, self)

    def __pos__(self):
        return self

    def _sympy_(self):
        raise TypeError(
            f"{self!r} is vector-valued and cannot be converted to a SymPy "
            "expression. Apply functions to its components, e.g. sp.sin(u[0])."
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _apply_ufunc(ufunc, method, inputs, kwargs)


def _apply_ufunc(ufunc, method, inputs, kwargs):
    """Translate a NumPy ufunc call on odecast objects."""
    if method != "__call__" or kwargs:
        return NotImplemented
    if any(isinstance(x, np.ndarray) and x.ndim > 0 for x in inputs):
        return NotImplemented
    inputs = [
        x.item() if isinstance(x, (np.generic, np.ndarray)) else x for x in inputs
    ]

    name = ufunc.__name__
    if name in _UFUNC_OPERATORS:
        return _UFUNC_OPERATORS[name](*inputs)
    if name in _UFUNC_FUNCTIONS:
        return _UFUNC_FUNCTIONS[name](*(as_sympy(x) for x in inputs))
    return NotImplemented


class Expression(_ScalarOps):
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
        if self.operator not in _BINARY_OPERATORS:
            raise ValueError(f"Unknown operator: {self.operator}")
        return _BINARY_OPERATORS[self.operator](
            as_sympy(self.left), as_sympy(self.right)
        )


class IndependentVariable(_ScalarOps):
    """
    The distinguished independent variable, typically time.

    It can be used directly in expressions and in SymPy functions,
    e.g. ``sp.sin(2*t)``.
    """

    def __init__(self, name: str = "t"):
        self.name = name
        self._symbol = sp.Symbol(name, real=True)

    @property
    def symbol(self) -> sp.Symbol:
        """Get the SymPy symbol representation."""
        return self._symbol

    def sympy(self) -> sp.Symbol:
        return self._symbol

    def __repr__(self):
        return self.name


class Variable(_ScalarOps):
    """
    A dependent variable that can appear in differential equations.

    Can be scalar (default) or vector/matrix-valued by specifying shape.
    """

    def __init__(
        self,
        name: str,
        order: int | None = None,
        shape: int | tuple[int, ...] | None = None,
    ):
        self.name = name
        self.intent_order = order  # Declared order, None if inferred
        self._derivatives = {}
        self._component_cache = {}  # Cache component variables for consistency

        # Handle shape parameter
        if shape is None:
            self.shape = None  # Scalar variable
        elif isinstance(shape, int):
            self.shape = (shape,)  # Vector variable
        else:
            self.shape = tuple(shape)  # Matrix/tensor variable

        _registry[name] = self

    @property
    def order(self) -> int | None:
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

        index = _normalize_index(index, self.shape, self.name)

        # Use cache to ensure same ComponentVariable objects
        if index not in self._component_cache:
            self._component_cache[index] = ComponentVariable(self, index)

        return self._component_cache[index]

    def components(self) -> list["ComponentVariable"]:
        """All components of a vector/matrix variable, in row-major order."""
        if self.shape is None:
            raise TypeError(f"Scalar variable '{self.name}' has no components")
        return [self[index] for index in np.ndindex(*self.shape)]

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

    def sympy(self) -> sp.Expr:
        """Convert to the SymPy function y(t)."""
        if self.shape is not None:
            return _VectorOps._sympy_(self)
        return sp.Function(self.name)(t.symbol)

    def __repr__(self):
        return self.name

    def __hash__(self):
        """Hash by identity for use as dict keys."""
        return id(self)

    def __eq__(self, other):
        """Equality by identity, so that distinct variables never compare equal."""
        return self is other

    # Vector variables combine element-wise
    def __add__(self, other):
        if self.shape is not None:
            return _VectorOps.__add__(self, other)
        return super().__add__(other)

    def __radd__(self, other):
        if self.shape is not None:
            return _VectorOps.__radd__(self, other)
        return super().__radd__(other)

    def __sub__(self, other):
        if self.shape is not None:
            return _VectorOps.__sub__(self, other)
        return super().__sub__(other)

    def __rsub__(self, other):
        if self.shape is not None:
            return _VectorOps.__rsub__(self, other)
        return super().__rsub__(other)

    def __mul__(self, other):
        if self.shape is not None:
            return _VectorOps.__mul__(self, other)
        return super().__mul__(other)

    def __rmul__(self, other):
        if self.shape is not None:
            return _VectorOps.__rmul__(self, other)
        return super().__rmul__(other)

    def __truediv__(self, other):
        if self.shape is not None:
            return _VectorOps.__truediv__(self, other)
        return super().__truediv__(other)

    def __rtruediv__(self, other):
        if self.shape is not None:
            return _VectorOps.__rtruediv__(self, other)
        return super().__rtruediv__(other)

    def __pow__(self, other):
        if self.shape is not None:
            return _VectorOps.__pow__(self, other)
        return super().__pow__(other)

    def __rpow__(self, other):
        if self.shape is not None:
            return _VectorOps.__rpow__(self, other)
        return super().__rpow__(other)

    def __neg__(self):
        if self.shape is not None:
            return _VectorOps.__neg__(self)
        return super().__neg__()


def _normalize_index(index, shape: tuple[int, ...], name: str):
    """Validate a component index; 1-D indices are plain ints."""
    if not isinstance(index, tuple):
        index = (index,)
    if len(index) != len(shape):
        raise IndexError(
            f"Variable '{name}' has shape {shape} and needs {len(shape)} "
            f"indices, got {len(index)}"
        )
    normalized = []
    for i, n in zip(index, shape, strict=True):
        if not isinstance(i, (int, np.integer)) or isinstance(i, bool):
            raise TypeError(f"Component indices must be integers, got {i!r}")
        i = int(i)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(
                f"Index {i} out of range for variable '{name}' of shape {shape}"
            )
        normalized.append(i)
    return normalized[0] if len(normalized) == 1 else tuple(normalized)


class ComponentVariable(Variable):
    """
    Represents a component of a vector/matrix variable (e.g., u[0]).
    Behaves like a scalar variable for most purposes.
    """

    def __init__(self, parent: Variable, index: int | tuple[int, ...]):
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


class VectorDerivative(_VectorOps):
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
        component = self.variable[index]  # Validates the index
        if component.index not in self._component_derivatives:
            self._component_derivatives[component.index] = component.d(self.order)
        return self._component_derivatives[component.index]

    def __repr__(self):
        if self.order == 1:
            return f"{self.variable.name}'"
        else:
            return f"{self.variable.name}^({self.order})"

    def sympy(self):
        """Vector derivatives have no scalar SymPy form."""
        return self._sympy_()


class VectorExpression(_VectorOps):
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
        """Vector expressions have no scalar SymPy form."""
        return self._sympy_()

    def component(self, index):
        """Scalar expression for the component at ``index``."""
        left = component_of(self.left, index)
        right = component_of(self.right, index)
        return Expression(self.operator, left, right)


def component_of(expr, index):
    """Component of a vector-valued operand; scalars are broadcast."""
    if isinstance(expr, Variable) and expr.shape is not None:
        return expr[index]
    if isinstance(expr, VectorDerivative):
        return expr[index]
    if isinstance(expr, VectorExpression):
        return expr.component(index)
    if isinstance(expr, Expression):
        return Expression(
            expr.operator,
            component_of(expr.left, index),
            component_of(expr.right, index),
        )
    return expr


class Derivative(_ScalarOps):
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


# The global independent variable instance
t = IndependentVariable("t")


def var(name: str, order: int | None = None) -> Variable:
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
    if isinstance(expr, sp.Basic):
        return expr
    if isinstance(expr, (_ScalarOps, _VectorOps)):
        return expr.sympy()
    return sp.sympify(expr)
