"""
Partial differential equations, solved with the method of lines

Spatial derivatives are discretized with finite differences from findiff;
the resulting system of ordinary differential equations is integrated with
scipy.integrate.solve_ivp. Requires ``findiff`` (``pip install
"odecast[pde]"``).
"""

import warnings

import numpy as np
import sympy as sp
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import splu

from .symbols import Expression, _ScalarOps, as_sympy, substitute_params, t


class Grid:
    """
    Rectangular grid of one to three dimensions.

    Args:
        **axes: Coordinates per axis, e.g. ``Grid(x=np.linspace(0, 1, 101))``
            or ``Grid(x=..., y=...)``. Non-uniform spacing is allowed except
            on periodic axes, whose grid must not contain the end point.

    The coordinates are available as SymPy symbols for use in equations,
    e.g. ``grid.x``.
    """

    def __init__(self, **axes):
        if not 1 <= len(axes) <= 3:
            raise ValueError("A grid has one to three axes")
        if "t" in axes:
            raise ValueError("The name t is reserved for time")
        self.names = list(axes)
        self.coords = [np.asarray(c, dtype=float) for c in axes.values()]
        self.shape = tuple(len(c) for c in self.coords)
        self.symbols = [sp.Symbol(name, real=True) for name in self.names]
        self.mesh = np.meshgrid(*self.coords, indexing="ij")

    def __getattr__(self, name):
        if name in self.__dict__.get("names", []):
            return self.symbols[self.names.index(name)]
        raise AttributeError(name)

    def __repr__(self):
        sizes = ", ".join(
            f"{n}: {len(c)}" for n, c in zip(self.names, self.coords, strict=True)
        )
        return f"Grid({sizes})"

    def axis(self, name) -> int:
        name = getattr(name, "name", name)
        if name not in self.names:
            raise ValueError(f"The grid has no axis {name}")
        return self.names.index(name)


class Field(_ScalarOps):
    """
    An unknown function of time and space on a grid, created with
    :func:`field`.
    """

    def __init__(self, name: str, grid: Grid):
        self.name = name
        self.grid = grid

    def __repr__(self):
        return self.name

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def sympy(self):
        return sp.Symbol(self.name)

    def d(self, order: int = 1) -> "FieldTerm":
        """Time derivative of the given order (1 or 2)."""
        if order not in (1, 2):
            raise ValueError("Only first and second time derivatives are supported")
        return FieldTerm(self, time=order)

    def diff(self, **orders) -> "FieldTerm":
        """Spatial derivative, e.g. ``u.diff(x=1, y=1)`` for u_xy."""
        spatial = [0] * len(self.grid.names)
        for name, order in orders.items():
            spatial[self.grid.axis(name)] = order
        return FieldTerm(self, spatial=tuple(spatial))

    def dx(self, order: int = 1):
        return self.diff(**{self.grid.names[0]: order})

    def dy(self, order: int = 1):
        return self.diff(**{self.grid.names[1]: order})

    def dz(self, order: int = 1):
        return self.diff(**{self.grid.names[2]: order})

    def laplace(self):
        """Sum of the second derivatives along all axes."""
        terms = [self.diff(**{n: 2}) for n in self.grid.names]
        result = terms[0]
        for term in terms[1:]:
            result = Expression("+", result, term)
        return result


class FieldTerm(_ScalarOps):
    """A time or spatial derivative of a field."""

    def __init__(self, field: Field, time: int = 0, spatial: tuple | None = None):
        self.field = field
        self.time = time
        self.spatial = spatial or (0,) * len(field.grid.names)

    def __repr__(self):
        if self.time:
            return f"{self.field.name}_" + "t" * self.time
        parts = "".join(
            n * k for n, k in zip(self.field.grid.names, self.spatial, strict=True)
        )
        return f"{self.field.name}_{parts}"

    def sympy(self):
        if self.time:
            return sp.Symbol(f"{self.field.name}__t{self.time}")
        return sp.Symbol(f"{self.field.name}__d" + "_".join(map(str, self.spatial)))


def field(name: str, grid: Grid) -> Field:
    """
    Create an unknown field u(t, x, ...) on a grid.

    Examples:
        >>> grid = Grid(x=np.linspace(0, 1, 101))
        >>> u = field("u", grid)
        >>> Eq(u.d(), 0.1 * u.dx(2))   # heat equation
    """
    return Field(name, grid)


class Dirichlet:
    """Prescribed value: a number, an array, or a function of t."""

    def __init__(self, value=0.0):
        self.value = value


class Neumann:
    """Prescribed derivative along the axis (not the outward normal)."""

    def __init__(self, value=0.0):
        self.value = value


class Periodic:
    """Periodic boundary; the grid must not contain the end point."""


class PDESolution:
    """
    Solution of :func:`solve_pde`.

    Attributes:
        t: Time points returned by the solver
        grid: The grid
    """

    def __init__(self, t, grid, fields, interpolant, unpack):
        self.t = t
        self.grid = grid
        self._fields = fields
        self._interpolant = interpolant
        self._unpack = unpack

    def __getitem__(self, f: Field) -> np.ndarray:
        """Values on the grid at the solver's time points, shape ``(*grid.shape, len(t))``."""
        return self.eval(f, self.t)

    def eval(self, f: Field, times) -> np.ndarray:
        """
        Values on the grid at arbitrary times.

        Returns:
            Array of shape ``grid.shape`` for a single time, or
            ``(*grid.shape, len(times))``
        """
        if f not in self._fields:
            raise KeyError(f"{f!r} is not part of the solution")
        scalar = np.isscalar(times)
        times = np.atleast_1d(np.asarray(times, dtype=float))
        states = self._interpolant(times)
        values = np.stack(
            [self._unpack(ti, states[:, k])[f] for k, ti in enumerate(times)], axis=-1
        )
        return values[..., 0] if scalar else values

    def plot(self, f: Field, times=None, ax=None, **kwargs):
        """
        Plot a field: profiles at several times on a 1D grid, or an image at
        one time on a 2D grid.

        Args:
            f: The field
            times: Times to show; five equally spaced times (1D) or the final
                time (2D) if omitted
            ax: Matplotlib axes; the current axes if omitted

        Returns:
            The Matplotlib axes
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        if len(self.grid.shape) == 1:
            times = np.linspace(self.t[0], self.t[-1], 5) if times is None else times
            for ti in np.atleast_1d(times):
                ax.plot(
                    self.grid.coords[0], self.eval(f, ti), label=f"t = {ti:g}", **kwargs
                )
            ax.set_xlabel(self.grid.names[0])
            ax.set_ylabel(f.name)
            ax.legend(fontsize="small")
        elif len(self.grid.shape) == 2:
            ti = self.t[-1] if times is None else float(np.atleast_1d(times)[0])
            x, y = self.grid.coords
            image = ax.pcolormesh(x, y, self.eval(f, ti).T, shading="auto", **kwargs)
            plt.colorbar(image, ax=ax, label=f.name)
            ax.set_xlabel(self.grid.names[0])
            ax.set_ylabel(self.grid.names[1])
            ax.set_title(f"t = {ti:g}")
        else:
            raise ValueError("plot() supports 1D and 2D grids")
        return ax


def solve_pde(
    equation,
    ic: dict,
    bc: dict,
    tspan: tuple[float, float],
    *,
    params: dict | None = None,
    acc: int = 2,
    **options,
) -> PDESolution:
    """
    Solve time-dependent partial differential equations with the method of lines.

    Args:
        equation: Eq or list of Eq objects in fields, their time derivatives
            ``u.d()``/``u.d(2)`` and spatial derivatives ``u.dx(n)``,
            ``u.diff(x=1, y=1)``, ``u.laplace()``. Coordinates (``grid.x``),
            ``t`` and parameters may appear as well.
        ic: Initial values per field: arrays of the grid shape, numbers, or
            functions of the coordinates. For equations of second order in
            time, also the initial time derivative: ``{u: ..., u.d(): ...}``.
        bc: Boundary conditions per field and axis, e.g.
            ``{u: {"x": (Dirichlet(0), Neumann(0))}}`` (lower and upper end)
            or ``{u: {"x": Periodic()}}``. Values may be functions of t.
        tspan: Time interval
        params: Parameter values
        acc: Accuracy order of the finite differences. Higher orders use
            one-sided stencils near non-periodic boundaries, which can make
            undamped wave equations unstable; acc=2 is safe there.
        **options: Passed to scipy.integrate.solve_ivp. The default method
            is "BDF", with the sparsity pattern of the Jacobian.

    Returns:
        PDESolution

    Examples:
        >>> grid = Grid(x=np.linspace(0, 1, 101))
        >>> u = field("u", grid)
        >>> sol = solve_pde(Eq(u.d(), 0.1 * u.dx(2)),
        ...                 ic={u: lambda x: np.sin(np.pi * x)},
        ...                 bc={u: {"x": (Dirichlet(0), Dirichlet(0))}},
        ...                 tspan=(0, 1))
    """
    try:
        from findiff import Diff
    except ImportError as e:
        raise ImportError(
            'solve_pde() requires findiff: pip install "odecast[pde]"'
        ) from e

    from .equation import Eq

    eqs = [equation] if isinstance(equation, Eq) else list(equation)
    residuals = [as_sympy(eq.lhs) - as_sympy(eq.rhs) for eq in eqs]

    fields = [k for k in ic if isinstance(k, Field)]
    if not fields:
        raise ValueError("ic must contain the initial values of the fields")
    grid = fields[0].grid
    if any(f.grid is not grid for f in fields):
        raise ValueError("All fields must be defined on the same grid")
    if len(fields) != len(eqs):
        raise ValueError(f"{len(eqs)} equation(s) for {len(fields)} field(s)")

    free = set().union(*(r.free_symbols for r in residuals))
    order = {
        f: max((k for k in (1, 2) if sp.Symbol(f"{f.name}__t{k}") in free), default=0)
        for f in fields
    }
    for f in fields:
        if order[f] == 0:
            raise ValueError(f"No time derivative of {f.name} in the equations")
    highest = [sp.Symbol(f"{f.name}__t{order[f]}") for f in fields]
    solutions = sp.solve(residuals, highest, dict=True)
    if not solutions:
        raise ValueError("Cannot solve the equations for the time derivatives")
    rates = [substitute_params(solutions[0][s], params) for s in highest]

    # Symbols of the right-hand sides
    spatial = sorted(
        {s for r in rates for s in r.free_symbols if "__d" in s.name},
        key=lambda s: s.name,
    )
    velocity = [sp.Symbol(f"{f.name}__t1") for f in fields if order[f] == 2]
    field_syms = [sp.Symbol(f.name) for f in fields]
    unknown = (
        set().union(*(r.free_symbols for r in rates))
        - set(spatial)
        - set(velocity)
        - set(field_syms)
        - set(grid.symbols)
        - {t.symbol}
    )
    if unknown:
        names = ", ".join(sorted(s.name for s in unknown))
        raise ValueError(f"Missing values for parameters: {names}")
    args = [t.symbol, *grid.symbols, *field_syms, *velocity, *spatial]
    rate_num = sp.lambdify(args, rates, modules="numpy", cse=True)

    by_name = {f.name: f for f in fields}
    periodic = _periodic_axes(fields, bc, grid)
    if (
        acc > 2
        and any(order[f] == 2 for f in fields)
        and len(periodic) < len(grid.names)
    ):
        warnings.warn(
            "The one-sided boundary stencils used for acc > 2 can make undamped "
            "equations of second order in time (e.g. the wave equation) unstable. "
            "Use acc=2 unless all boundaries are periodic.",
            stacklevel=2,
        )

    def operator(orders):
        op = None
        for axis, k in enumerate(orders):
            if k == 0:
                continue
            coords = grid.coords[axis]
            spacing = coords[1] - coords[0]
            uniform = np.allclose(np.diff(coords), spacing)
            d = (
                Diff(
                    axis,
                    spacing if uniform or axis in periodic else coords,
                    acc=acc,
                    periodic=axis in periodic,
                )
                ** k
            )
            op = d if op is None else op * d
        return op.matrix(grid.shape).tocsr()

    derivative_terms = []
    for s in spatial:
        name, rest = s.name.split("__d")
        derivative_terms.append(
            (by_name[name], operator(tuple(int(k) for k in rest.split("_"))))
        )

    boundaries = {f: _Boundary(f, bc.get(f), grid, periodic, operator) for f in fields}

    # State: the free nodes of every field, then the velocities of fields of
    # second order in time
    blocks = [(f, 0) for f in fields] + [(f, 1) for f in fields if order[f] == 2]
    sizes = [int(boundaries[f].free.sum()) for f, _ in blocks]
    offsets = np.concatenate([[0], np.cumsum(sizes)])

    def unpack(time, y):
        full = {}
        for (f, level), a, b in zip(blocks, offsets[:-1], offsets[1:], strict=True):
            if level == 0:
                full[f] = boundaries[f].fill(time, y[a:b])
        return full

    def rhs(time, y):
        full = unpack(time, y)
        values = [time, *grid.mesh, *(full[f] for f in fields)]
        for (f, level), a, b in zip(blocks, offsets[:-1], offsets[1:], strict=True):
            if level == 1:
                v = np.zeros(grid.shape)
                v[boundaries[f].free] = y[a:b]
                values.append(v)
        values += [
            (m @ full[f].ravel()).reshape(grid.shape) for f, m in derivative_terms
        ]
        rate = rate_num(*values)
        out = []
        for (f, level), a, b in zip(blocks, offsets[:-1], offsets[1:], strict=True):
            free = boundaries[f].free
            if order[f] == 2 and level == 0:
                k = blocks.index((f, 1))
                out.append(y[offsets[k] : offsets[k + 1]])
            else:
                r = np.broadcast_to(rate[fields.index(f)], grid.shape)
                out.append(r[free])
        return np.concatenate(out)

    y0 = []
    for f, level in blocks:
        key = f if level == 0 else f.d()
        value = _lookup(ic, key)
        if value is None:
            if level == 1:
                raise ValueError(
                    f"{f.name} is of second order in time; give ic[{f.name}.d()]"
                )
            raise ValueError(f"Missing initial values for {f.name}")
        values = value(*grid.mesh) if callable(value) else value
        values = np.broadcast_to(np.asarray(values, dtype=float), grid.shape)
        y0.append(values[boundaries[f].free])
    y0 = np.concatenate(y0)

    options = dict(options)
    options.setdefault("method", "BDF")
    options.setdefault("rtol", 1e-3)
    options.setdefault("atol", 1e-6)
    options["dense_output"] = True
    if options["method"] in ("BDF", "Radau") and "jac_sparsity" not in options:
        options["jac_sparsity"] = _sparsity(blocks, boundaries, derivative_terms, grid)

    sol = solve_ivp(rhs, tspan, y0, **options)
    if not sol.success:
        raise RuntimeError(f"SciPy solve_ivp failed: {sol.message}")
    return PDESolution(sol.t, grid, fields, sol.sol, unpack)


def _lookup(ic, key):
    for k, v in ic.items():
        if k is key or (
            isinstance(k, FieldTerm)
            and isinstance(key, FieldTerm)
            and k.field is key.field
            and k.time == key.time
        ):
            return v
    return None


def _periodic_axes(fields, bc, grid) -> set:
    periodic = set()
    for f in fields:
        for name, spec in (bc.get(f) or {}).items():
            if isinstance(spec, Periodic):
                periodic.add(grid.axis(name))
    inconsistent = [
        (f, grid.names[axis])
        for f in fields
        for axis in periodic
        if not isinstance((bc.get(f) or {}).get(grid.names[axis]), Periodic)
    ]
    if inconsistent:
        f, name = inconsistent[0]
        raise ValueError(
            f"Axis {name} is periodic for one field, so it must be periodic for "
            f"{f.name} as well"
        )
    return periodic


class _Boundary:
    """
    Boundary nodes of one field and how to fill them.

    Dirichlet nodes take the prescribed value. Neumann nodes are determined
    by requiring that the one-sided finite-difference derivative at the
    boundary equals the prescribed value; this is a small sparse linear
    system in the Neumann nodes. All other nodes are free (integrated).
    """

    def __init__(self, f, spec, grid, periodic, operator):
        spec = spec or {}
        self.grid = grid
        ndim = len(grid.shape)
        self.dirichlet = []  # (selection, value)
        neumann = []  # (axis, selection, value)
        for axis, name in enumerate(grid.names):
            if axis in periodic:
                continue
            if name not in spec:
                raise ValueError(
                    f"Missing boundary conditions for {f.name} along {name}"
                )
            lower, upper = spec[name]
            for end, condition in ((0, lower), (-1, upper)):
                selection = tuple(
                    end if a == axis else slice(None) for a in range(ndim)
                )
                if isinstance(condition, Dirichlet):
                    self.dirichlet.append((selection, condition.value))
                elif isinstance(condition, Neumann):
                    neumann.append((axis, selection, condition.value))
                else:
                    raise TypeError(
                        "Boundary conditions must be Dirichlet, Neumann or Periodic, "
                        f"got {condition!r}"
                    )

        dirichlet_mask = np.zeros(grid.shape, dtype=bool)
        for selection, _ in self.dirichlet:
            dirichlet_mask[selection] = True

        # Neumann nodes on corners shared with a Dirichlet boundary are Dirichlet
        flat = np.arange(int(np.prod(grid.shape))).reshape(grid.shape)
        self._neumann = (
            []
        )  # (nodes, value, keep) with keep selecting within the boundary
        rows = []
        for axis, selection, value in neumann:
            keep = ~dirichlet_mask[selection]
            nodes = flat[selection][keep]
            if nodes.size:
                first = operator(tuple(int(a == axis) for a in range(ndim)))
                rows.append(first[nodes])
                self._neumann.append((nodes, value, keep))

        self.free = ~dirichlet_mask
        if rows:
            nodes = np.concatenate([n for n, _, _ in self._neumann])
            self.free.flat[nodes] = False
            M = sparse.vstack(rows).tocsc()
            self._nodes = nodes
            self._known = np.setdiff1d(np.arange(M.shape[1]), nodes)
            self._M_known = M[:, self._known]
            self._lu = splu(M[:, nodes].tocsc())

    def fill(self, time, free_values):
        u = np.zeros(self.grid.shape)
        u[self.free] = free_values
        for selection, value in self.dirichlet:
            u[selection] = value(time) if callable(value) else value
        if self._neumann:
            h = np.concatenate(
                [
                    np.broadcast_to(
                        value(time) if callable(value) else value, keep.shape
                    )[keep]
                    for _, value, keep in self._neumann
                ]
            )
            flat = u.ravel()
            flat[self._nodes] = self._lu.solve(h - self._M_known @ flat[self._known])
        return u


def _sparsity(blocks, boundaries, derivative_terms, grid):
    """Sparsity pattern of the Jacobian (generously estimated)."""
    n = int(np.prod(grid.shape))
    pattern = sparse.identity(n, format="csr", dtype=bool)
    for _, m in derivative_terms:
        pattern = pattern + (m != 0)
    # Boundary nodes filled from interior values couple neighbours further
    pattern = (pattern @ pattern).astype(bool)
    rows = []
    for f, _ in blocks:
        free_rows = np.flatnonzero(boundaries[f].free.ravel())
        row = [
            pattern[free_rows][:, np.flatnonzero(boundaries[g].free.ravel())]
            for g, _ in blocks
        ]
        rows.append(row)
    return sparse.bmat(rows, format="csr")
