Contributing
============

We welcome contributions to Odecast! This guide will help you get started.

Development Setup
-----------------

First, fork the repository on GitHub and clone your fork:

.. code-block:: bash

   git clone https://github.com/yourusername/odecast.git
   cd odecast

Create a virtual environment and install development dependencies:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"

This installs Odecast in editable mode with all development tools.

Code Quality
------------

We maintain high code quality standards using several tools:

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=odecast

   # Run specific test file
   pytest tests/test_api.py

   # Run with verbose output
   pytest -v

Code Formatting
~~~~~~~~~~~~~~~

We use Black for code formatting:

.. code-block:: bash

   # Format all code
   black .

   # Check formatting without making changes
   black --check .

Linting
~~~~~~~

We use Ruff for fast linting:

.. code-block:: bash

   # Lint all code
   ruff check .

   # Fix auto-fixable issues
   ruff check --fix .

Type Checking
~~~~~~~~~~~~~

We use mypy for static type checking:

.. code-block:: bash

   mypy src/odecast

Pre-commit Hooks
~~~~~~~~~~~~~~~~

Install pre-commit hooks to automatically run checks:

.. code-block:: bash

   pre-commit install

   # Run hooks manually
   pre-commit run --all-files

Testing Guidelines
------------------

Test-Driven Development
~~~~~~~~~~~~~~~~~~~~~~~

Odecast follows a test-driven development approach. All functionality should be
defined by comprehensive tests before implementation.

Writing Tests
~~~~~~~~~~~~~

* Place tests in the ``tests/`` directory
* Use descriptive test names: ``test_solve_second_order_with_initial_conditions``
* Test both success and failure cases
* Include edge cases and boundary conditions
* Use pytest fixtures for common setup

Example test structure:

.. code-block:: python

   import pytest
   from odecast import var, Eq, solve
   from odecast.errors import ODEValidationError

   def test_simple_harmonic_oscillator():
       """Test solving y'' + y = 0 with initial conditions."""
       y = var("y")
       eq = Eq(y.d(2) + y, 0)
       sol = solve(eq, ivp={y: 1, y.d(): 0}, tspan=(0, 1))
       
       assert sol is not None
       assert len(sol.t) > 0
       assert len(sol[y]) == len(sol.t)

   def test_missing_initial_condition_raises_error():
       """Test that missing initial conditions raise validation errors."""
       y = var("y")
       eq = Eq(y.d(2) + y, 0)
       
       with pytest.raises(ODEValidationError):
           solve(eq, ivp={y: 1}, tspan=(0, 1))  # Missing y.d()

Documentation
-------------

Code Documentation
~~~~~~~~~~~~~~~~~~

* Use Google-style docstrings for all public functions and classes
* Include parameter types and descriptions
* Provide usage examples in docstrings
* Document return values and exceptions

Example docstring:

.. code-block:: python

   def solve(equations, ivp=None, bvp=None, tspan=None, backend="auto"):
       """Solve a system of ordinary differential equations.

       Args:
           equations: Single equation or list of equations to solve.
           ivp: Initial value conditions as a dictionary.
           bvp: Boundary value conditions as a dictionary.
           tspan: Time span as a tuple (t_start, t_end).
           backend: Solver backend ("scipy", "sympy", or "auto").

       Returns:
           Solution: Object containing the solution data.

       Raises:
           ODEValidationError: If initial/boundary conditions are invalid.
           ValueError: If equation system is malformed.

       Example:
           >>> y = var("y")
           >>> eq = Eq(y.d(2) + y, 0)
           >>> sol = solve(eq, ivp={y: 1, y.d(): 0}, tspan=(0, 10))
       """

Sphinx Documentation
~~~~~~~~~~~~~~~~~~~~~

* Update relevant .rst files when adding new features
* Include examples in the documentation
* Build docs locally to test changes:

.. code-block:: bash

   cd docs
   make html
   open _build/html/index.html

Contribution Workflow
---------------------

1. **Create an Issue**: Discuss your idea or bug report in a GitHub issue
2. **Fork and Branch**: Create a feature branch from main
3. **Implement**: Write code following our guidelines
4. **Test**: Ensure all tests pass and add new tests for your changes
5. **Document**: Update documentation if needed
6. **Submit PR**: Create a pull request with a clear description

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

* Use descriptive commit messages
* Keep changes focused and atomic
* Update CHANGELOG.md if applicable
* Ensure all CI checks pass
* Request review from maintainers

Commit Message Format
~~~~~~~~~~~~~~~~~~~~~

Use conventional commit format:

.. code-block::

   type(scope): description

   body (optional)

   footer (optional)

Types: ``feat``, ``fix``, ``docs``, ``test``, ``refactor``, ``perf``, ``ci``

Examples:

.. code-block::

   feat(api): add support for boundary value problems
   fix(solve): handle edge case in vector validation
   docs: update installation instructions
   test: add comprehensive tests for symbolic backend

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

* Clear description of the problem
* Minimal reproducible example
* Expected vs. actual behavior
* Environment details (Python version, OS, package versions)

Feature Requests
~~~~~~~~~~~~~~~~

For new features, provide:

* Clear use case and motivation
* Proposed API design
* Examples of intended usage
* Discussion of alternatives considered

Code Contributions
~~~~~~~~~~~~~~~~~~

Areas where contributions are especially welcome:

* New example applications
* Performance improvements
* Additional solver backends
* Enhanced error messages
* Documentation improvements

Examples and Tutorials
~~~~~~~~~~~~~~~~~~~~~~

We particularly welcome:

* Real-world application examples
* Jupyter notebook tutorials
* Domain-specific use cases
* Performance comparisons

Community Guidelines
--------------------

* Be respectful and inclusive
* Help others learn and contribute
* Provide constructive feedback
* Follow the project's coding standards
* Credit others for their contributions

Getting Help
------------

* GitHub Issues: Bug reports and feature requests
* GitHub Discussions: General questions and community chat
* Documentation: Read the docs for detailed guidance

Recognition
-----------

Contributors are recognized in:

* CONTRIBUTORS.md file
* Release notes for significant contributions
* GitHub contributor statistics

Thank you for helping make Odecast better!
