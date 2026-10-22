Contributing
============

Bug reports, feature requests and pull requests are welcome on
`GitHub <https://github.com/maroba/odecast>`_.

Reporting bugs
--------------

Please include a minimal example that reproduces the problem, the full
traceback, and the versions of Python, odecast, NumPy, SciPy and SymPy.

Development setup
-----------------

Fork the repository, then:

.. code-block:: bash

   git clone https://github.com/<your-username>/odecast.git
   cd odecast
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"

Tests
-----

The test suite uses pytest:

.. code-block:: bash

   pytest
   pytest --cov=odecast          # with coverage
   pytest tests/test_api_vector.py

New functionality should come with tests covering both the intended use and
the relevant error cases. Bug fixes should include a test that fails without
the fix.

Code style
----------

CI checks formatting with Black and linting with Ruff. Both versions are
pinned in ``pyproject.toml`` so that local results match CI; the ``dev``
extra installs them. Run both before submitting:

.. code-block:: bash

   black .
   ruff check .

Public functions and classes have Google-style docstrings, which are
rendered in the :doc:`api_reference`. Please follow the same convention.

Documentation
-------------

The documentation is built with Sphinx:

.. code-block:: bash

   pip install -r docs/requirements.txt
   cd docs
   make html

The result is in ``docs/_build/html``. If a change affects user-facing
behaviour, please update the relevant page and add an entry to
``CHANGELOG.md``.

Pull requests
-------------

Keep pull requests focused on a single change and describe the motivation
in the description. For larger changes, consider opening an issue first to
discuss the design.
