# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Odecast"
copyright = "2025, Matthias Baer"
author = "Matthias Baer"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/odecast-logo.png"
html_favicon = "_static/odecast-logo.png"

# GitHub Pages configuration
html_baseurl = "https://maroba.github.io/odecast/"
html_theme_options = {
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

# Math rendering
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

# Plot configuration
plot_include_source = True
plot_html_show_source_link = False
plot_formats = ["png", "svg"]
plot_basedir = "_static/plots"

# NBSphinx configuration
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True
