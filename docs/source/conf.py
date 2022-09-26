# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("../src/"))

import jptranstokenizer

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "jptranstokenizer"
copyright = "2022, Masahiro Suzuki"
author = "Masahiro Suzuki"
release = jptranstokenizer.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
]

# templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "jptranstokenizer"
# html_logo = "path/to/logo.png"
# html_favicon = "path/to/favicon.ico"
# html_theme = "pydata_sphinx_theme"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
todo_include_todos = True

html_sidebars = {"**": ["sidebar-nav-bs"], "left_sidebar_end": []}

html_context = {
    "display_github": True,
    "github_user": "retarfi",
    "github_repo": "jptranstokenizer",
    #   'github_version': 'master/docs/',
}

html_theme_options = {"collapse_navigation": True, "navigation_depth": 1}
