# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
import datetime
sys.path.append(os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))
import besttracks


# -- Project information -----------------------------------------------------

project = 'besttracks'
copyright = f'{datetime.datetime.today().year}, MiniUFO'
author = 'MiniUFO'

# The full version, including alpha/beta/rc tags
version = besttracks.__version__
# The full version, including alpha/beta/rc tags
release = besttracks.__version__


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',     # api auto-gen
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',     # math
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'numpydoc',
]

# The suffix(es) of source filenames, either a string or list.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    'conf.py', 'sphinxext', '_build', '_templates', '_themes',
    '**.ipynb_checkpoints' '.DS_Store', 'trash', 'tmp',
]


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'collapse_navigation': True,
    'navigation_depth': 4,
    'prev_next_buttons_location': 'bottom',
}

html_static_path = ['_static']
