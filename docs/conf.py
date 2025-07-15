# Sphinx configuration for fastkde
import os
import sys
from sphinx.highlighting import lexers
from pygments.lexers import PythonLexer

# fallback for editable installs
try:
    from fastkde import __version__ as release
except ImportError:
    release = "editable"

sys.path.insert(0, os.path.abspath('../src'))

project = 'fastkde'
author = "Travis A. O'Brien"

extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'jupyter_execute/*', '.jupyter_cache']
html_theme = 'sphinx_rtd_theme'
lexers['ipython3'] = PythonLexer()
copyright = "2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory"
nb_execution_mode = "cache"