#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# liEensemble documentation build configuration file, created by
# sphinx-quickstart on Fri Aug 18 11:52:31 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime

exec(open('../libensemble/version.py').read())


if sys.version_info >= (3, 3):
    from unittest.mock import MagicMock
else:
    from mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
            return MagicMock()

MOCK_MODULES = [
        'argparse',
        'dfols',
        'math',
        'mpi4py',
        'mpmath',
        'nlopt',
        'numpy',
        'numpy.lib',
        'numpy.lib.recfunctions',
        'numpy.linalg',
        'PETSc',
        'petsc4py',
        'psutil',
        'scipy',
        'scipy.io',
        'scipy.sparse',
        'scipy.spatial',
        'scipy.spatial.distance',
        'scipy.stats',
        'surmise.calibration',
        'surmise.emulation',
        'Tasmanian',
        ]

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

#from libensemble import *
#from libensemble.alloc_funcs import *
#from libensemble.gen_funcs import *
#from libensemble.sim_funcs import *


#sys.path.insert(0, os.path.abspath('.'))
sys.path.append(os.path.abspath('../libensemble'))
##sys.path.append(os.path.abspath('../libensemble'))
sys.path.append(os.path.abspath('../libensemble/alloc_funcs'))
sys.path.append(os.path.abspath('../libensemble/gen_funcs'))
sys.path.append(os.path.abspath('../libensemble/sim_funcs'))
sys.path.append(os.path.abspath('../libensemble/comms'))
sys.path.append(os.path.abspath('../libensemble/utils'))
sys.path.append(os.path.abspath('../libensemble/tools'))
sys.path.append(os.path.abspath('../libensemble/executors'))
sys.path.append(os.path.abspath('../libensemble/resources'))
# print(sys.path)

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = '2.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinxcontrib.bibtex',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              # 'sphinx.ext.autosectionlabel',
              'sphinx.ext.intersphinx',
              'sphinx.ext.imgconverter',
              'sphinx.ext.mathjax']
bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'
# autosectionlabel_prefix_document = True
# extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.imgconverter']
#breathe_projects = { "libEnsemble": "../code/src/xml/" }
#breathe_default_project = "libEnsemble"

##breathe_projects_source = {"libEnsemble" : ( "../code/src/", ["libE.py", "test.cpp"] )}
#breathe_projects_source = {"libEnsemble" : ( "../code/src/", ["test.cpp","test2.cpp"] )}

intersphinx_mapping = {
    'community': ('https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/', None)
}

autodoc_mock_imports = ["balsam"]
extlinks = {'duref': ('http://docutils.sourceforge.net/docs/ref/rst/'
                      'restructuredtext.html#%s', ''),
            'durole': ('http://docutils.sourceforge.net/docs/ref/rst/'
                       'roles.html#%s', ''),
            'dudir': ('http://docutils.sourceforge.net/docs/ref/rst/'
                      'directives.html#%s', '')}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Configure bibtex_bibfiles setting for sphinxcontrib-bibtex 2.0.0
bibtex_bibfiles = ['references.bib']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The latex toctree document.
latex_doc = 'latex_index'

# General information about the project.
project = 'libEnsemble'
copyright = str(datetime.now().year) + ' Argonne National Laboratory'
author = 'Jeffrey Larson, Stephen Hudson, Stefan M. Wild, David Bindel and John-Luke Navarro'
today_fmt = '%B %-d, %Y'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
#html_theme = 'default'
#html_theme = 'graphite'
# html_theme = 'sphinxdoc'
html_theme = 'sphinx_rtd_theme'

html_logo = './images/libE_logo_white.png'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {'navigation_depth': 3,
                      'collapse_navigation': False,
                      'logo_only': True}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']
html_static_path = []

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
#html_sidebars = {
#    '**': [
#        'about.html',
#        'navigation.html',
#        'relations.html',  # needs 'show_related': True theme option to display
#        'searchbox.html',
#        'donate.html',
#    ]
#}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'libEnsembledoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # Sonny, Lenny, or Bjornstrup
    'fncychap': '\\usepackage[Lenny]{fncychap}',
    'extraclassoptions': 'openany',
    'preamble':
    r'''
    \protected\def\sphinxcrossref#1{\texttt{#1}}

    \newsavebox\mytempbox
    \definecolor{sphinxnoteBgColor}{RGB}{221,233,239}
    \renewenvironment{sphinxnote}[1]{%
    \begin{lrbox}{\mytempbox}%
    \begin{minipage}{\columnwidth}%
    \begin{sphinxlightbox}%
    \sphinxstrong{#1}}%
    {\end{sphinxlightbox}%
    \end{minipage}%
    \end{lrbox}%
    \colorbox{sphinxnoteBgColor}{\usebox{\mytempbox}}}
                ''',
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (latex_doc, 'libEnsemble.tex', 'libEnsemble User Manual',
     'Stephen Hudson, Jeffrey Larson, Stefan M. Wild, \\\\ \\hfill David Bindel, John-Luke Navarro', 'manual'),
]

latex_logo = 'images/libE_logo.png'

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (latex_doc, 'libEnsemble', 'libEnsemble User Manual',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (latex_doc, 'libEnsemble', 'libEnsemble User Manual',
     author, 'libEnsemble', 'One line description of project.',
     'Miscellaneous'),
]
