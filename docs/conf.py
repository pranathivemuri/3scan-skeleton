import os

# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

project = 'Skeletonization'
copyright = '2019, Skeletonization'
author = 'Skeletonization'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = ''


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.graphviz',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinxcontrib.plantuml'
]
source_parsers = {
    '.md': 'recommonmark.parser.CommonMarkParser',
}

templates_path = ['.templates']
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'
language = None
exclude_patterns = []
pygments_style = 'sphinx'
plantuml = 'java -jar %s' % os.path.abspath("plantuml.jar")


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "canonical_url": "",  # maybe a github page
    'logo_only': True,
    'prev_next_buttons_location': 'bottom',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
}
html_static_path = ['_static']


def setup(app):
    app.add_stylesheet('css/custom.css')


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'skeletonizationdoc'

latex_elements = {
    # TODO
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'skeletonization.tex', 'skeletonization Documentation',
     'Skeleton', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'skeletonization', 'skeletonization Documentation', [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'skeletonization', 'skeletonization Documentation',
     author, 'skeletonization', 'One line description of project.',
     'Miscellaneous'),
]

# -- Options for _todo extension ----------------------------------------------
todo_include_todos = True
