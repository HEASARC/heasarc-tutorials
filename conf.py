# Configuration file for the Sphinx documentation builder.

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HEASARC Tutorials'
copyright = '2025, HEASARC developers'
author = 'HEASARC developers'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Sphinx-specific extensions - the most important here is MyST, and the copybutton extension will also
#  add a small copy button (isn't that shocking) next to code blocks
extensions = ['myst_nb', 'sphinx_copybutton']

# Adding amsmath and dollarmath enables LaTeX style math environments
# The smartquotes extension will automatically convert '' and "" to their nice typeset open and closed versions
# The substitution extension allows us to define keys that will be substituted for a value set in the frontmatter
#  or a centralized file - will be good for defining a value we might want to change everywhere easily
# The colon_fence extension lets us use ::: in place of ``` to delimit directives (I am more used to ::: from using
#  MySTMD)
myst_enable_extensions = ['amsmath', 'dollarmath', 'smartquotes', 'substitution', 'colon_fence']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.tox', '.tmp', '.pytest_cache', 'README.md',
                    '**/*_template*', '**/README.md', '*_template*']

# MyST-NB configuration
nb_execution_timeout = 1200
nb_merge_streams = True
nb_execution_mode = "cache"
# nb_execution_mode = "force"
nb_scroll_outputs = True

nb_execution_excludepatterns = ['*notebook_template*', '*pull_request_template*', '*README*', '**/*README*']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'
html_title = 'HEASARC Tutorial Notebooks'
# html_logo = '_static/irsa_logo.png'
# html_favicon = '_static/irsa-favicon.ico'
html_theme_options = {
    "github_url": "https://github.com/HEASARC/heasarc-tutorials",
    "repository_url": "https://github.com/HEASARC/heasarc-tutorials",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "logo": {
        "link": "https://heasarc.gsfc.nasa.gov/",
        "alt_text": "High Energy Astrophysics Science Archive Research Center - Home",
    },
    "home_page_in_toc": True,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# myst configurations
myst_heading_anchors = 4
