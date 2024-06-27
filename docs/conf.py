from __future__ import annotations

import os
import sys
import importlib.metadata

# Add the current directory to the system path
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'DAS4Whales'
copyright = "2024, Bouffaut & Goestchel"
author = 'Léa Bouffaut, Quentin Goestchel'
# version = release = importlib.metadata.version("package")

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",  # To include links to the source code
    "sphinx_copybutton",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
]

templates_path = ['_templates']
source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo' #'sphinx_rtd_theme'

html_theme_options = {
    # "light_logo": "images/light_logo.png",  # Path to your light theme logo
    # "dark_logo": "images/dark_logo.png",    # Path to your dark theme logo
    "light_css_variables": {
        "color-brand-primary": "#005f73",  # Ocean Blue
        "color-brand-content": "#0a9396",  # Aqua/Turquoise
        "color-background-primary": "#ffffff",  # White
        "color-background-secondary": "#e0fbfc",  # Very light aqua
        "color-foreground-primary": "#023047",  # Deep Sea Green
        "color-foreground-secondary": "#8ecae6",  # Light aqua
        "color-link": "#ffb703",  # Coral
        "color-link--hover": "#fb8500",  # Darker Coral
    },
    "dark_css_variables": {
        "color-brand-primary": "#0a9396",  # Aqua/Turquoise
        "color-brand-content": "#94d2bd",  # Lighter Aqua
        "color-background-primary": "#001219",  # Very dark blue
        "color-background-secondary": "#003049",  # Deep Ocean Blue
        "color-foreground-primary": "#e0fbfc",  # Very light aqua
        "color-foreground-secondary": "#94d2bd",  # Lighter Aqua
        "color-link": "#ffb703",  # Coral
        "color-link--hover": "#fb8500",  # Darker Coral
    },
    'github_url': 'https://github.com/qgoestch/DAS4Whales'
}

html_static_path = ['_static']
html_css_files = [
    "css/custom.css",
]

myst_enable_extensions = [
    "colon_fence",
    "amsmath",
    "deflist", # For lists of definitions
    "dollarmath", # For inline math
    "html_admonition",  # For custom admonitions (e.g., danger, note)
    "html_image",  # For aligning images
    "smartquotes", # For automatic curly quotes
    "substitution", # For |variable| substitutions
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True

# -- Extension configuration -------------------------------------------------

# Add any custom extension configuration here...
