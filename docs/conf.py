# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.util
import os
import sys
import warnings
from pathlib import Path

from sphinx.deprecation import RemovedInSphinx10Warning

warnings.filterwarnings("ignore", category=RemovedInSphinx10Warning, module="sphinx_autodoc_typehints")


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))


project = "GRouNdScale"
copyright = "2026, Milos Micik"
author = "Milos Micik"
version = release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_favicon",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

suppress_warnings = ["sphinx_autodoc_typehints.guarded_import"]

# Mock heavy dependencies that are not installed in the docs build environment so autodoc
# can still import the source modules and document their classes and functions.
# _CANDIDATE_MOCKED_IMPORTS = [
#     "arboreto",
#     "click",
#     "matplotlib",
#     "networkx",
#     "numpy",
#     "optuna",
#     "optunahub",
#     "pandas",
#     "ray",
#     "rich",
#     "rich_click",
#     "scanpy",
#     "scipy",
#     "seaborn",
#     "sklearn",
#     "sparselinear",
#     "torch",
#     "torch_scatter",
#     "torch_sparse",
#     "tqdm",
#     "umap",
# ]
# autodoc_mock_imports = [
#     name for name in _CANDIDATE_MOCKED_IMPORTS if importlib.util.find_spec(name) is None
# ]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.11", None),
    "pytorch": ("https://docs.pytorch.org/docs/2.9/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# NOTE: autodoc_type_aliases is deliberately NOT used. Mapping TYPE_CHECKING-only string
# annotations through it makes Sphinx 9.0.4 render nested annotations as
# `tuple[TypeAliasForwardRef('...')]` (non-linked and ugly). Types that should be linked
# must be resolvable at runtime instead: e.g. `from pathlib import Path` must live outside
# the `if TYPE_CHECKING:` block so autodoc can resolve the annotation to the real class.

# napoleon_use_rtype = False  # sphinx.ext.napoleon setting
# NOTE: napoleon_preprocess_types is kept off; its comma-splitting mangles generic type
# annotations such as dict[str, float] and produces broken cross-references.
napoleon_google_docstring = False  # sphinx.ext.napoleon setting
napoleon_type_aliases = {
    "Tensor": "torch.Tensor",
    "Module": "torch.nn.Module",
    "nn.Module": "torch.nn.Module",
    "nn.Sequential": "torch.nn.Sequential",
    "Trial": "optuna.Trial",
    "Path": "pathlib.Path",
    "Figure": "matplotlib.figure.Figure",
    "DataFrame": "pandas.DataFrame",
    "pd.DataFrame": "pandas.DataFrame",
    "AnnData": "anndata.AnnData",
    "sc.AnnData": "anndata.AnnData",
    "np.ndarray": "numpy.ndarray",
    "csr_matrix": "scipy.sparse.csr_matrix",
    "DataLoader": "torch.utils.data.DataLoader",
    "LRScheduler": "torch.optim.lr_scheduler.LRScheduler",
}

# Objects referenced by autodoc signatures/docstrings for which no Sphinx cross-reference
# target can exist: type aliases (Tensor, Path, ...), library-internal objects, and
# fragments of generic annotations.
nitpick_ignore_regex = [
    (r"py:class", r"Tensor"),
    (r"py:class", r"Tensor\]"),
    (r"py:class", r"\.?\.?tuple\[Tensor"),
    (r"py:class", r"\.?Tuple\[Tensor"),
    (r"py:class", r"tuple\[nn\.Module"),
    (r"py:class", r"tuple\[np\.ndarray"),
    (r"py:class", r"np\.ndarray \| None\]"),
    (r"py:class", r"np\.ndarray"),
    (r"py:class", r"dict\[str"),
    (r"py:class", r"float\]"),
    (r"py:class", r"str \| None"),
    (r"py:class", r"Path"),
    (r"py:class", r"Figure"),
    (r"py:class", r"optional"),
    (r"py:class", r"Module"),
    (r"py:class", r"nn\.Module"),
    (r"py:class", r"nn\.Sequential"),
    (r"py:class", r"DataLoader"),
    (r"py:class", r"Trial"),
    (r"py:class", r"LRScheduler"),
    (r"py:class", r"csr_matrix"),
    (r"py:class", r"dict\[str, float\]"),
    (r"py:class", r"tuple\[Tensor, Tensor\]"),
    (r"py:class", r"'?\)"),
    (r"py:class", r"pandas\.core\..*"),
    (r"py:class", r"anndata\._core\..*"),
    # String annotations on the GRN inference backends. The modules import
    # pandas/scanpy lazily inside the functions so that the standalone
    # scripts stay cheap to import, which leaves these unresolvable at runtime.
    (r"py:class", r"pd\.DataFrame"),
    (r"py:class", r"sc\.AnnData"),
    (r"py:class", r"preprocessing\.grn_creation\._T"),
    (r"py:class", r"hyperparameter_optimization\._T"),
    (r"py:obj", r"hyperparameter_optimization\._T"),
    # Third-party classes without a usable Sphinx inventory (rich, tqdm)
    (r"py:class", r"rich(\..*)?"),
    (r"py:class", r"RichHandler"),
    (r"py:class", r"std_tqdm"),
    (r"py:class", r"tqdm(\..*)?"),
    (r"py:class", r"torch\.autograd\.function\..*"),
    (r"py:class", r"training\.dicts\.Losses"),
    (r"py:attr", r"ctx(\..*)?"),
    (r"py:func", r"ctx(\..*)?"),
    (r"py:obj", r"training\.dicts\.Losses"),
]

typehints_fully_qualified = False  # show unqualified type names
python_use_unqualified_type_names = True  # show unqualified type names
typehints_use_signature = True  # show parameter types in signature
typehints_use_signature_return = True  # show return type in signature
always_document_param_types = True
typehints_defaults = "comma"
autodoc_typehints = "none"

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__, __len__, __getitem__",
    # TypedDict keys such as "Generator Loss" contain spaces and cannot be documented as
    # py:attribute directives; excluding them before autodoc resolves names prevents
    # "invalid signature for autoattribute" warnings.
    "exclude-members": (
        "Generator Loss, "
        "Generator Total Loss, "
        "Critic Fake Loss, "
        "Critic Real Loss, "
        "Critic Gradient Penalty Loss, "
        "Critic Total Loss, "
        "Total Loss, "
        "Generator Labeler Loss, "
        "Generator Antilabeler Loss, "
        "Labeler Fake Loss, "
        "Labeler Real Loss, "
        "Antilabeler Loss"
    ),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_context = {
    "display_github": True,
    "github_user": "Emad-COMBINE-lab",
    "github_repo": "GRouNdScale",
    "github_version": "master/",
    "conf_py_path": "docs/",  # Path in the checkout to the docs root
    "default_mode": "light",
}

# pygments_style = "monokai"  # or xcode'
html_logo = "_static/logo.svg"


html_theme_options = {
    "repository_url": "https://github.com/milos7250/GRouNdScale",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_source_button": True,
    "repository_branch": "master",
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "use_download_button": True,
    # 'collapse_navigation': False,
    "logo": {
        "alt_text": "GRouNdScale - Home",
        # "text": "",
    },
    # "show_navbar_depth": 4,
    # "home_page_in_toc": True,
    "show_toc_level": 2,
    #   "announcement": "My announcement!",
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            "url": "https://github.com/milos7250/GRouNdScale",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Docker",
            "url": "https://hub.docker.com/repository/docker/milos7250/groundscale/",
            "icon": "fa-brands fa-docker",
            "type": "fontawesome",
        },
        {
            # Label for this link
            "name": "Manuscript",
            "url": "https://doi.org/10.1038/s41467-024-48516-6",
            "icon": "fa-solid fa-book",
            "type": "fontawesome",
        },
        {
            "name": "Docs Build",
            "url": "https://github.com/milos7250/GRouNdScale/actions",
            "icon": "https://github.com/milos7250/GRouNdScale/actions/workflows/documentation.yaml/badge.svg?branch=master",
            "type": "url",
        },
    ],
}


def _skip_invalid_typed_dict_members(app, what, name, obj, skip, options):
    """Avoid documenting TypedDict keys as Python attributes."""
    return skip or (what == "attribute" and " " in name)


def setup(app):
    app.connect("autodoc-skip-member", _skip_invalid_typed_dict_members)


favicons = [
    "logo.svg",
]

# html_sidebars = {
# }
