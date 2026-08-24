project = "fastlowess"
author = "Amir Valizadeh"
copyright = "2025-2026, Amir Valizadeh"
release = ""

extensions = [
    "myst_parser",
    "jupyter_sphinx",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

html_theme = "furo"
html_title = "fastlowess"

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "deflist",
    "tasklist",
]
myst_heading_anchors = 3
# Treat ```mermaid fenced blocks as the sphinxcontrib.mermaid directive
myst_fence_as_directive = ["mermaid"]

# jupyter-sphinx: allow blocks to fail without breaking the build
jupyter_allow_errors = True
jupyter_execute_kwargs = {"timeout": 30, "allow_errors": True}

html_static_path = ["assets"]
html_extra_path = ["assets"]

suppress_warnings = ["myst.header", "docutils"]
