# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MkDocs hooks for build-time customizations."""

import posixpath
import re
from typing import Any

from pygments.formatters.html import HtmlFormatter

# pymdownx.highlight passes filename=None to HtmlFormatter when no title is set
# (mkdocstrings source rendering hits this path). Pygments 2.20.0 added
# html.escape() over options.get('filename', ''), which raises AttributeError on
# None. Coerce None to "" so we keep the security floor without breaking docs.
_orig_html_formatter_init = HtmlFormatter.__init__


def _patched_html_formatter_init(self: HtmlFormatter, **options: Any) -> None:
    if options.get("filename") is None:
        options["filename"] = ""
    _orig_html_formatter_init(self, **options)


setattr(HtmlFormatter, "__init__", _patched_html_formatter_init)


def on_page_content(html, page, config, **_kwargs):
    """Rewrite relative doc links in notebook pages to absolute URLs.

    mkdocs-jupyter renders notebook markdown cells without MkDocs's link
    normalization, so relative .md/.ipynb hrefs land in the HTML as-is and
    become broken links on the deployed site. This hook fixes them.
    """
    if not page.file.src_path.endswith(".ipynb"):
        return html

    site_url = (config.site_url or "/").rstrip("/")
    page_dir = posixpath.dirname(page.file.src_path)

    def rewrite(match):
        href = match.group(1)
        if href.startswith(("http://", "https://", "#", "/", "mailto:")):
            return match.group(0)

        href_path, _, fragment = href.partition("#")
        if not any(href_path.endswith(ext) for ext in (".md", ".ipynb")):
            return match.group(0)

        resolved = posixpath.normpath(posixpath.join(page_dir, href_path))
        for ext in (".md", ".ipynb"):
            if resolved.endswith(ext):
                resolved = resolved[: -len(ext)]
                break

        suffix = f"/#{fragment}" if fragment else "/"
        return f'href="{site_url}/{resolved.lstrip("/")}{suffix}"'

    return re.sub(r'href="([^"]*)"', rewrite, html)
