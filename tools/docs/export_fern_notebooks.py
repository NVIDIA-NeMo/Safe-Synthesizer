# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the tutorial notebooks to Fern MDX without executing any cells.

The notebooks are the authoring source. This exporter preserves Markdown and
code cells; saved outputs are intentionally omitted so runtime data is not
published accidentally. Run from the repository root with ``uv run
--no-project tools/docs/export_fern_notebooks.py`` after editing a notebook.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = """---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""


def rewrite_link(match: re.Match[str]) -> str:
    """Keep notebook links independent of the selected Fern version."""
    target = match[1]
    if target.startswith(("https://", "http://", "#", "mailto:")):
        return match[0]
    path, sep, fragment = target.partition("#")
    if path.endswith(".ipynb"):
        path = f"/tutorials/{Path(path).stem}"
    elif path.endswith(".md"):
        path = "/" + path.removeprefix("../").removesuffix(".md")
    return f"]({path}{sep}{fragment})"


def export_notebook(path: Path) -> None:
    """Write a static reading companion with a link to the runnable source."""
    notebook = json.loads(path.read_text())
    cells = []
    title = path.stem
    for cell in notebook["cells"]:
        source = "".join(cell["source"]).strip()
        if not source:
            continue
        if cell["cell_type"] == "markdown":
            if not cells and source.startswith("# "):
                first, _, source = source.partition("\n")
                title = first[2:]
            # Keep dataset citations as a regular list in the reading companion.
            if "> Dataset citations:" in source:
                source = re.sub(r"^> ?", "", source, flags=re.MULTILINE)
            cells.append(re.sub(r"\]\(([^)]+)\)", rewrite_link, source))
        elif cell["cell_type"] == "code":
            fence = "`" * max(3, max((len(x) + 1 for x in re.findall(r"`+", source)), default=0))
            cells.append(f"{fence}python\n{source}\n{fence}")
    source_url = f"https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/docs/tutorials/{path.name}"
    content = HEADER + f"title: {json.dumps(title)}\n---\n\n"
    content += "{/* Generated from the adjacent notebook; regenerate with mise run docs:fern:notebooks. */}\n\n"
    content += f"[Open the runnable notebook on GitHub]({source_url}). Download the tutorial datasets from the same directory.\n\n"
    content += "\n\n".join(cells) + "\n"
    path.with_suffix(".mdx").write_text(content)


if __name__ == "__main__":
    for notebook_path in sorted((ROOT / "docs/tutorials").glob("*.ipynb")):
        export_notebook(notebook_path)
