"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path("src")
package = root / "nemo_safe_synthesizer"

for path in sorted(package.rglob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    doc_path = path.relative_to(root).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    # Skip private modules, __main__, and test utilities
    if any(part.startswith("_") for part in parts):
        continue
    if parts[-1] == "__init__":
        continue
    if "test" in parts[-1]:
        continue
    # Skip non-importable directories (assets, templates, etc.)
    if "assets" in parts or "jinja" in parts or "css" in parts or "js" in parts:
        continue

    # Build the module identifier string
    identifier = ".".join(parts)

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
