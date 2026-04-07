# typings/

Minimal type stubs for third-party packages that lack upstream `py.typed` support
or whose published stubs are incomplete.

## When to add a stub

Add a `.pyi` file here when:

- A third-party package has no `py.typed` marker and no published stub package
  (e.g. `faiss`, `opacus`, `unsloth`).
- The published stubs are missing definitions that cause `ty: ignore` annotations
  in source code and a small local stub can eliminate the suppression.

## How it works

`pyproject.toml` declares `typings/` as an extra path for `ty`:

```toml
[tool.ty.environment]
extra-paths = ["typings"]
```

Type checkers resolve imports by searching `extra-paths` after the default
source root, so a stub at `typings/faiss/__init__.pyi` shadows the missing
upstream stubs for `import faiss`.

## Guidelines

- Keep stubs minimal -- only declare the symbols actually used in this repo.
- Mirror the package's public API structure (e.g. `opacus/optimizers/__init__.pyi`).
- Include the SPDX copyright header.
- Prefer `Any` for parameters or return types that are not exercised by our code.
- When upstream adds `py.typed` or publishes a stub package, delete the local
  stub and remove any now-unnecessary `ty: ignore` annotations.
