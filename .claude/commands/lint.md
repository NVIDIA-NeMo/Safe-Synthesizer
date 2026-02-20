---
description: Lint and typecheck code
---
Run linting, type checking, and copyright verification.

* Run with: `make lint`
* Runs three steps:
  1. `bash tools/lint/ruff-lint.sh` -- ruff linting
  2. `bash tools/lint/run-ty-check.sh` -- ty type checking
  3. `python tools/lint/copyright_fixer.py --check .` -- copyright headers
* Always run after `make format`
