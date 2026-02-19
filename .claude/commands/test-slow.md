---
description: Run all tests including slow
---
Run all tests including slow tests (excludes e2e).

* Run with: `make test-slow`
* Underlying command: `uv run --frozen pytest tests -n auto --dist loadscope -vv -m "not e2e" --run-slow`
