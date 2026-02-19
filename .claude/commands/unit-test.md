---
description: Run unit tests
---
Run unit tests excluding slow tests.

* Run all unit tests: `make test`
* Run a single test file: `uv run --frozen pytest path/to/test.py -vvs -n0`
* Run a single test function: `uv run --frozen pytest path/to/test.py::test_name -vvs -n0`
* Underlying command: `uv run --frozen pytest -n auto --dist loadscope -vv -m "unit and not slow"`
