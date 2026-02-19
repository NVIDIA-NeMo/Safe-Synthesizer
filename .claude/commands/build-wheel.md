---
description: Build Python wheel package
---
Build the wheel package. Version comes from git tags via uv-dynamic-versioning.

* Run with: `make build-wheel`
* Underlying commands:
  ```bash
  rm -rf dist/
  uv build --wheel
  ```
* Output: `dist/*.whl`
