---
description: Run GPU integration and e2e tests
---
Run GPU-dependent tests. Requires CUDA.

* GPU integration tests: `make test-gpu-integration`
* All e2e tests: `make test-e2e`
* Default e2e only: `make test-e2e-default`
* DP e2e only: `make test-e2e-dp`
* Note: e2e tests run with `-n 0` (single process)
