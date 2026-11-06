#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Submit NSS experiments to the Slurm cluster. The logic lives in nss_slurm.py
# ("submit" subcommand); run with --help for the options.

set -euo pipefail

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nss_slurm.sh" submit "$@"
