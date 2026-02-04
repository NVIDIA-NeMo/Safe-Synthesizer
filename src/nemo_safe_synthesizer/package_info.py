# Copyright (c) 2024-2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Package version information for NeMo Safe Synthesizer.

This file is used by the FW-CI-templates release workflow to manage versions.
The MAJOR, MINOR, PATCH, and PRE_RELEASE variables are automatically updated
during the release process.
"""

MAJOR = 0
MINOR = 0
PATCH = 0
PRE_RELEASE = ""


def get_version() -> str:
    """Get the full version string.

    Returns:
        Version string in the format MAJOR.MINOR.PATCH[PRE_RELEASE]
    """
    version = f"{MAJOR}.{MINOR}.{PATCH}"
    if PRE_RELEASE:
        version = f"{version}{PRE_RELEASE}"
    return version


__version__ = get_version()
