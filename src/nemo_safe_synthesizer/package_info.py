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

Version is determined at build time from git tags via uv-dynamic-versioning.
At runtime, the version is read from the installed package metadata.

To create a release version, tag a commit:
    git tag v0.2.0
    uv build --wheel
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("nemo-safe-synthesizer")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for editable installs without metadata
