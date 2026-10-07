# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harden vLLM's Outlines caches against diskcache poisoning (CVE-2025-69872).

diskcache (pulled in transitively by outlines and used by vLLM's optional
on-disk outlines cache) deserializes cached values with pickle/cloudpickle and
is therefore RCE-vulnerable if another principal can write into the cache
directory. Neither library exposes a way to swap the serializer, so NSS
mitigates at the boundary:

1. Keep vLLM's opt-in diskcache off (its default is an in-memory LRUCache).
   Hard-set rather than defaulted, so a user environment can't silently turn
   on a pickle-deserializing code path.
2. Pin ``OUTLINES_CACHE_DIR`` to a per-user path and chmod it to 0700, since
   outlines always uses diskcache for its FSM/index cache.

Kept free of vLLM imports so callers that only prepare a vLLM subprocess
environment can apply the same protection.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import MutableMapping
from pathlib import Path

from .observability import get_logger

__all__ = ["harden_outlines_cache", "secure_outlines_cache_dir"]

logger = get_logger(__name__)


def secure_outlines_cache_dir(environ: MutableMapping[str, str] | None = None) -> Path:
    """Pin ``OUTLINES_CACHE_DIR`` to a per-user path and tighten permissions.

    Respects an explicit ``OUTLINES_CACHE_DIR`` set by the operator (so CI and
    multi-tenant deployments can choose their own private location), but always
    creates the directory with 0700 permissions to prevent co-tenants from
    poisoning the diskcache (CVE-2025-69872).

    When unset, picks a per-user path under ``$XDG_CACHE_HOME`` or
    ``$HOME/.cache`` and falls back to a UID-scoped subdir of the system temp
    dir for distroless/rootless containers where ``$HOME`` is ``/``.

    Args:
        environ: Environment to read and update; defaults to ``os.environ``.

    Returns:
        The cache directory.
    """
    env = os.environ if environ is None else environ
    cache_dir_env = env.get("OUTLINES_CACHE_DIR")
    if cache_dir_env:
        cache_dir = Path(cache_dir_env)
    else:
        xdg_cache_home = env.get("XDG_CACHE_HOME")
        try:
            home_dir: Path | None = Path.home()
        except RuntimeError:
            home_dir = None
        if xdg_cache_home:
            cache_root = Path(xdg_cache_home)
        elif home_dir is not None and home_dir != Path("/") and home_dir.is_dir():
            cache_root = home_dir / ".cache"
        else:
            uid = getattr(os, "getuid", lambda: "default")()
            cache_root = Path(tempfile.gettempdir()) / f".cache-{uid}"
        cache_dir = cache_root / "nemo-safe-synthesizer" / "outlines"
        env["OUTLINES_CACHE_DIR"] = str(cache_dir)

    try:
        # Set the umask to 077 to prevent other principals from writing to the
        # cache directory between the mkdir and chmod calls.
        old_umask = os.umask(0o077)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        finally:
            os.umask(old_umask)
        # Also explicitly set permissions to 0700 for the situation where the
        # directory already exists and is not 0700.
        cache_dir.chmod(0o700)
    except OSError as exc:
        logger.warning(
            "Could not enforce 0700 permissions on outlines cache dir %s: %s. "
            "If this path is shared with other principals, set OUTLINES_CACHE_DIR "
            "to a private location (CVE-2025-69872).",
            cache_dir,
            exc,
        )
    return cache_dir


def harden_outlines_cache(environ: MutableMapping[str, str] | None = None) -> None:
    """Turn off vLLM's on-disk Outlines cache and secure ``OUTLINES_CACHE_DIR``.

    Args:
        environ: Environment to update; defaults to ``os.environ``. Pass a
            subprocess environment to protect a separately launched vLLM server.
    """
    env = os.environ if environ is None else environ
    env["VLLM_V1_USE_OUTLINES_CACHE"] = "0"
    secure_outlines_cache_dir(env)
