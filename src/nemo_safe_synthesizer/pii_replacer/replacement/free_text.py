# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Propagating a unit's structured replacements into the prose that mentions them."""

from __future__ import annotations

import re
from collections.abc import Hashable, Mapping
from typing import cast

import pandas as pd

from ...config.pii_replacement import PiiReplacementPlan
from ...errors import InternalError
from ..entities import Config
from ..models import FreeTextDetection, PersonaInstance, ScopedValueMap
from ..patterns import split_full_name, split_title

# These helpers only read ``originals`` / ``synthetic``, so a plain mapping with
# those keys works as well as a full instance.
InstanceLike = PersonaInstance | Mapping[str, object]

__all__ = [
    "build_text_substituter",
    "instance_text_pair_labels",
    "instance_text_pairs",
    "resolve_freetext_detections",
]

_TEXT_NAME_LABELS = (
    "first_name",
    "last_name",
    "middle_name",
    "full_name",
    "email",
    "phone_number",
)


# A name token is letters, with the punctuation a name carries inside it ("O'Brien",
# "Smith-Jones") but not the punctuation a column writes around it ("SMITH,").
_NAME_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*")


def resolve_freetext_detections(
    detections: list[FreeTextDetection],
    *,
    original_df: pd.DataFrame,
    contextual_pairs: dict[Hashable, list[tuple[str, str]]],
    standalone_maps: dict[str, ScopedValueMap],
    plan: PiiReplacementPlan,
    cfg: Config,
    group_key: str | None,
) -> dict[Hashable, list[tuple[str, str]]]:
    """Resolve LLM evidence into programmatic, scope-aware substitutions.

    This is the explicit migration seam for the prototype. A future resolver
    will first reuse persona/standalone mappings when a detected value is
    already known, then generate a scoped value through the entity handler for
    newly detected PII. Providers must never generate synthetic values.

    The shipped enhancers return no detections (or fail as not implemented), so
    non-empty input is rejected rather than silently leaving detected PII in
    place or making the provider own replacement semantics.

    Args:
        detections: Free-text detections from the LLM enhancer.
        original_df: Source dataframe.
        contextual_pairs: Row- or unit-local substitution pairs already known.
        standalone_maps: Standalone column maps from the structured pass.
        plan: Resolved replacement plan.
        cfg: Replacement configuration.
        group_key: Training group-key column when scope is ``"group"``.

    Returns:
        Mapping from row index to ``(original, synthetic)`` pairs.

    Raises:
        InternalError: If ``detections`` is non-empty (resolution not implemented).
    """
    if not detections:
        return {}
    raise InternalError(
        "The PII enhancer returned free-text detections, but programmatic "
        "free-text detection resolution is not implemented in this release."
    )


def _name_token(text: str, min_len: int) -> str | None:
    """One word of a name as free text would mention it, or None if it is not a name."""
    match = _NAME_TOKEN_RE.fullmatch(text.strip(" .,;:()"))
    return match.group(0) if match and len(match.group(0)) >= min_len else None


def _iter_instance_text_triples(inst: InstanceLike, cfg: Config | None = None):
    """Yield ``(original, synthetic, taxonomy_label)`` triples for free-text rewrites.

    Field and composite entries overwrite on the same original; name-token aliases
    keep the first mapping so they never clobber a structured field. Both
    ``instance_text_pairs`` and ``instance_text_pair_labels`` derive from this so
    eval labels cannot drift from the emitted pairs.

    Yields:
        ``(original, synthetic, taxonomy_label)`` tuples, longest originals first
        when consumed via ``instance_text_pairs``.

    Example:
        ``full_name`` ``"Smith, Jane"`` -> ``"Jones, Robert"`` also yields
        ``("Smith", "Jones", "last_name")`` so ``"Dr. Smith"`` becomes ``"Dr. Jones"``.
    """
    syn = cast(Mapping[str, str], inst.get("synthetic", {}))
    orig = cast(Mapping[str, str], inst["originals"])
    ordered: list[tuple[str, str, str]] = []
    index_of: dict[str, int] = {}

    def _put(original: str, synthetic: str, label: str, *, keep_first: bool) -> None:
        if original in index_of:
            if keep_first:
                return
            ordered[index_of[original]] = (original, synthetic, label)
            return
        index_of[original] = len(ordered)
        ordered.append((original, synthetic, label))

    for label in _TEXT_NAME_LABELS:
        if label in syn and label in orig:
            _put(str(orig[label]), str(syn[label]), label, keep_first=False)
    if {"first_name", "last_name"} <= set(syn) and {"first_name", "last_name"} <= set(orig):
        _put(
            f"{orig['first_name']} {orig['last_name']}",
            f"{syn['first_name']} {syn['last_name']}",
            "full_name",
            keep_first=False,
        )
    # Name-token aliases: for a person known only by a FULL name (no separate first/last
    # columns), also propagate each name token so honorific/partial mentions stay consistent
    # ("John Smith" -> "Robert Jones" also rewrites a later "Dr. Smith" -> "Dr. Jones"). Done
    # by positionally pairing the original vs synthetic tokens (titles stripped), which line
    # up because the synthetic is written in the column's own convention; only tokens
    # >= freetext_alias_min_token_len letters are aliased so short common words aren't hit.
    # Free text mentions a name without the punctuation the column writes around it, so
    # 'SMITH, Jane' -> 'JONES, Robert' aliases SMITH to JONES, commas left behind.
    # Which token is the surname is the value's own business: a column writing
    # 'SMITH, Jane' leads with one, a column writing 'Jane Smith' ends with it.
    if cfg is None or cfg.freetext_name_token_aliases:
        min_len = cfg.freetext_alias_min_token_len if cfg is not None else 3
        for label in ("full_name",):
            if label not in syn or label not in orig:
                continue
            _, o_rest = split_title(str(orig[label]))
            _, s_rest = split_title(str(syn[label]))
            o_toks, s_toks = o_rest.split(), s_rest.split()
            if len(o_toks) != len(s_toks) or len(o_toks) <= 1:
                continue
            role_of = {text: role for role, text in split_full_name(o_rest).items()}
            for ot, st in zip(o_toks, s_toks):
                o_name, s_name = _name_token(ot, min_len), _name_token(st, min_len)
                if not (o_name and s_name and o_name != s_name):
                    continue
                role = role_of.get(o_name)
                if role is None:
                    continue
                _put(o_name, s_name, role, keep_first=True)
    # NOTE: no age-drift propagation -- the synthetic DOB is only perturbed within
    # +/- 1 year (see synth_dob_programmatic), so any age mentioned in free text is at
    # most 1 year off, which is acceptable and not worth rewriting.
    yield from ordered


def instance_text_pairs(inst: InstanceLike, cfg: Config | None = None) -> list[tuple[str, str]]:
    """Return original→synthetic pairs for propagating structured values into free text.

    Args:
        inst: Persona instance (or mapping with ``originals`` / ``synthetic`` keys).
        cfg: Replacement configuration (controls name-token aliases).

    Returns:
        Pairs sorted longest-original-first for regex alternation precedence.
    """
    pairs = [(original, synthetic) for original, synthetic, _ in _iter_instance_text_triples(inst, cfg)]
    return sorted(pairs, key=lambda kv: len(kv[0]), reverse=True)


def instance_text_pair_labels(inst: InstanceLike, cfg: Config | None = None) -> dict[str, str]:
    """Map original values to taxonomy labels for free-text propagation logging.

    Used only to attach an entity label to free-text propagations for the eval's
    precision/recall log (generation itself is label-agnostic).

    Args:
        inst: Persona instance (or mapping with ``originals`` / ``synthetic`` keys).
        cfg: Replacement configuration (controls name-token aliases).

    Returns:
        ``{original_value: taxonomy_label}`` for the pairs ``instance_text_pairs`` emits.

    Example:
        ``{"Smith": "last_name", "Jane Smith": "full_name"}``
    """
    return {original: label for original, _, label in _iter_instance_text_triples(inst, cfg)}


def _match_case_style(matched: str, synthetic: str) -> str:
    """Reshape ``synthetic`` to the case style of the matched free-text token.

    Args:
        matched: Token as it appeared in the source text.
        synthetic: Replacement string in its stored casing.

    Returns:
        ``synthetic`` adjusted to match ``matched`` (upper, lower, or title case).

    Example:
        Matched ``"SMITH"`` with synthetic ``"Jones"`` -> ``"JONES"``.
    """
    if not matched or not synthetic:
        return synthetic
    if matched.isupper():
        return synthetic.upper()
    if matched.islower():
        return synthetic.lower()
    if matched.istitle():
        return synthetic.title()
    return synthetic


def build_text_substituter(pairs: list[tuple[str, str]]):
    """Build a row-local substituter from original→synthetic pairs.

    One alternation regex over all originals (longest-first), case-insensitive.
    Matching ignores case; the synthetic is reshaped to the matched token's case
    style (all-upper / all-lower / title), otherwise kept as stored.

    Args:
        pairs: ``(original, synthetic)`` pairs; empty originals are skipped.

    Returns:
        Callable that rewrites a string, or ``None`` when ``pairs`` is empty.

    Example:
        Pairs ``[("Jane Smith", "Robert Jones"), ("Jane", "Robert")]``::
            ``"Call JANE SMITH"`` -> ``"Call ROBERT JONES"``
    """
    pairs = [(o, s) for o, s in pairs if o]
    if not pairs:
        return None
    # Longest-first so "Jane Smith" wins over "Jane"; case-insensitive dedupe keeps
    # the first (longest) mapping when the same token appears in multiple casings.
    pairs = sorted(pairs, key=lambda p: len(p[0]), reverse=True)
    by_lower: dict[str, str] = {}
    originals: list[str] = []
    for orig, syn in pairs:
        key = orig.lower()
        if key in by_lower:
            continue
        by_lower[key] = syn
        originals.append(orig)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(o) for o in originals) + r")(?!\w)",
        re.IGNORECASE,
    )

    def _sub(text: object) -> object:
        if not isinstance(text, str) or not text:
            return text

        def _repl(m: re.Match[str]) -> str:
            matched = m.group(1)
            syn = by_lower[matched.lower()]
            return _match_case_style(matched, syn)

        return pattern.sub(_repl, text)

    return _sub
