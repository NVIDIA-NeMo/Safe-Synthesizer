# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pydantic_click_options: _collect_params, _click_type, parse_overrides, decorator."""

from __future__ import annotations

import click
from click.testing import CliRunner
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.configurator.pydantic_click_options import (
    FlagParam,
    LeafParam,
    _collect_params,
    parse_overrides,
    pydantic_options,
)

# ---------------------------------------------------------------------------
# Minimal fixture models
# ---------------------------------------------------------------------------


class Inner(BaseModel):
    value: int = Field(default=1, description="An inner value.")


class Mid(BaseModel):
    inner: Inner = Field(default_factory=Inner)
    label: str = Field(default="x", description="A label.")


class Deep(BaseModel):
    mid: Mid = Field(default_factory=Mid)
    flag: bool = Field(default=False, description="A flag.")


class ModelWithNullable(BaseModel):
    nested: Inner | None = Field(default_factory=Inner, description="A nullable nested model.")
    scalar: str | None = Field(default=None, description="A nullable scalar.")


class ModelWithBoth(BaseModel):
    """A model with a plain sub-model and a nullable sub-model side by side."""

    plain: Inner = Field(default_factory=Inner)
    optional: Inner | None = Field(default_factory=Inner, description="Nullable sub-model.")


# ---------------------------------------------------------------------------
# parse_overrides
# ---------------------------------------------------------------------------


def test_parse_overrides_no_flag_true_injects_none():
    assert parse_overrides({"no_replace_pii": True}) == {"replace_pii": None}


def test_parse_overrides_no_flag_false_is_dropped():
    assert parse_overrides({"no_replace_pii": False}) == {}


def test_parse_overrides_no_flag_with_other_keys():
    result = parse_overrides({"no_replace_pii": True, "training__batch_size": "4"})
    assert result == {"replace_pii": None, "training": {"batch_size": "4"}}


def test_parse_overrides_none_values_dropped():
    result = parse_overrides({"training__batch_size": None, "training__lr": "0.001"})
    assert result == {"training": {"lr": "0.001"}}


def test_parse_overrides_empty_and_none():
    assert parse_overrides({}) == {}
    assert parse_overrides(None) == {}


def test_parse_overrides_nested_setdefault_does_not_overwrite():
    result = parse_overrides({"training__batch_size": "4", "training__lr": "0.001"})
    assert result == {"training": {"batch_size": "4", "lr": "0.001"}}


def test_parse_overrides_field_sep_dot():
    """Config validate uses field_sep='.' -- no_ handling must still work."""
    result = parse_overrides({"no_replace_pii": True, "training.batch_size": "4"}, field_sep=".")
    assert result == {"replace_pii": None, "training": {"batch_size": "4"}}


def test_parse_overrides_deep_nesting():
    result = parse_overrides({"a__b__c": "x", "a__b__d": "y"})
    assert result == {"a": {"b": {"c": "x", "d": "y"}}}


def test_parse_overrides_five_levels():
    result = parse_overrides({"a__b__c__d__e": "v"})
    assert result == {"a": {"b": {"c": {"d": {"e": "v"}}}}}


def test_parse_overrides_mixed_depth():
    result = parse_overrides(
        {
            "training__batch_size": "4",
            "replace_pii__globals__seed": "42",
        }
    )
    assert result == {
        "training": {"batch_size": "4"},
        "replace_pii": {"globals": {"seed": "42"}},
    }


def test_parse_overrides_empty_segment_raises():
    import pytest

    with pytest.raises(ValueError, match="Invalid override key"):
        parse_overrides({"a____b": "x"})


# ---------------------------------------------------------------------------
# _collect_params
# ---------------------------------------------------------------------------


def test_collect_params_leaf_for_scalar():
    params = _collect_params(Inner)
    assert len(params) == 1
    p = params[0]
    assert isinstance(p, LeafParam)
    assert p.name == "value"
    assert isinstance(p.field, FieldInfo)


def test_collect_params_flag_for_nullable_model():
    params = _collect_params(ModelWithNullable)
    names = {p.name for p in params}
    # sub-field of Inner
    assert "nested.value" in names
    # is-flag for the nullable model
    assert "no_nested" in names
    # scalar | None must NOT produce a flag
    assert "no_scalar" not in names


def test_collect_params_flag_has_correct_field_name():
    params = _collect_params(ModelWithNullable)
    flag = next(p for p in params if isinstance(p, FlagParam))
    assert flag.name == "no_nested"
    assert flag.field_name == "nested"


def test_collect_params_no_raw_nested_leaf():
    """The nullable-model field itself must not appear as a LeafParam."""
    params = _collect_params(ModelWithNullable)
    leaf_names = {p.name for p in params if isinstance(p, LeafParam)}
    assert "nested" not in leaf_names


def test_collect_params_sorted():
    params = _collect_params(ModelWithNullable)
    names = [p.name for p in params]
    assert names == sorted(names)


def test_collect_params_deep_nesting():
    params = _collect_params(Deep)
    names = {p.name for p in params}
    assert "mid.inner.value" in names
    assert "mid.label" in names
    assert "flag" in names


def test_collect_params_mixed_plain_and_nullable():
    params = _collect_params(ModelWithBoth)
    names = {p.name for p in params}
    assert "plain.value" in names
    assert "optional.value" in names
    assert "no_optional" in names
    assert "no_plain" not in names


# ---------------------------------------------------------------------------
# decorator -- param inspection
# ---------------------------------------------------------------------------


def test_no_double_emit_for_nullable_model():
    """Nullable-model fields must not appear as both sub-fields and a leaf STRING option."""

    @pydantic_options(ModelWithNullable)
    @click.command()
    def cmd(**kwargs):
        pass

    param_names = {p.name for p in cmd.params}
    assert "nested__value" in param_names
    # raw "nested" must not be a plain STRING option
    assert "nested" not in param_names


def test_no_flag_emitted_for_nullable_model():
    @pydantic_options(ModelWithNullable)
    @click.command()
    def cmd(**kwargs):
        pass

    param_names = {p.name for p in cmd.params}
    assert "no_nested" in param_names


def test_no_flag_is_bool_type():
    @pydantic_options(ModelWithNullable)
    @click.command()
    def cmd(**kwargs):
        pass

    flag = next(p for p in cmd.params if p.name == "no_nested")
    assert flag.is_flag


def test_no_flag_not_emitted_for_nullable_scalar():
    @pydantic_options(ModelWithNullable)
    @click.command()
    def cmd(**kwargs):
        pass

    param_names = {p.name for p in cmd.params}
    assert "no_scalar" not in param_names


def test_decorator_mixed_plain_and_nullable():
    """Plain sub-models get sub-fields only; nullable sub-models get sub-fields plus a --no_ flag."""

    @pydantic_options(ModelWithBoth)
    @click.command()
    def cmd(**kwargs):
        pass

    param_names = {p.name for p in cmd.params}
    assert "plain__value" in param_names
    assert "optional__value" in param_names
    assert "no_optional" in param_names
    assert "no_plain" not in param_names
    assert "plain" not in param_names
    assert "optional" not in param_names


# ---------------------------------------------------------------------------
# SafeSynthesizerParameters -- integration checks
# ---------------------------------------------------------------------------


def test_no_replace_pii_flag_on_nss_params():
    @pydantic_options(SafeSynthesizerParameters, field_separator="__")
    @click.command()
    def cmd(**kwargs):
        pass

    param_names = {p.name for p in cmd.params}
    assert "no_replace_pii" in param_names


def test_no_privacy_flag_on_nss_params():
    @pydantic_options(SafeSynthesizerParameters, field_separator="__")
    @click.command()
    def cmd(**kwargs):
        pass

    param_names = {p.name for p in cmd.params}
    assert "no_privacy" in param_names


def test_no_replace_pii_end_to_end_via_click_runner():
    """--no_replace_pii must produce replace_pii=None in parsed overrides."""
    captured: dict = {}

    @pydantic_options(SafeSynthesizerParameters, field_separator="__")
    @click.command()
    def cmd(**kwargs):
        captured.update(parse_overrides(kwargs))

    result = CliRunner().invoke(cmd, ["--no_replace_pii"])
    assert result.exit_code == 0, result.output
    assert captured.get("replace_pii") is None


def test_no_replace_pii_absent_does_not_inject():
    """Omitting --no_replace_pii must not inject replace_pii into overrides."""
    captured: dict = {}

    @pydantic_options(SafeSynthesizerParameters, field_separator="__")
    @click.command()
    def cmd(**kwargs):
        captured.update(parse_overrides(kwargs))

    result = CliRunner().invoke(cmd, [])
    assert result.exit_code == 0, result.output
    assert "replace_pii" not in captured


def test_leaf_override_end_to_end_via_click_runner():
    """A nested leaf option flows through the decorator and parse_overrides correctly."""
    captured: dict = {}

    @pydantic_options(SafeSynthesizerParameters, field_separator="__")
    @click.command()
    def cmd(**kwargs):
        captured.update(parse_overrides(kwargs))

    result = CliRunner().invoke(cmd, ["--training__batch_size", "4"])
    assert result.exit_code == 0, result.output
    assert captured["training"]["batch_size"] == 4


def test_deep_nested_override_end_to_end_via_click_runner():
    """A deeply nested option (3+ segments) flows through decorator + parse_overrides."""
    captured: dict = {}

    @pydantic_options(SafeSynthesizerParameters, field_separator="__")
    @click.command()
    def cmd(**kwargs):
        captured.update(parse_overrides(kwargs))

    result = CliRunner().invoke(cmd, ["--replace_pii__globals__seed", "42"])
    assert result.exit_code == 0, result.output
    assert captured["replace_pii"]["globals"]["seed"] == 42
