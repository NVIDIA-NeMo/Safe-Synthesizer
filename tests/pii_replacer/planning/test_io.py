# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.replace_pii import (
    ConditioningColumn,
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiReplacementSettings,
    PiiSamplerBackend,
    PiiSamplerConfig,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer.planning import save_config, save_replacement
from nemo_safe_synthesizer.pii_replacer.planning.review import DependencyValueReview, PlanReviewNotes


def _reloaded(path: Path) -> PiiReplacementSettings:
    return ReplacePiiConfig.model_validate({"replacement": str(path)}).replacement


@pytest.mark.unit
class TestReplacementFile:
    def test_save_then_load_round_trip_with_a_sparse_plan(self, tmp_path: Path) -> None:
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(column_name="first", entity_type=EntityType.FIRST_NAME),
                PiiColumnPlan(
                    column_name="email",
                    entity_type=EntityType.EMAIL,
                    depends_on=[
                        ConditioningColumn(column_name="first"),
                        ConditioningColumn(column_name="company", entity_type=EntityType.ORGANIZATION),
                    ],
                ),
            ],
            data_to_sampler_value_mapping={"company": {"Independent": ["independent"]}},
        )
        settings = PiiReplacementSettings(
            locale="en_GB", seed=7, sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER), plan=plan
        )

        path = save_replacement(settings, tmp_path / "nested" / "pii_replacement.yaml")

        assert path == tmp_path / "nested" / "pii_replacement.yaml"
        assert yaml.safe_load(path.read_text()) == {
            "locale": "en_GB",
            "seed": 7,
            "sampler": {"backend": "faker", "nemotron_personas_path": None},
            "plan": {
                "columns_to_replace": [
                    {"column_name": "first", "entity_type": "first_name"},
                    {
                        "column_name": "email",
                        "entity_type": "email",
                        "depends_on": [
                            {"column_name": "first", "entity_type": "first_name"},
                            {"column_name": "company", "entity_type": "organization"},
                        ],
                    },
                ],
                "data_to_sampler_value_mapping": {"company": {"Independent": ["independent"]}},
            },
        }
        reloaded = _reloaded(path)
        assert reloaded == settings
        assert save_replacement(reloaded, tmp_path / "second.yaml").read_text() == path.read_text()

    def test_auto_discovery_is_written_as_the_sentinel(self, tmp_path: Path) -> None:
        path = save_replacement(PiiReplacementSettings(), tmp_path / "pii_replacement.yaml")

        assert yaml.safe_load(path.read_text())["plan"] == "auto_discovery"
        assert _reloaded(path) == PiiReplacementSettings()

    def test_load_rejects_non_mapping_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "pii_replacement.yaml"
        path.write_text("- not\n- a\n- mapping\n")

        with pytest.raises(ValidationError, match="must contain a mapping"):
            _reloaded(path)

    def test_load_rejects_unknown_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "pii_replacement.yaml"
        path.write_text(yaml.safe_dump({"scope": "group"}))

        with pytest.raises(ValidationError, match="Unknown configuration field 'scope'"):
            _reloaded(path)

    def test_saved_config_names_the_replacement_file_beside_it(self, tmp_path: Path) -> None:
        plan = PiiReplacementPlan(columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)])
        config = SafeSynthesizerParameters(replace_pii=ReplacePiiConfig(replacement=PiiReplacementSettings(plan=plan)))
        run_dir = tmp_path / "run"

        config_path = save_config(config, run_dir / "config.yaml", replacement_path=run_dir / "pii_replacement.yaml")

        assert yaml.safe_load(config_path.read_text())["replace_pii"]["replacement"] == "pii_replacement.yaml"
        assert SafeSynthesizerParameters.from_yaml(config_path).replace_pii == config.replace_pii


def _commented_plan() -> PiiReplacementPlan:
    return PiiReplacementPlan(
        columns_to_replace=[
            PiiColumnPlan(
                column_name="first",
                entity_type=EntityType.FIRST_NAME,
                depends_on=[ConditioningColumn(column_name="sex", entity_type=EntityType.GENDER)],
            )
        ],
        data_to_sampler_value_mapping={"sex": {"Woman": ["female"]}},
    )


def _comment_above(text: str, key_line: str) -> str:
    lines = text.splitlines()
    index = lines.index(key_line)
    comment: list[str] = []
    for line in reversed(lines[:index]):
        if not line.lstrip().startswith("#"):
            break
        comment.insert(0, line.lstrip()[2:])
    return " ".join(comment)


@pytest.mark.unit
class TestCommentedYaml:
    def test_saved_replacement_comments_each_plan_section_with_its_description(self, tmp_path: Path) -> None:
        settings = PiiReplacementSettings(plan=_commented_plan())

        path = save_replacement(settings, tmp_path / "pii_replacement.yaml")
        text = path.read_text()

        assert text.startswith("# PII replacement for this dataset")
        assert _comment_above(text, "  columns_to_replace:") == (
            PiiReplacementPlan.model_fields["columns_to_replace"].description
        )
        mapping_comment = _comment_above(text, "  data_to_sampler_value_mapping:")
        assert mapping_comment.startswith("Maps dataset values in depends_on columns")
        assert mapping_comment.endswith("You can usually leave this section as is.")
        assert _reloaded(path) == settings

    def test_saved_inline_config_comments_only_plan_sections(self, tmp_path: Path) -> None:
        plan = _commented_plan()
        config = SafeSynthesizerParameters(replace_pii=ReplacePiiConfig(replacement=PiiReplacementSettings(plan=plan)))

        text = save_config(config, tmp_path / "config.yaml").read_text()

        assert _comment_above(text, "      columns_to_replace:") == (
            PiiReplacementPlan.model_fields["columns_to_replace"].description
        )
        assert _comment_above(text, "      data_to_sampler_value_mapping:").startswith("Maps dataset values")
        uncommented = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        assert uncommented + "\n" == yaml.safe_dump(yaml.safe_load(text))
        reloaded = SafeSynthesizerParameters.model_validate(yaml.safe_load(text))
        assert reloaded.replace_pii is not None
        assert reloaded.replace_pii.inline_plan == plan

    def test_review_notes_follow_the_field_descriptions(self, tmp_path: Path) -> None:
        plan = _commented_plan()
        config = SafeSynthesizerParameters(replace_pii=ReplacePiiConfig(replacement=PiiReplacementSettings(plan=plan)))
        notes = PlanReviewNotes(
            unchanged_columns=("sex", "admission_id"),
            dependency_values=(
                DependencyValueReview("sex", EntityType.GENDER, ("Female", "Male"), ("female", "male")),
            ),
        )

        text = save_config(config, tmp_path / "config.yaml", review_notes=notes).read_text()

        assert _comment_above(text, "      columns_to_replace:").endswith(
            "Draft review: not in this plan, so kept unchanged: sex, admission_id. Review notes describe the plan "
            "as drafted and are not updated when you edit it."
        )
        assert _comment_above(text, "      data_to_sampler_value_mapping:").endswith(
            "Draft review: sex (gender): Female, Male are used as is. Sampler values: female, male."
        )
        reloaded = SafeSynthesizerParameters.model_validate(yaml.safe_load(text))
        assert reloaded.replace_pii is not None
        assert reloaded.replace_pii.inline_plan == plan

    def test_review_notes_cap_long_value_lists(self, tmp_path: Path) -> None:
        values = tuple(f"group {index}" for index in range(30))
        notes = PlanReviewNotes(
            unchanged_columns=(),
            dependency_values=(DependencyValueReview("race", EntityType.ETHNIC_BACKGROUND, values, values),),
        )

        settings = PiiReplacementSettings(plan=_commented_plan())
        text = save_replacement(settings, tmp_path / "pii_replacement.yaml", review_notes=notes).read_text()

        assert "Draft review: every dataset column is in this plan." in _comment_above(text, "  columns_to_replace:")
        mapping_comment = _comment_above(text, "  data_to_sampler_value_mapping:")
        assert "group 19, ... (30 in total) are used as is" in mapping_comment
        assert "group 20" not in mapping_comment
