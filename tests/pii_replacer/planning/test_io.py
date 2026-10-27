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
    PiiSamplerBackend,
    PiiSamplerConfig,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer.planning import save_config, save_plan
from nemo_safe_synthesizer.pii_replacer.planning.review import DependencyValueReview, PlanReviewNotes


def _reloaded(path: Path) -> PiiReplacementPlan:
    plan = ReplacePiiConfig.model_validate({"replacement_plan": str(path)}).inline_plan
    assert plan is not None
    return plan


@pytest.mark.unit
class TestPlanFile:
    def test_save_then_load_round_trip_with_a_sparse_plan(self, tmp_path: Path) -> None:
        plan = PiiReplacementPlan(
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
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

        path = save_plan(plan, tmp_path / "nested" / "pii_replacement_plan.yaml")

        assert path == tmp_path / "nested" / "pii_replacement_plan.yaml"
        assert yaml.safe_load(path.read_text()) == {
            "sampler": {"backend": "faker", "nemotron_personas_path": None},
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
        }
        reloaded = _reloaded(path)
        assert reloaded == plan
        assert save_plan(reloaded, tmp_path / "second.yaml").read_text() == path.read_text()

    def test_load_rejects_non_mapping_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "pii_replacement_plan.yaml"
        path.write_text("- not\n- a\n- mapping\n")

        with pytest.raises(ValidationError, match="must contain a mapping"):
            _reloaded(path)

    def test_load_rejects_unknown_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "pii_replacement_plan.yaml"
        path.write_text(yaml.safe_dump({"scope": "group"}))

        with pytest.raises(ValidationError, match="Unknown configuration field 'scope'"):
            _reloaded(path)

    def test_relative_personas_path_resolves_against_the_plan_file(self, tmp_path: Path) -> None:
        path = tmp_path / "plans" / "pii_replacement_plan.yaml"
        path.parent.mkdir()
        path.write_text(yaml.safe_dump({"sampler": {"nemotron_personas_path": "personas"}}))

        sampler = _reloaded(path).sampler

        assert sampler.resolved_nemotron_personas_path() == tmp_path / "plans" / "personas"

    def test_relative_personas_path_survives_saving_the_plan_elsewhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "config.yaml").write_text(
            "replace_pii:\n  replacement_plan:\n    sampler:\n      nemotron_personas_path: personas\n"
        )
        monkeypatch.chdir(tmp_path)
        config = SafeSynthesizerParameters.from_yaml(Path("configs") / "config.yaml")
        run_dir = tmp_path / "runs" / "example"

        config_path = save_config(config, run_dir / "config.yaml", plan_path=run_dir / "pii_replacement_plan.yaml")

        reloaded = SafeSynthesizerParameters.from_yaml(config_path).replace_pii
        assert reloaded is not None
        assert reloaded.inline_plan is not None
        assert reloaded.inline_plan.sampler.resolved_nemotron_personas_path() == tmp_path / "configs" / "personas"

    def test_saved_config_names_the_plan_file_beside_it(self, tmp_path: Path) -> None:
        plan = PiiReplacementPlan(columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)])
        config = SafeSynthesizerParameters(replace_pii=ReplacePiiConfig(replacement_plan=plan))
        run_dir = tmp_path / "run"

        config_path = save_config(config, run_dir / "config.yaml", plan_path=run_dir / "pii_replacement_plan.yaml")

        assert yaml.safe_load(config_path.read_text())["replace_pii"]["replacement_plan"] == "pii_replacement_plan.yaml"
        assert SafeSynthesizerParameters.from_yaml(config_path).replace_pii == config.replace_pii

    def test_saved_config_names_a_plan_file_elsewhere_by_its_absolute_path(self, tmp_path: Path) -> None:
        plan = PiiReplacementPlan(columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)])
        config = SafeSynthesizerParameters(replace_pii=ReplacePiiConfig(replacement_plan=plan))
        plan_path = tmp_path / "plans" / "pii_replacement_plan.yaml"

        config_path = save_config(config, tmp_path / "run" / "config.yaml", plan_path=plan_path)

        assert yaml.safe_load(config_path.read_text())["replace_pii"]["replacement_plan"] == plan_path.as_posix()
        assert SafeSynthesizerParameters.from_yaml(config_path).replace_pii == config.replace_pii

    def test_auto_discovery_config_keeps_the_sentinel_inline(self, tmp_path: Path) -> None:
        config = SafeSynthesizerParameters(replace_pii=ReplacePiiConfig())

        config_path = save_config(config, tmp_path / "config.yaml", plan_path=tmp_path / "pii_replacement_plan.yaml")

        assert yaml.safe_load(config_path.read_text())["replace_pii"]["replacement_plan"] == "auto_discovery"
        assert not (tmp_path / "pii_replacement_plan.yaml").exists()


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
        plan = _commented_plan()

        path = save_plan(plan, tmp_path / "pii_replacement_plan.yaml")
        text = path.read_text()

        assert text.startswith("# PII replacement plan for this dataset")
        assert _comment_above(text, "sampler:") == ""
        assert _comment_above(text, "columns_to_replace:") == (
            PiiReplacementPlan.model_fields["columns_to_replace"].description
        )
        mapping_comment = _comment_above(text, "data_to_sampler_value_mapping:")
        assert mapping_comment.startswith("Maps dataset values in depends_on columns")
        assert mapping_comment.endswith("You can usually leave this section as is.")
        assert _reloaded(path) == plan

    def test_saved_inline_config_comments_only_plan_sections(self, tmp_path: Path) -> None:
        plan = _commented_plan()
        config = SafeSynthesizerParameters(replace_pii=ReplacePiiConfig(replacement_plan=plan))

        text = save_config(config, tmp_path / "config.yaml").read_text()

        assert _comment_above(text, "    columns_to_replace:") == (
            PiiReplacementPlan.model_fields["columns_to_replace"].description
        )
        assert _comment_above(text, "    data_to_sampler_value_mapping:").startswith("Maps dataset values")
        uncommented = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        assert uncommented + "\n" == yaml.safe_dump(yaml.safe_load(text))
        reloaded = SafeSynthesizerParameters.model_validate(yaml.safe_load(text))
        assert reloaded.replace_pii is not None
        assert reloaded.replace_pii.inline_plan == plan

    def test_review_notes_follow_the_field_descriptions(self, tmp_path: Path) -> None:
        plan = _commented_plan()
        config = SafeSynthesizerParameters(replace_pii=ReplacePiiConfig(replacement_plan=plan))
        notes = PlanReviewNotes(
            unchanged_columns=("sex", "admission_id"),
            dependency_values=(
                DependencyValueReview("sex", EntityType.GENDER, ("Female", "Male"), ("female", "male")),
            ),
        )

        text = save_config(config, tmp_path / "config.yaml", review_notes=notes).read_text()

        assert _comment_above(text, "    columns_to_replace:").endswith(
            "Draft review: not in this plan, so kept unchanged: sex, admission_id. Review notes describe the plan "
            "as drafted and are not updated when you edit it."
        )
        assert _comment_above(text, "    data_to_sampler_value_mapping:").endswith(
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

        text = save_plan(_commented_plan(), tmp_path / "pii_replacement_plan.yaml", review_notes=notes).read_text()

        assert "Draft review: every dataset column is in this plan." in _comment_above(text, "columns_to_replace:")
        mapping_comment = _comment_above(text, "data_to_sampler_value_mapping:")
        assert "group 19, ... (30 in total) are used as is" in mapping_comment
        assert "group 20" not in mapping_comment
