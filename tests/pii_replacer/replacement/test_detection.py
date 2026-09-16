# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from nemo_safe_synthesizer.config.replace_pii import EntityType, FreeTextDetectionConfig
from nemo_safe_synthesizer.errors import GenerationError
from nemo_safe_synthesizer.pii_replacer.replacement.detection import (
    CompositeFreeTextDetector,
    Gliner2Detector,
    RegexDetector,
    fresh_detection_entity_types,
    resolve_overlapping_spans,
)
from nemo_safe_synthesizer.pii_replacer.replacement.types import DetectedSpan, DetectionCell, DetectionCellId


def _cell(text: str = "Contact ada@example.com") -> DetectionCell:
    return DetectionCell(
        DetectionCellId(0, "notes"),
        text,
        fresh_detection_entity_types(),
    )


class _FakeModel:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.calls: list[tuple[list[str], dict[str, dict[str, float]], dict[str, object]]] = []

    def batch_extract_entities(
        self,
        texts: list[str],
        labels: dict[str, dict[str, float]],
        **kwargs: object,
    ) -> list[object]:
        self.calls.append((texts, labels, kwargs))
        return self.results


class _StaticDetector:
    def __init__(self, spans: Sequence[DetectedSpan]) -> None:
        self.spans = tuple(spans)

    def detect(self, cells: Sequence[DetectionCell]) -> tuple[DetectedSpan, ...]:
        return self.spans


@pytest.mark.unit
class TestGliner2Detector:
    def test_loader_uses_cuda_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        class FakeGliner2:
            @classmethod
            def from_pretrained(cls, model_id: str, **kwargs: object) -> _FakeModel:
                calls.append((model_id, kwargs))
                return _FakeModel([{"entities": {}}])

        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
        monkeypatch.setitem(sys.modules, "gliner2", SimpleNamespace(GLiNER2=FakeGliner2))

        Gliner2Detector(FreeTextDetectionConfig(model_id="model-id")).detect([_cell()])

        assert calls == [("model-id", {"map_location": "cuda"})]

    def test_logs_completion_without_source_text(self, caplog: pytest.LogCaptureFixture) -> None:
        text = "Contact ada@example.com"
        model = _FakeModel([{"entities": {}}])
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda _: model)

        with caplog.at_level(logging.INFO):
            detector.detect([_cell(text)])

        assert "GLiNER2 free-text PII inference complete" in caplog.text
        assert text not in caplog.text

    def test_loads_lazily_and_reuses_inference_for_identical_texts(self) -> None:
        text = "Call +1 202 555 0101"
        start = text.index("+")
        model = _FakeModel(
            [
                {
                    "entities": {
                        "phone_number": [{"text": text[start:], "start": start, "end": len(text), "confidence": 0.9}]
                    }
                }
            ]
        )
        loads: list[str] = []
        detector = Gliner2Detector(
            FreeTextDetectionConfig(),
            model_loader=lambda model_id: loads.append(model_id) or model,
        )
        cells = [_cell(text), DetectionCell(DetectionCellId(1, "other_notes"), text, _cell().allowed_entity_types)]

        assert loads == []
        spans = detector.detect(cells)

        assert len(model.calls) == 1
        assert model.calls[0][0] == [text]
        assert "government_id" in model.calls[0][1]
        assert not {
            "email",
            "payment_card",
            "card_number",
            "ip_address",
            "ssn",
            "ipv4",
            "ipv6",
        } & set(model.calls[0][1])
        assert loads == [FreeTextDetectionConfig().model_id]
        assert [(span.cell_id, span.start, span.end) for span in spans] == [
            (cells[0].cell_id, start, len(text)),
            (cells[1].cell_id, start, len(text)),
        ]

    def test_applies_the_configured_threshold_for_each_entity(self) -> None:
        texts = ["Mycobacterium marinum", "+1 202 555 0101"]
        model = _FakeModel(
            [
                {"entities": {"person": [{"start": 0, "end": len(texts[0]), "confidence": 0.89}]}},
                {"entities": {"phone_number": [{"start": 0, "end": len(texts[1]), "confidence": 0.89}]}},
            ]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda _: model)
        cells = [
            DetectionCell(DetectionCellId(position, "notes"), text, fresh_detection_entity_types())
            for position, text in enumerate(texts)
        ]

        spans = detector.detect(cells)

        assert [(span.cell_id.row_position, span.entity_type) for span in spans] == [(1, EntityType.PHONE_NUMBER)]
        labels = model.calls[0][1]
        assert labels["first_name"] == {"threshold": 0.9}
        assert labels["middle_name"] == {"threshold": 0.9}
        assert labels["last_name"] == {"threshold": 0.9}
        assert labels["person"] == {"threshold": 0.9}
        assert labels["phone_number"] == {"threshold": 0.5}

    def test_government_id_uses_the_lower_model_floor_then_the_normalized_entity_threshold(self) -> None:
        texts = ["123-45-6789", "AB-12345"]
        model = _FakeModel(
            [
                {"entities": {"government_id": [{"start": 0, "end": len(texts[0]), "confidence": 0.7}]}},
                {"entities": {"government_id": [{"start": 0, "end": len(texts[1]), "confidence": 0.7}]}},
            ]
        )
        entity_thresholds = FreeTextDetectionConfig().entity_thresholds | {
            EntityType.SSN: 0.8,
            EntityType.NATIONAL_ID: 0.6,
        }
        detector = Gliner2Detector(
            FreeTextDetectionConfig(entity_thresholds=entity_thresholds),
            model_loader=lambda _: model,
        )

        cells = [
            DetectionCell(DetectionCellId(position, "notes"), text, fresh_detection_entity_types())
            for position, text in enumerate(texts)
        ]

        spans = detector.detect(cells)

        assert [(span.cell_id.row_position, span.entity_type) for span in spans] == [
            (1, EntityType.NATIONAL_ID),
        ]
        labels = model.calls[0][1]
        assert labels["government_id"] == {"threshold": 0.6}

    def test_remaps_overlapping_chunk_offsets_to_the_complete_cell(self) -> None:
        model = _FakeModel(
            [
                {"entities": {}},
                {"entities": {"person": [{"start": 4, "end": 7, "confidence": 0.99}]}},
            ]
        )
        detector = Gliner2Detector(
            FreeTextDetectionConfig(chunk_length=10, chunk_overlap=4),
            model_loader=lambda _: model,
        )

        spans = detector.detect([_cell("0123456789Ada!xx")])

        assert [(span.start, span.end, span.entity_type) for span in spans] == [(10, 13, EntityType.FULL_NAME)]

    def test_clamps_span_over_gliner2_synthetic_terminal_period(self) -> None:
        text = "Contact Ada Lovelace"
        model = _FakeModel(
            [
                {
                    "entities": {
                        "person": [
                            {
                                "start": text.index("Ada"),
                                "end": len(text) + 1,
                                "confidence": 0.99,
                            }
                        ]
                    }
                }
            ]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda _: model)

        spans = detector.detect([_cell(text)])

        assert [(span.start, span.end, span.entity_type) for span in spans] == [
            (text.index("Ada"), len(text), EntityType.FULL_NAME)
        ]

    def test_ignores_span_containing_only_gliner2_synthetic_terminal_period(self) -> None:
        text = "No PII here"
        model = _FakeModel(
            [
                {
                    "entities": {
                        "person": [
                            {
                                "start": len(text),
                                "end": len(text) + 1,
                                "confidence": 0.8,
                            }
                        ]
                    }
                }
            ]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda _: model)

        assert detector.detect([_cell(text)]) == ()

    @pytest.mark.parametrize(
        ("text", "end"),
        [
            ("Contact Ada Lovelace!", len("Contact Ada Lovelace!") + 1),
            ("Contact Ada Lovelace", len("Contact Ada Lovelace") + 2),
        ],
    )
    def test_rejects_span_overruns_other_than_gliner2_synthetic_period(self, text: str, end: int) -> None:
        model = _FakeModel(
            [
                {
                    "entities": {
                        "person": [
                            {
                                "start": text.index("Ada"),
                                "end": end,
                                "confidence": 0.8,
                            }
                        ]
                    }
                }
            ]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda _: model)

        with pytest.raises(GenerationError, match="invalid span offsets"):
            detector.detect([_cell(text)])

    def test_rejects_unparseable_birth_date_model_spans_without_logging_the_value(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        text = "Born sometime in spring"
        model = _FakeModel(
            [{"entities": {"date_of_birth": [{"text": "spring", "start": 17, "end": 23, "confidence": 0.9}]}}]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda _: model)

        with caplog.at_level(logging.WARNING):
            assert detector.detect([_cell(text)]) == ()

        assert "birth-date candidates were ignored" in caplog.text
        assert "spring" not in caplog.text

    @pytest.mark.parametrize("value", ["5 April 1990", "April 5th, 1990", "5 avril 1990"])
    def test_accepts_complete_natural_language_birth_dates(self, value: str) -> None:
        text = f"Born on {value}"
        start = text.index(value)
        model = _FakeModel(
            [{"entities": {"date_of_birth": [{"start": start, "end": start + len(value), "confidence": 0.9}]}}]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda _: model)

        spans = detector.detect([_cell(text)])

        assert [(span.start, span.end, span.entity_type) for span in spans] == [
            (start, start + len(value), EntityType.DATE_OF_BIRTH)
        ]

    @pytest.mark.parametrize(
        ("label", "value", "expected"),
        [
            ("first_name", "Ada", EntityType.FIRST_NAME),
            ("middle_name", "Augusta", EntityType.MIDDLE_NAME),
            ("last_name", "Lovelace", EntityType.LAST_NAME),
            ("person", "Ada Lovelace", EntityType.FULL_NAME),
            ("phone_number", "+1 202 555 0101", EntityType.PHONE_NUMBER),
            ("date_of_birth", "5 April 1990", EntityType.DATE_OF_BIRTH),
            ("street_address", "12 Main Street", EntityType.STREET_ADDRESS),
            ("address", "12 Main Street", EntityType.STREET_ADDRESS),
            ("government_id", "123-45-6789", EntityType.SSN),
            ("government_id", "AB-12345", EntityType.NATIONAL_ID),
            ("national_id_number", "AB-12345", EntityType.NATIONAL_ID),
            ("api_key", "sk-ABC123", EntityType.API_KEY),
        ],
    )
    def test_normalizes_every_requested_checkpoint_label(
        self,
        label: str,
        value: str,
        expected: EntityType,
    ) -> None:
        model = _FakeModel([{"entities": {label: [{"start": 0, "end": len(value), "confidence": 0.99}]}}])
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda _: model)

        spans = detector.detect([_cell(value)])

        assert [span.entity_type for span in spans] == [expected]

    def test_normalizes_documented_government_id_label_to_ssn(self) -> None:
        text = "Identifier 123-45-6789"
        model = _FakeModel(
            [{"entities": {"government_id": [{"text": "123-45-6789", "start": 11, "end": 22, "confidence": 0.9}]}}]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda _: model)

        spans = detector.detect([_cell(text)])

        assert [(span.entity_type, span.start, span.end) for span in spans] == [(EntityType.SSN, 11, 22)]

    def test_wraps_model_load_failures_as_generation_errors(self) -> None:
        detector = Gliner2Detector(
            FreeTextDetectionConfig(),
            model_loader=lambda _: (_ for _ in ()).throw(RuntimeError("secret")),
        )

        with pytest.raises(GenerationError, match="could not be loaded"):
            detector.detect([_cell()])

    def test_wraps_inference_failures_without_exposing_model_details(self) -> None:
        class FailingModel:
            def batch_extract_entities(
                self,
                texts: list[str],
                labels: dict[str, dict[str, float]],
                **kwargs: object,
            ) -> list[object]:
                raise RuntimeError("raw secret input")

        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda _: FailingModel())

        with pytest.raises(GenerationError, match="inference failed") as exc_info:
            detector.detect([_cell()])

        assert "raw secret input" not in str(exc_info.value)

    def test_rejects_invalid_batch_result_shape(self) -> None:
        detector = Gliner2Detector(
            FreeTextDetectionConfig(),
            model_loader=lambda _: _FakeModel([]),
        )

        with pytest.raises(GenerationError, match="invalid batch result"):
            detector.detect([_cell()])

    def test_forwards_batch_size_for_multiple_unique_texts(self) -> None:
        texts = ["First value", "Second value", "Third value"]
        model = _FakeModel([{"entities": {}} for _ in texts])
        detector = Gliner2Detector(
            FreeTextDetectionConfig(batch_size=2),
            model_loader=lambda _: model,
        )

        assert detector.detect([_cell(text) for text in texts]) == ()
        assert model.calls[0][0] == texts
        assert model.calls[0][2]["batch_size"] == 2


@pytest.mark.unit
class TestRegexDetector:
    def test_detects_only_supported_structurally_valid_values(self) -> None:
        text = "Email ada@example.com, card 4111 1111 1111 1111, IP 192.168.1.1, invalid 999.2.3.4."

        spans = RegexDetector().detect([_cell(text)])

        assert [(text[span.start : span.end], span.entity_type) for span in spans] == [
            ("ada@example.com", EntityType.EMAIL),
            ("4111 1111 1111 1111", EntityType.CREDIT_DEBIT_CARD),
            ("192.168.1.1", EntityType.IPV4),
        ]

    def test_enforces_per_rule_match_limit(self) -> None:
        text = " ".join(f"person{index}@example.com" for index in range(5))

        with pytest.raises(GenerationError, match="exceeded the maximum of 2 matches") as exc_info:
            RegexDetector(max_matches_per_rule=2).detect([_cell(text)])
        assert "person0@example.com" not in str(exc_info.value)

    def test_detects_unicode_email_and_rejects_invalid_local_part(self) -> None:
        text = "Valid élise@exämple.fr and 用户@example.公司, invalid ada..lovelace@example.com."

        spans = RegexDetector().detect([_cell(text)])

        assert [(text[span.start : span.end], span.entity_type) for span in spans] == [
            ("élise@exämple.fr", EntityType.EMAIL),
            ("用户@example.公司", EntityType.EMAIL),
        ]

    def test_rejects_overlong_email_local_part(self) -> None:
        text = f"{'a' * 65}@example.com"

        assert RegexDetector().detect([_cell(text)]) == ()

    def test_rejects_repeated_digit_and_non_ascii_card_candidates(self) -> None:
        text = "Invalid cards 4111 1111 1111 1112, 0000 0000 0000 0000, and ٤١١١ ١١١١ ١١١١ ١١١١."

        assert RegexDetector().detect([_cell(text)]) == ()

    def test_ipv4_can_be_followed_by_sentence_punctuation(self) -> None:
        text = "Connect to 192.168.1.1. Then continue."

        spans = RegexDetector().detect([_cell(text)])

        assert [(text[span.start : span.end], span.entity_type) for span in spans] == [("192.168.1.1", EntityType.IPV4)]

    def test_wraps_regex_timeouts_as_generation_errors(self) -> None:
        text = "person@example.com " * 100

        with pytest.raises(GenerationError, match="timed out"):
            RegexDetector(timeout_seconds=1e-12).detect([_cell(text)])

    def test_detects_valid_ipv6(self) -> None:
        text = "Connect to 2001:db8::1 now"

        spans = RegexDetector().detect([_cell(text)])

        assert [(text[span.start : span.end], span.entity_type) for span in spans] == [("2001:db8::1", EntityType.IPV6)]


@pytest.mark.unit
class TestOverlapResolution:
    def test_longer_spans_win_and_touching_spans_survive(self) -> None:
        cell_id = DetectionCellId(0, "notes")
        spans = [
            DetectedSpan(cell_id, 0, 10, EntityType.FULL_NAME, "gliner", 0.6),
            DetectedSpan(cell_id, 0, 4, EntityType.FIRST_NAME, "gliner", 0.9),
            DetectedSpan(cell_id, 10, 15, EntityType.EMAIL, "regex"),
        ]

        assert resolve_overlapping_spans(spans) == (spans[0], spans[2])

    def test_regex_wins_an_exact_cross_source_tie_and_gliner_confidence_breaks_its_ties(self) -> None:
        cell_id = DetectionCellId(0, "notes")
        low = DetectedSpan(cell_id, 0, 4, EntityType.EMAIL, "gliner", 0.5)
        high = DetectedSpan(cell_id, 0, 4, EntityType.EMAIL, "gliner", 0.9)
        regex_span = DetectedSpan(cell_id, 0, 4, EntityType.EMAIL, "regex")

        assert resolve_overlapping_spans([low, high]) == (high,)
        assert resolve_overlapping_spans([high, regex_span]) == (regex_span,)

    def test_composite_resolves_cross_source_candidates(self) -> None:
        cell = _cell("ada@example.com")
        gliner_span = DetectedSpan(cell.cell_id, 0, 3, EntityType.FULL_NAME, "gliner", 0.9)
        regex_span = DetectedSpan(cell.cell_id, 0, 15, EntityType.EMAIL, "regex")
        detector = CompositeFreeTextDetector(
            FreeTextDetectionConfig(),
            gliner=_StaticDetector([gliner_span]),
            regex_detector=_StaticDetector([regex_span]),
        )

        assert detector.detect([cell]) == (regex_span,)

    def test_nested_duplicates_have_stable_order_independent_of_input_order(self) -> None:
        first_cell = DetectionCellId(0, "notes")
        second_cell = DetectionCellId(1, "notes")
        preferred = DetectedSpan(first_cell, 2, 12, EntityType.FULL_NAME, "gliner", 0.9)
        duplicate = DetectedSpan(first_cell, 2, 12, EntityType.FULL_NAME, "gliner", 0.5)
        nested = DetectedSpan(first_cell, 4, 8, EntityType.LAST_NAME, "gliner", 0.99)
        later_cell = DetectedSpan(second_cell, 0, 3, EntityType.FIRST_NAME, "gliner", 0.8)

        expected = (preferred, later_cell)
        assert resolve_overlapping_spans([nested, duplicate, later_cell, preferred]) == expected
        assert resolve_overlapping_spans([preferred, later_cell, duplicate, nested]) == expected
