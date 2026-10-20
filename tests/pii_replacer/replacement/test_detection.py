# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from nemo_safe_synthesizer.config.replace_pii import (
    DEFAULT_GLINER2_MODEL_ID,
    DEFAULT_GLINER2_MODEL_REVISION,
    EntityType,
    FreeTextDetectionConfig,
)
from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.pii_replacer.replacement.detection import (
    CompositeFreeTextDetector,
    Gliner2Detector,
    RegexDetector,
    fresh_detection_entity_types,
    resolve_overlapping_spans,
)
from nemo_safe_synthesizer.pii_replacer.replacement.types import (
    DetectedSpan,
    DetectionCell,
    DetectionCellId,
    FreeTextDetections,
    ProtectedSpan,
)


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

    def detect(self, cells: Sequence[DetectionCell]) -> FreeTextDetections:
        return FreeTextDetections(self.spans)


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

        Gliner2Detector(FreeTextDetectionConfig()).detect([_cell()]).spans

        assert calls == [
            (DEFAULT_GLINER2_MODEL_ID, {"revision": DEFAULT_GLINER2_MODEL_REVISION, "map_location": "cuda"})
        ]

    def test_logs_completion_without_source_text(self, caplog: pytest.LogCaptureFixture) -> None:
        text = "Contact ada@example.com"
        model = _FakeModel([{"entities": {}}])
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)

        with caplog.at_level(logging.INFO):
            detector.detect([_cell(text)]).spans

        assert "GLiNER2 free-text PII inference complete" in caplog.text
        assert text not in caplog.text

    def test_loads_lazily_and_reuses_inference_for_identical_texts(self) -> None:
        text = "Call +1 202 555 0101"
        start = text.index("+")
        model = _FakeModel(
            [
                {
                    "entities": {
                        "phone_number": [{"text": text[start:], "start": start, "end": len(text), "confidence": 0.999}]
                    }
                }
            ]
        )
        loads: list[tuple[str, str | None]] = []
        detector = Gliner2Detector(
            FreeTextDetectionConfig(),
            model_loader=lambda model_id, revision: loads.append((model_id, revision)) or model,
        )
        cells = [_cell(text), DetectionCell(DetectionCellId(1, "other_notes"), text, _cell().allowed_entity_types)]

        assert loads == []
        spans = detector.detect(cells).spans

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
        assert loads == [(DEFAULT_GLINER2_MODEL_ID, DEFAULT_GLINER2_MODEL_REVISION)]
        assert [(span.cell_id, span.start, span.end) for span in spans] == [
            (cells[0].cell_id, start, len(text)),
            (cells[1].cell_id, start, len(text)),
        ]

    def test_applies_the_configured_threshold_for_each_entity(self) -> None:
        texts = ["Ada Lovelace", "Mycobacterium marinum", "+1 202 555 0101"]
        model = _FakeModel(
            [
                {"entities": {"full_name": [{"start": 0, "end": len(texts[0]), "confidence": 0.99}]}},
                {"entities": {"person": [{"start": 0, "end": len(texts[1]), "confidence": 0.99}]}},
                {"entities": {"phone_number": [{"start": 0, "end": len(texts[2]), "confidence": 0.995}]}},
            ]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)
        cells = [
            DetectionCell(DetectionCellId(position, "notes"), text, fresh_detection_entity_types())
            for position, text in enumerate(texts)
        ]

        spans = detector.detect(cells).spans

        assert [(span.cell_id.row_position, span.entity_type) for span in spans] == [
            (0, EntityType.FULL_NAME),
            (2, EntityType.PHONE_NUMBER),
        ]
        labels = model.calls[0][1]
        assert labels["full_name"] == {"threshold": 0.964}
        assert labels["first_name"] == {"threshold": 0.946}
        assert labels["middle_name"] == {"threshold": 0.7}
        assert labels["last_name"] == {"threshold": 0.618}
        assert "person" not in labels
        assert "address" not in labels
        assert labels["phone_number"] == {"threshold": 0.990}

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
            model_loader=lambda *_: model,
        )

        cells = [
            DetectionCell(DetectionCellId(position, "notes"), text, fresh_detection_entity_types())
            for position, text in enumerate(texts)
        ]

        spans = detector.detect(cells).spans

        assert [(span.cell_id.row_position, span.entity_type) for span in spans] == [
            (1, EntityType.NATIONAL_ID),
        ]
        labels = model.calls[0][1]
        assert labels["government_id"] == {"threshold": 0.6}

    def test_remaps_overlapping_chunk_offsets_to_the_complete_cell(self) -> None:
        model = _FakeModel(
            [
                {"entities": {}},
                {"entities": {"first_name": [{"start": 4, "end": 7, "confidence": 0.99}]}},
            ]
        )
        detector = Gliner2Detector(
            FreeTextDetectionConfig(chunk_length=10, chunk_overlap=4),
            model_loader=lambda *_: model,
        )

        spans = detector.detect([_cell("0123456789Ada!xx")]).spans

        assert [(span.start, span.end, span.entity_type) for span in spans] == [(10, 13, EntityType.FIRST_NAME)]

    def test_clamps_span_over_gliner2_synthetic_terminal_period(self) -> None:
        text = "Contact Ada Lovelace"
        model = _FakeModel(
            [
                {
                    "entities": {
                        "first_name": [
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
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)

        spans = detector.detect([_cell(text)]).spans

        assert [(span.start, span.end, span.entity_type) for span in spans] == [
            (text.index("Ada"), len(text), EntityType.FIRST_NAME)
        ]

    def test_ignores_span_containing_only_gliner2_synthetic_terminal_period(self) -> None:
        text = "No PII here"
        model = _FakeModel(
            [
                {
                    "entities": {
                        "first_name": [
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
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)

        assert detector.detect([_cell(text)]).spans == ()

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
                        "first_name": [
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
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)

        with pytest.raises(GenerationError, match="invalid span offsets"):
            detector.detect([_cell(text)]).spans

    def test_rejects_unparseable_birth_date_model_spans_without_logging_the_value(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        text = "Born sometime in spring"
        model = _FakeModel(
            [{"entities": {"date_of_birth": [{"text": "spring", "start": 17, "end": 23, "confidence": 1.0}]}}]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)

        with caplog.at_level(logging.WARNING):
            assert detector.detect([_cell(text)]).spans == ()

        assert "birth-date candidates were ignored" in caplog.text
        assert "spring" not in caplog.text

    @pytest.mark.parametrize("value", ["5 April 1990", "April 5th, 1990", "5 avril 1990"])
    def test_accepts_complete_natural_language_birth_dates(self, value: str) -> None:
        text = f"Born on {value}"
        start = text.index(value)
        model = _FakeModel(
            [{"entities": {"date_of_birth": [{"start": start, "end": start + len(value), "confidence": 1.0}]}}]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)

        spans = detector.detect([_cell(text)]).spans

        assert [(span.start, span.end, span.entity_type) for span in spans] == [
            (start, start + len(value), EntityType.DATE_OF_BIRTH)
        ]

    @pytest.mark.parametrize(
        ("label", "value", "expected"),
        [
            ("first_name", "Ada", EntityType.FIRST_NAME),
            ("full_name", "Ada Lovelace", EntityType.FULL_NAME),
            ("last_name", "Lovelace", EntityType.LAST_NAME),
            ("phone_number", "+1 202 555 0101", EntityType.PHONE_NUMBER),
            ("fax_number", "+1 202 555 0102", EntityType.PHONE_NUMBER),
            ("date_of_birth", "5 April 1990", EntityType.DATE_OF_BIRTH),
            ("street_address", "12 Main Street", EntityType.STREET_ADDRESS),
            ("government_id", "123-45-6789", EntityType.SSN),
            ("government_id", "AB-12345", EntityType.NATIONAL_ID),
            ("government_id", "912-45-6789", EntityType.NATIONAL_ID),
            ("government_id", "666-45-6789", EntityType.NATIONAL_ID),
            ("government_id", "123-00-6789", EntityType.NATIONAL_ID),
            ("government_id", "123-45-0000", EntityType.NATIONAL_ID),
            ("national_id_number", "AB-12345", EntityType.NATIONAL_ID),
            ("passport_number", "X1234567", EntityType.NATIONAL_ID),
            ("tax_id", "12-3456789", EntityType.NATIONAL_ID),
            ("tax_number", "12-3456789", EntityType.NATIONAL_ID),
            ("api_key", "sk-ABC123", EntityType.API_KEY),
            ("customer_id", "CUST-00412", EntityType.UNIQUE_IDENTIFIER),
            ("employee_id", "E-7731", EntityType.UNIQUE_IDENTIFIER),
            ("medical_record_number", "MRN 448102", EntityType.UNIQUE_IDENTIFIER),
            ("member_id", "M90311", EntityType.UNIQUE_IDENTIFIER),
            ("health_plan_beneficiary_number", "HPB-55102", EntityType.UNIQUE_IDENTIFIER),
            ("certificate_number", "RN-448102", EntityType.UNIQUE_IDENTIFIER),
        ],
    )
    def test_normalizes_every_requested_checkpoint_label(
        self,
        label: str,
        value: str,
        expected: EntityType,
    ) -> None:
        model = _FakeModel([{"entities": {label: [{"start": 0, "end": len(value), "confidence": 1.0}]}}])
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)

        spans = detector.detect([_cell(value)]).spans

        assert [span.entity_type for span in spans] == [expected]
        assert label in model.calls[0][1]

    def test_drops_name_spans_overlapping_an_exclusion_label_and_protects_the_exclusion(self) -> None:
        text = "Mother reports Raynaud symptoms; Ada called."
        mother = (0, len("Mother"))
        raynaud = (text.index("Raynaud"), text.index("Raynaud") + len("Raynaud"))
        ada = (text.index("Ada"), text.index("Ada") + len("Ada"))
        model = _FakeModel(
            [
                {
                    "entities": {
                        "first_name": [
                            {"start": mother[0], "end": mother[1], "confidence": 0.99},
                            {"start": ada[0], "end": ada[1], "confidence": 0.99},
                        ],
                        "last_name": [{"start": raynaud[0], "end": raynaud[1], "confidence": 0.99}],
                        "family_relationship": [{"start": mother[0], "end": mother[1], "confidence": 0.5}],
                        "medical_condition": [
                            {"start": raynaud[0], "end": text.index(";"), "confidence": 0.8},
                        ],
                    }
                }
            ]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)

        cell = _cell(text)
        detections = detector.detect([cell])

        assert [(span.start, span.end, span.entity_type) for span in detections.spans] == [
            (*ada, EntityType.FIRST_NAME)
        ]
        assert detections.protected == (
            ProtectedSpan(cell.cell_id, *mother),
            ProtectedSpan(cell.cell_id, raynaud[0], text.index(";")),
        )
        assert model.calls[0][1]["family_relationship"] == {"threshold": 0.062}

    def test_exclusion_below_its_threshold_keeps_the_name(self) -> None:
        text = "Mother"
        model = _FakeModel(
            [
                {
                    "entities": {
                        "first_name": [{"start": 0, "end": len(text), "confidence": 0.99}],
                        "family_relationship": [{"start": 0, "end": len(text), "confidence": 0.05}],
                    }
                }
            ]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)
        detections = detector.detect([_cell(text)])

        assert [span.entity_type for span in detections.spans] == [EntityType.FIRST_NAME]
        assert detections.protected == ()

    def test_exclusion_labels_do_not_drop_non_name_spans(self) -> None:
        text = "+1 202 555 0101"
        model = _FakeModel(
            [
                {
                    "entities": {
                        "phone_number": [{"start": 0, "end": len(text), "confidence": 1.0}],
                        "medical_condition": [{"start": 0, "end": len(text), "confidence": 0.9}],
                    }
                }
            ]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)

        assert [span.entity_type for span in detector.detect([_cell(text)]).spans] == [EntityType.PHONE_NUMBER]

    def test_empty_exclusion_mapping_requests_no_exclusion_labels(self) -> None:
        text = "Mother"
        model = _FakeModel([{"entities": {"first_name": [{"start": 0, "end": len(text), "confidence": 0.99}]}}])
        detector = Gliner2Detector(
            FreeTextDetectionConfig(name_exclusion_thresholds={}),
            model_loader=lambda *_: model,
        )

        assert [span.entity_type for span in detector.detect([_cell(text)]).spans] == [EntityType.FIRST_NAME]
        assert not set(FreeTextDetectionConfig().name_exclusion_thresholds) & set(model.calls[0][1])

    def test_requests_exclusion_labels_only_when_a_name_type_is_allowed(self) -> None:
        model = _FakeModel([{"entities": {}}])
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)
        cell = DetectionCell(DetectionCellId(0, "notes"), "Call me", frozenset({EntityType.PHONE_NUMBER}))

        detector.detect([cell]).spans

        assert set(model.calls[0][1]) == {"fax_number", "phone_number"}

    def test_rejects_exclusion_label_that_is_a_detection_label(self) -> None:
        config = FreeTextDetectionConfig(name_exclusion_thresholds={"Government ID": 0.5})

        with pytest.raises(ParameterError, match=r"\['government_id'\]"):
            Gliner2Detector(config)

    @pytest.mark.parametrize(
        ("text", "pieces", "kept"),
        [
            ("between a small and medium", {"middle_name": "medium"}, []),
            (
                "Ada Augusta Byron",
                {"first_name": "Ada", "middle_name": "Augusta", "last_name": "Byron"},
                [EntityType.FIRST_NAME, EntityType.MIDDLE_NAME, EntityType.LAST_NAME],
            ),
            (
                "Ada Augusta Lovelace",
                {"full_name": "Ada Augusta Lovelace", "middle_name": "Augusta"},
                [EntityType.FULL_NAME, EntityType.MIDDLE_NAME],
            ),
            ("Ada Augusta", {"first_name": "Ada", "middle_name": "Augusta"}, [EntityType.FIRST_NAME]),
        ],
    )
    def test_keeps_middle_names_only_beside_other_name_pieces(
        self,
        text: str,
        pieces: dict[str, str],
        kept: list[EntityType],
    ) -> None:
        entities = {
            label: [{"start": text.index(value), "end": text.index(value) + len(value), "confidence": 0.99}]
            for label, value in pieces.items()
        }
        model = _FakeModel([{"entities": entities}])
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)

        assert sorted(span.entity_type for span in detector.detect([_cell(text)]).spans) == sorted(kept)

    def test_middle_name_rule_can_be_disabled(self) -> None:
        text = "between a small and medium"
        start = text.index("medium")
        model = _FakeModel(
            [{"entities": {"middle_name": [{"start": start, "end": start + len("medium"), "confidence": 0.99}]}}]
        )
        detector = Gliner2Detector(
            FreeTextDetectionConfig(require_middle_name_neighbors=False),
            model_loader=lambda *_: model,
        )

        assert [span.entity_type for span in detector.detect([_cell(text)]).spans] == [EntityType.MIDDLE_NAME]

    def test_drops_identifier_predictions_without_a_digit(self) -> None:
        text = "Customer Loyalty"
        model = _FakeModel([{"entities": {"customer_id": [{"start": 0, "end": len(text), "confidence": 1.0}]}}])
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)

        assert detector.detect([_cell(text)]).spans == ()

    def test_normalizes_documented_government_id_label_to_ssn(self) -> None:
        text = "Identifier 123-45-6789"
        model = _FakeModel(
            [{"entities": {"government_id": [{"text": "123-45-6789", "start": 11, "end": 22, "confidence": 0.9}]}}]
        )
        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: model)

        spans = detector.detect([_cell(text)]).spans

        assert [(span.entity_type, span.start, span.end) for span in spans] == [(EntityType.SSN, 11, 22)]

    def test_wraps_model_load_failures_as_generation_errors(self) -> None:
        detector = Gliner2Detector(
            FreeTextDetectionConfig(),
            model_loader=lambda *_: (_ for _ in ()).throw(RuntimeError("secret")),
        )

        with pytest.raises(GenerationError, match="could not be loaded"):
            detector.detect([_cell()]).spans

    def test_wraps_inference_failures_without_exposing_model_details(self) -> None:
        class FailingModel:
            def batch_extract_entities(
                self,
                texts: list[str],
                labels: dict[str, dict[str, float]],
                **kwargs: object,
            ) -> list[object]:
                raise RuntimeError("raw secret input")

        detector = Gliner2Detector(FreeTextDetectionConfig(), model_loader=lambda *_: FailingModel())

        with pytest.raises(GenerationError, match="inference failed") as exc_info:
            detector.detect([_cell()]).spans

        assert "raw secret input" not in str(exc_info.value)

    def test_rejects_invalid_batch_result_shape(self) -> None:
        detector = Gliner2Detector(
            FreeTextDetectionConfig(),
            model_loader=lambda *_: _FakeModel([]),
        )

        with pytest.raises(GenerationError, match="invalid batch result"):
            detector.detect([_cell()]).spans

    def test_forwards_batch_size_for_multiple_unique_texts(self) -> None:
        texts = ["First value", "Second value", "Third value"]
        model = _FakeModel([{"entities": {}} for _ in texts])
        detector = Gliner2Detector(
            FreeTextDetectionConfig(batch_size=2),
            model_loader=lambda *_: model,
        )

        assert detector.detect([_cell(text) for text in texts]).spans == ()
        assert model.calls[0][0] == texts
        assert model.calls[0][2]["batch_size"] == 2


@pytest.mark.unit
class TestRegexDetector:
    def test_detects_only_supported_structurally_valid_values(self) -> None:
        text = "Email ada@example.com, card 4111 1111 1111 1111, IP 192.168.1.1, invalid 999.2.3.4."

        spans = RegexDetector().detect([_cell(text)]).spans

        assert [(text[span.start : span.end], span.entity_type) for span in spans] == [
            ("ada@example.com", EntityType.EMAIL),
            ("4111 1111 1111 1111", EntityType.CREDIT_DEBIT_CARD),
            ("192.168.1.1", EntityType.IPV4),
        ]

    def test_enforces_per_rule_match_limit(self) -> None:
        text = " ".join(f"person{index}@example.com" for index in range(5))

        with pytest.raises(GenerationError, match="exceeded the maximum of 2 matches") as exc_info:
            RegexDetector(max_matches_per_rule=2).detect([_cell(text)]).spans
        assert "person0@example.com" not in str(exc_info.value)

    def test_detects_unicode_email_and_rejects_invalid_local_part(self) -> None:
        text = "Valid élise@exämple.fr and 用户@example.公司, invalid ada..lovelace@example.com."

        spans = RegexDetector().detect([_cell(text)]).spans

        assert [(text[span.start : span.end], span.entity_type) for span in spans] == [
            ("élise@exämple.fr", EntityType.EMAIL),
            ("用户@example.公司", EntityType.EMAIL),
        ]

    def test_rejects_overlong_email_local_part(self) -> None:
        text = f"{'a' * 65}@example.com"

        assert RegexDetector().detect([_cell(text)]).spans == ()

    def test_accepts_15_digit_cards_only_with_an_american_express_prefix(self) -> None:
        text = "Amex 3782 822463 10005, device IMEI 490154203237518."

        spans = RegexDetector().detect([_cell(text)]).spans

        assert [(text[span.start : span.end], span.entity_type) for span in spans] == [
            ("3782 822463 10005", EntityType.CREDIT_DEBIT_CARD)
        ]

    def test_rejects_repeated_digit_and_non_ascii_card_candidates(self) -> None:
        text = "Invalid cards 4111 1111 1111 1112, 0000 0000 0000 0000, and ٤١١١ ١١١١ ١١١١ ١١١١."

        assert RegexDetector().detect([_cell(text)]).spans == ()

    def test_ipv4_can_be_followed_by_sentence_punctuation(self) -> None:
        text = "Connect to 192.168.1.1. Then continue."

        spans = RegexDetector().detect([_cell(text)]).spans

        assert [(text[span.start : span.end], span.entity_type) for span in spans] == [("192.168.1.1", EntityType.IPV4)]

    def test_wraps_regex_timeouts_as_generation_errors(self) -> None:
        text = "person@example.com " * 100

        with pytest.raises(GenerationError, match="timed out"):
            RegexDetector(timeout_seconds=1e-12).detect([_cell(text)]).spans

    def test_detects_valid_ipv6(self) -> None:
        text = "Connect to 2001:db8::1 now"

        spans = RegexDetector().detect([_cell(text)]).spans

        assert [(text[span.start : span.end], span.entity_type) for span in spans] == [("2001:db8::1", EntityType.IPV6)]

    @pytest.mark.parametrize(
        ("text", "address"),
        [
            ("ip:2001:db8::1", "2001:db8::1"),
            ("::ffff:192.0.2.1", "::ffff:192.0.2.1"),
            ("Mapped ::ffff:192.0.2.1 here", "::ffff:192.0.2.1"),
        ],
    )
    def test_ipv6_after_a_label_colon_and_ipv4_mapped_ipv6_are_replaced_whole(self, text: str, address: str) -> None:
        spans = resolve_overlapping_spans(RegexDetector().detect([_cell(text)]).spans)

        assert [(text[span.start : span.end], span.entity_type) for span in spans] == [(address, EntityType.IPV6)]

    @pytest.mark.parametrize("text", ["Meet at 12:30:45 today", "a:::1", "1::2::3", "2001:db8::1:"])
    def test_rejects_ipv6_shaped_non_addresses(self, text: str) -> None:
        assert RegexDetector().detect([_cell(text)]).spans == ()


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

    def test_more_confident_span_wins_a_same_length_partial_overlap(self) -> None:
        cell_id = DetectionCellId(0, "notes")
        earlier = DetectedSpan(cell_id, 0, 5, EntityType.FIRST_NAME, "gliner", 0.6)
        later = DetectedSpan(cell_id, 2, 7, EntityType.LAST_NAME, "gliner", 0.9)

        assert resolve_overlapping_spans([earlier, later]) == (later,)
        assert resolve_overlapping_spans([later, earlier]) == (later,)

    def test_regex_counts_as_fully_confident_in_a_same_length_partial_overlap(self) -> None:
        cell_id = DetectionCellId(0, "notes")
        gliner_span = DetectedSpan(cell_id, 0, 5, EntityType.API_KEY, "gliner", 0.99)
        regex_span = DetectedSpan(cell_id, 2, 7, EntityType.EMAIL, "regex")

        assert resolve_overlapping_spans([gliner_span, regex_span]) == (regex_span,)

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

        assert detector.detect([cell]).spans == (regex_span,)

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
