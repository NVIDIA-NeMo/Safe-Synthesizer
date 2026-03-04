#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sweep NeMo Safe Synthesizer .out logs and collect summary metrics into a CSV.

Columns:
  exp_setting, file_name, dataset_name, status,
  synthetic_quality_score, column_correlation_stability, deep_structure_stability,
  column_distribution_stability, text_semantic_similarity, text_structure_similarity,
  data_privacy_score, membership_inference_protection_score, attribute_inference_protection_score,
  valid_record_fraction, total_time, pii_replacer_time, training_time, generation_time, evaluation_time

Usage:
  python collect_nss_results.py \
    --root /lustre/fsw/portfolios/llmservice/users/seayang/nss_results/first_run \
    --output /lustre/fsw/portfolios/llmservice/users/seayang/nss_results_first_run.csv

Notes:
  - "status" is "COMPLETED" if the SafeSynthesizer summary block is found; otherwise "INCOMPLETE".
  - The script tolerates None values in metrics and leaves cells blank if a value is missing.
  - Special case: if job finished generation/eval steps but produced no valid output, status is "No Records".
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional

COLUMNS: list[str] = [
    "exp_setting",
    "file_name",
    "dataset_name",
    "replication",
    "status",
    "synthetic_quality_score",
    "column_correlation_stability",
    "deep_structure_stability",
    "column_distribution_stability",
    "text_semantic_similarity",
    "text_structure_similarity",
    "data_privacy_score",
    "membership_inference_protection_score",
    "attribute_inference_protection_score",
    "valid_record_fraction",
    "total_time",
    "pii_replacer_time",
    "training_time",
    "generation_time",
    "evaluation_time",
]


SUMMARY_HEADER = "SafeSynthesizerSummary metrics:"


def _safe_lower(text: Optional[str]) -> str:
    return (text or "").lower()


def parse_summary_block(file_text: str) -> dict[str, str]:
    """Parse the SafeSynthesizer summary block into a dict of raw string values.

    Returns an empty dict if the block is not present.
    """
    lines = file_text.splitlines()
    # Find last occurrence of the summary header
    header_indices = [i for i, line in enumerate(lines) if line.strip() == SUMMARY_HEADER]
    if not header_indices:
        return {}
    start_idx = header_indices[-1] + 1
    values: dict[str, str] = {}
    for i in range(start_idx, len(lines)):
        line = lines[i]
        # Block lines are indented by two spaces followed by key: value
        if not line.startswith("  "):
            break
        # Expect format: two spaces, key, colon, space, value
        # Example: "  synthetic_data_quality_score: 9.1"
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        values[key] = value
    return values


def parse_dataset_name(file_text: str) -> Optional[str]:
    """Parse dataset name from the log tail.

    Supports formats including:
      - "Dataset:  <name>"
      - "Dataset <name>" (no colon, observed in sept19 logs)
    Returns None if not found.
    """
    lines = file_text.splitlines()
    # Search from the end for robustness
    for line in reversed(lines[-300:]):  # Only scan last ~300 lines for speed
        stripped = line.strip()
        if not stripped.lower().startswith("dataset"):
            continue
        # Handle with colon
        if ":" in stripped:
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                candidate = parts[1].strip()
                # Ignore placeholder strings sometimes printed by base code changes
                if candidate and candidate.lower() not in {"magic"}:
                    return candidate
        # Handle without colon: e.g., "Dataset adult"
        parts_ws = stripped.split()
        if len(parts_ws) >= 2:
            # Join the rest to allow multi-word names
            candidate = " ".join(parts_ws[1:]).strip()
            if candidate and candidate.lower() not in {"magic"}:
                return candidate
    # Fallback: scan for matrix line with dataset=NAME
    # Example: "[Matrix] dataset=ai_generated_essays config=..."
    match = re.search(r"\bdataset\s*=\s*([^\s]+)", file_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def determine_exp_setting(root_dir: Path, file_path: Path) -> str:
    """Compute exp_setting as the first two path components under root.

    Example: root/short/unsloth/slurm_*.out -> exp_setting = "short/unsloth"
    Falls back to the immediate parent if fewer components exist.
    """
    rel = file_path.parent.relative_to(root_dir)
    return str(rel)


def build_row(
    root_dir: Path,
    file_path: Path,
    file_text: str,
) -> dict[str, str]:
    """Build a CSV row for the given file contents."""
    values = parse_summary_block(file_text)
    dataset = parse_dataset_name(file_text) or ""
    has_summary = bool(values)
    status = "COMPLETED" if has_summary else "INCOMPLETE"

    # Read matching .err content for downstream classification and signals
    err_path = file_path.with_suffix(".err")
    try:
        err_text = err_path.read_text(errors="ignore") if err_path.exists() else ""
    except Exception:
        err_text = ""

    # Direct copy for keys that match CSV column names
    summary_keys = [
        "synthetic_quality_score",
        "column_correlation_stability",
        "deep_structure_stability",
        "column_distribution_stability",
        "text_semantic_similarity",
        "text_structure_similarity",
        "data_privacy_score",
        "membership_inference_protection_score",
        "attribute_inference_protection_score",
        "valid_record_fraction",
        "total_time",
        "pii_replacer_time",
        "training_time",
        "generation_time",
        "evaluation_time",
    ]

    row: dict[str, str] = {col: "" for col in COLUMNS}
    row["exp_setting"] = determine_exp_setting(root_dir, file_path)
    row["file_name"] = file_path.name
    row["dataset_name"] = dataset
    row["replication"] = parse_replication_id(file_text, file_path)
    # Override COMPLETED if generation ended prematurely due to low valid rate
    if status == "COMPLETED" and is_low_valid_rate_failure(file_text, err_text):
        status = "Failed (Low Rate)"

    row["status"] = status

    for key in summary_keys:
        if key in values:
            row[key] = values[key]

    # If we have a summary but some timing fields are missing, backfill from raw text lines
    if has_summary:
        missing_any_timing = any(
            row.get(col, "") == ""
            for col in ("total_time", "pii_replacer_time", "training_time", "generation_time", "evaluation_time")
        )
        if missing_any_timing:
            fill_timing_metrics_from_text(row, file_text)

    # If not completed, inspect the matching .err file and .out text to refine status
    # Only run incomplete classification when no summary is present. If the
    # job wrote the SafeSynthesizer summary, we keep any special overrides
    # (e.g., "Failed (Low Rate)") and do not downgrade to IN_PROGRESS.
    if not has_summary:
        # First, detect the "No Records" case
        if is_no_records(file_text, err_text):
            status = "No Records"
        else:
            status = classify_incomplete_status(err_text)
        # Attach error message for ERROR runs
        if status == "ERROR":
            msg = extract_error_message(err_text)
            if msg:
                status = f"ERROR ({msg})"
        row["status"] = status

        # Backfill timing metrics if present anywhere in logs
        fill_timing_metrics_from_text(row, file_text)
        if any(
            row.get(col, "") == ""
            for col in ("total_time", "pii_replacer_time", "training_time", "generation_time", "evaluation_time")
        ):
            fill_timing_metrics_from_text(row, err_text)

    return row


def classify_incomplete_status(err_text: str) -> str:
    """Classify an incomplete job as ERROR or IN_PROGRESS using the .err content.

    Heuristics:
      - If error signatures are present, return "ERROR".
      - Otherwise, return "IN_PROGRESS".
    """
    if not err_text:
        return "IN_PROGRESS"

    lowered = _safe_lower(err_text)
    error_tokens = [
        "traceback (most recent call last):",
        "error:",
        "exception",
        "runtimeerror",
        "valueerror",
        "keyerror",
        "memoryerror",
        "segmentation fault",
        "killed",
        "exited with exit code",
        "nccl error",
        "cuda error",
        "oom",
        "out of memory",
    ]
    if any(tok in lowered for tok in error_tokens):
        return "ERROR"
    return "IN_PROGRESS"


def is_no_records(out_text: str, err_text: str) -> bool:
    """Detect runs that finished but produced zero valid outputs.

    Heuristics search both .out and .err content for telltale phrases.
    """
    tokens = [
        "number of valid records generated: 0",
        "no valid records were generated",
        "output is empty!",
        "0 valid records",
    ]
    lowered_out = _safe_lower(out_text)
    lowered_err = _safe_lower(err_text)
    return any(tok in lowered_out for tok in tokens) or any(tok in lowered_err for tok in tokens)


def is_low_valid_rate_failure(out_text: str, err_text: str) -> bool:
    """Detect generation runs that stopped early due to high invalid fraction.

    Looks for explicit stopping messages emitted by the generator when the
    running invalid fraction exceeds the configured threshold.
    """
    lowered_out = _safe_lower(out_text)
    lowered_err = _safe_lower(err_text)
    tokens = [
        "stopping condition reached: invalid_fraction",
        "stopping generation prematurely",
        "running average invalid fraction",
        "please consider increasing the 'num_input_records_to_sample'",
    ]
    return any(tok in lowered_out for tok in tokens) or any(tok in lowered_err for tok in tokens)


def fill_timing_metrics_from_text(row: dict[str, str], text: str) -> None:
    """Parse *_time_sec fields from arbitrary text and populate row timing columns if missing."""
    if not text:
        return
    pattern = re.compile(
        r"^\s*(total_time(?:_sec)?|pii_replacer_time(?:_sec)?|training_time(?:_sec)?|generation_time(?:_sec)?|evaluation_time(?:_sec)?)\s*:\s*([^\s]+)",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    for match in pattern.finditer(text):
        src_key = match.group(1).lower()
        value = match.group(2).strip()
        # Normalize to columns without _sec suffix
        dst_col = src_key.replace("_sec", "")
        if dst_col in row and not row.get(dst_col):
            row[dst_col] = value


def parse_replication_id(file_text: str, file_path: Path) -> str:
    """Extract replication/run id from header lines or path.

    Priority:
      1) [Matrix] ... run_idx=K
      2) [Matrix] output_dir=.../run_K
      3) Filename pattern slurm_<jobid>_<K>.out/.err if present
    Returns empty string if none found.
    """
    # Search early header for run_idx
    header = "\n".join(file_text.splitlines()[:80])
    m = re.search(r"\brun_idx\s*=\s*(\d+)", header)
    if m:
        return m.group(1)
    m = re.search(r"output_dir=.*?/run_(\d+)\b", file_text)
    if m:
        return m.group(1)
    # Fallback from filename like slurm_15752874_11.out
    stem = file_path.stem  # e.g., slurm_15752874_11
    m = re.search(r"_(\d+)$", stem)
    if m:
        return m.group(1)
    return ""


def extract_error_message(err_text: str) -> Optional[str]:
    """Extract a concise error message from .err contents.

    Priority:
      1) Final Python exception line like "ValueError: Output is empty!"
      2) Structured ERROR log line message after the module segment
      3) srun exit code line "Exited with exit code X"
      4) First generic line containing "error"
    """
    if not err_text:
        return None

    lines = err_text.splitlines()
    # 1) Look from the end for Python *Error: ...
    for line in reversed(lines[-400:]):
        stripped = line.strip()
        # Drop rank prefixes like "[rank0]:"
        stripped = re.sub(r"^\[[^\]]+\]\s*:\s*", "", stripped)
        m = re.search(r"([A-Za-z_]+Error: .+)$", stripped)
        if m:
            return m.group(1)

    # 2) Structured ERROR log line: timestamp - ERROR - module - message
    for line in reversed(lines[-400:]):
        stripped = line.strip()
        m = re.search(r"-\s*ERROR\s*-\s*[^-]+-\s*(.+)$", stripped, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # 3) srun exit code
    for line in reversed(lines[-400:]):
        m = re.search(r"exited with exit code\s+(\d+)", line, flags=re.IGNORECASE)
        if m:
            return f"exited with exit code {m.group(1)}"

    # 4) First generic line containing 'error'
    for line in lines[-400:]:
        if re.search(r"error", line, flags=re.IGNORECASE):
            return line.strip()

    return None


def collect_rows(root_dir: Path, skip_incomplete: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    # Scan all .out files; include two-stage and e2e runs uniformly.
    # Training-only logs (slurm_train*.out) are ignored unless their matching .err shows an ERROR.
    all_out_files = sorted(root_dir.rglob("*.out"))
    candidates: list[Path] = []
    for p in all_out_files:
        if p.name.startswith("slurm_train"):
            err_path = p.with_suffix(".err")
            try:
                err_text = err_path.read_text(errors="ignore") if err_path.exists() else ""
            except Exception:
                err_text = ""
            if classify_incomplete_status(err_text) == "ERROR":
                candidates.append(p)
        else:
            candidates.append(p)
    for file_path in candidates:
        try:
            text = file_path.read_text(errors="ignore")
        except Exception:
            continue
        row = build_row(root_dir, file_path, text)
        if skip_incomplete and row.get("status") != "COMPLETED":
            continue
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in COLUMNS})


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SafeSynthesizer metrics from .out logs into a CSV")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/lustre/fsw/portfolios/llmservice/users/seayang/nss_results/first_run"),
        help="Root directory containing experiment subfolders with .out files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/lustre/fsw/portfolios/llmservice/users/seayang/nss_results_first_run.csv"),
        help="Output CSV file path",
    )
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="If set, only include rows for completed jobs",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root_dir: Path = args.root
    if not root_dir.exists() or not root_dir.is_dir():
        print(f"Root directory not found or is not a directory: {root_dir}", file=sys.stderr)
        return 2

    rows = collect_rows(root_dir, skip_incomplete=args.skip_incomplete)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
