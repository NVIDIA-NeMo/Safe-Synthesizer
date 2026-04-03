#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "textual>=0.70",
# ]
# ///
"""NSS Top — a k9s-style TUI for monitoring SLURM jobs.

Usage:
    uv run nss_top.py [--user USERNAME] [--log-dir PATH] [--refresh SECONDS]

Log directory defaults to $BASE_LOG_DIR if set, otherwise you must pass --log-dir.
"""

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, RichLog, Static

# squeue format fields and matching column headers
_SQUEUE_FIELDS = ["%i", "%j", "%T", "%P", "%M", "%l", "%D", "%C", "%R"]
_SQUEUE_SEP = "\x1f"  # unit separator — safe delimiter, never appears in SLURM fields
COLUMNS = ["JobID", "Name", "State", "Partition", "Time", "Limit", "Nodes", "CPUs", "Reason"]

STATE_STYLE: dict[str, str] = {
    "RUNNING": "bold green",
    "PENDING": "yellow",
    "COMPLETING": "cyan",
    "FAILED": "bold red",
    "CANCELLED": "red",
    "TIMEOUT": "magenta",
    "COMPLETED": "dim",
    "OUT_OF_MEMORY": "bold red",
    "NODE_FAIL": "bold red",
}

LOG_TAIL_LINES = 500


async def _run(*args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    return stdout.decode(errors="replace")


async def fetch_jobs(user: str) -> list[dict[str, str]]:
    fmt = _SQUEUE_SEP.join(_SQUEUE_FIELDS)
    output = await _run("squeue", "-u", user, f"--format={fmt}", "--noheader")
    jobs = []
    for line in output.strip().splitlines():
        parts = line.split(_SQUEUE_SEP)
        if len(parts) == len(COLUMNS):
            jobs.append(dict(zip(COLUMNS, [p.strip() for p in parts])))
    return jobs


async def fetch_sstat(job_id: str) -> str:
    """Return a short CPU/mem summary for a running job step, or empty string."""
    output = await _run(
        "sstat",
        "-j",
        f"{job_id}.batch",
        "--format=AveCPU,AveRSS,MaxRSS",
        "--noheader",
        "--parsable2",
    )
    lines = [line for line in output.strip().splitlines() if line]
    if not lines:
        return ""
    parts = lines[0].split("|")
    if len(parts) >= 3:
        avg_cpu, avg_rss, max_rss = parts[0], parts[1], parts[2]
        return f"AveCPU={avg_cpu}  AveRSS={avg_rss}  MaxRSS={max_rss}"
    return ""


def find_log_pair(log_dir: Path, job_id: str) -> tuple[Path | None, Path | None]:
    """Locate .out and .err files for a squeue job ID like '12345_3' or '12345'."""
    if not log_dir.exists():
        return None, None

    # Normalise: squeue returns "12345_3" for array tasks
    # Log files are named slurm_12345_3.{out,err}
    safe_id = job_id.replace("[", "_").replace("]", "")

    out_file = next(log_dir.rglob(f"slurm_{safe_id}.out"), None)
    err_file = next(log_dir.rglob(f"slurm_{safe_id}.err"), None)
    return out_file, err_file


def _tail_lines(path: Path, n: int) -> list[str]:
    """Read the last *n* lines of *path* without loading the whole file."""
    chunk = 1 << 14  # 16 KiB — small enough for a login node
    lines: list[str] = []
    with path.open("rb") as fh:
        fh.seek(0, 2)
        remaining = fh.tell()
        buf = b""
        while remaining > 0 and len(lines) <= n:
            read_size = min(chunk, remaining)
            remaining -= read_size
            fh.seek(remaining)
            buf = fh.read(read_size) + buf
            lines = buf.decode(errors="replace").splitlines()
    return lines[-n:]


class NSSTop(App[None]):
    CSS = """
    Screen { layout: vertical; }

    #job-table {
        height: 50%;
        border: solid $primary;
    }

    #stats-bar {
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
    }

    #log-label {
        height: 1;
        background: $panel;
        padding: 0 1;
    }

    #log-panel {
        height: 1fr;
        border: solid $secondary;
    }

    #status-bar {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("r", "refresh_all", "Refresh"),
        Binding("l", "show_stdout", "Stdout"),
        Binding("e", "show_stderr", "Stderr"),
        Binding("q", "quit", "Quit"),
    ]

    _selected_job: reactive[str | None] = reactive(None)
    _log_mode: reactive[str] = reactive("stdout")  # "stdout" | "stderr"

    def __init__(self, user: str, log_dir: Path | None, refresh_secs: int) -> None:
        super().__init__()
        self.user = user
        self.log_dir = log_dir
        self.refresh_secs = refresh_secs
        self._jobs: list[dict[str, str]] = []

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield DataTable(id="job-table", cursor_type="row", zebra_stripes=True)
            yield Static("", id="stats-bar")
            yield Static("Select a job to view logs", id="log-label")
            yield RichLog(id="log-panel", highlight=True, markup=False, wrap=True)
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#job-table", DataTable)
        for col in COLUMNS:
            table.add_column(col, key=col)
        self.set_interval(self.refresh_secs, self._refresh_jobs)
        self.call_after_refresh(self._refresh_jobs)

    # ── Data fetching ──────────────────────────────────────────────────────────

    async def _refresh_jobs(self) -> None:
        jobs = await fetch_jobs(self.user)
        self._jobs = jobs
        table = self.query_one("#job-table", DataTable)
        cursor_row = table.cursor_row

        table.clear()
        for job in jobs:
            state = job["State"]
            style = STATE_STYLE.get(state, "")
            row: list[Text | str] = []
            for col in COLUMNS:
                val = job.get(col, "")
                row.append(Text(val, style=style) if col == "State" else val)
            table.add_row(*row, key=job["JobID"])

        if jobs and cursor_row < len(jobs):
            table.move_cursor(row=cursor_row)

        status = self.query_one("#status-bar", Static)
        status.update(
            f"User: {self.user}  |  Jobs: {len(jobs)}"
            f"  |  Last refresh: {datetime.now().strftime('%H:%M:%S')}"
            f"  |  Auto-refresh: {self.refresh_secs}s"
        )

    async def _refresh_stats(self, job_id: str) -> None:
        stats_bar = self.query_one("#stats-bar", Static)
        job = next((j for j in self._jobs if j["JobID"] == job_id), None)
        if job is None:
            stats_bar.update("")
            return

        state = job["State"]
        base = f"[bold]{job_id}[/bold]  State: {state}  Nodes: {job['Nodes']}  CPUs: {job['CPUs']}  Time: {job['Time']} / {job['Limit']}"

        if state == "RUNNING":
            sstat = await fetch_sstat(job_id)
            stats_bar.update(f"{base}  |  {sstat}" if sstat else base)
        else:
            reason = job.get("Reason", "")
            stats_bar.update(f"{base}  |  {reason}" if reason and reason != "None" else base)

    async def _load_log(self) -> None:
        job_id = self._selected_job
        log_widget = self.query_one("#log-panel", RichLog)
        label = self.query_one("#log-label", Static)

        if not job_id:
            label.update("Select a job to view logs  [l] stdout  [e] stderr")
            log_widget.clear()
            return

        if not self.log_dir:
            label.update(f"[yellow]{job_id} — no log directory set (use --log-dir or $BASE_LOG_DIR)[/yellow]")
            log_widget.clear()
            return

        out_file, err_file = find_log_pair(self.log_dir, job_id)
        is_stdout = self._log_mode == "stdout"
        target = out_file if is_stdout else err_file
        mode_label = "stdout [l]" if is_stdout else "stderr [e]"

        if not target or not target.exists():
            label.update(f"[yellow]{job_id} — {mode_label}: log not found in {self.log_dir}[/yellow]")
            log_widget.clear()
            log_widget.write("No log file found. The job may still be starting, or BASE_LOG_DIR may be wrong.")
            return

        label.update(f"{job_id} — {mode_label}: {target}")
        log_widget.clear()
        for line in _tail_lines(target, LOG_TAIL_LINES):
            log_widget.write(line)

    # ── Event handlers ─────────────────────────────────────────────────────────

    @on(DataTable.RowHighlighted)
    async def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key and event.row_key.value:
            # Use a separate variable for selected_job and assign at the end to
            # be explicit and avoid type check issues where ty wasn't able to
            # correctly narrow the type.
            selected_job = str(event.row_key.value)
            await self._refresh_stats(selected_job)
            await self._load_log()
            self._selected_job = selected_job

    # ── Actions ────────────────────────────────────────────────────────────────

    async def action_refresh_all(self) -> None:
        await self._refresh_jobs()
        if self._selected_job:
            await self._refresh_stats(self._selected_job)
            await self._load_log()

    async def action_show_stdout(self) -> None:
        self._log_mode = "stdout"
        await self._load_log()

    async def action_show_stderr(self) -> None:
        self._log_mode = "stderr"
        await self._load_log()


def _infer_log_dir(user: str) -> Path | None:
    """Infer the NSS results directory using the same logic as env_variables.sh.

    Resolution order:
    1. $BASE_LOG_DIR
    2. $LUSTRE_DIR/nss_results
    3. /lustre/fsw/portfolios/llmservice/users/<user>/nss_results  (default LUSTRE_DIR)
    """
    if base := os.environ.get("BASE_LOG_DIR"):
        return Path(base)
    lustre_dir = os.environ.get("LUSTRE_DIR") or f"/lustre/fsw/portfolios/llmservice/users/{user}"
    candidate = Path(lustre_dir) / "nss_results"
    return candidate if candidate.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="NSS Top — SLURM job monitor TUI")
    parser.add_argument(
        "--user",
        default=os.environ.get("USER_NAME") or os.environ.get("USER"),
        help="SLURM username (default: $USER_NAME or $USER)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Base log directory (default: $BASE_LOG_DIR, then $LUSTRE_DIR/nss_results)",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=30,
        help="Auto-refresh interval in seconds (default: 30)",
    )
    args = parser.parse_args()

    if not args.user:
        parser.error("Could not determine SLURM username. Set $USER_NAME or pass --user.")

    log_dir = Path(args.log_dir) if args.log_dir else _infer_log_dir(args.user)
    NSSTop(user=args.user, log_dir=log_dir, refresh_secs=args.refresh).run()


if __name__ == "__main__":
    main()
