"""Companion worker for exported Insight alert monitors.

The webapp's "Alert monitor" export is a JSON descriptor pairing a read-only
DuckDB query with a cron hint, a threshold and a Slack-ready webhook template.
This worker closes the loop: it registers the same CSVs as tables, runs every
monitor's query on a schedule through the identical sandboxed DuckSession
(validate_statement + interrupt watchdog + row cap), applies the threshold,
and posts the rendered payload when the condition changes state.

Usage:
    python -m tools.alert_runner --once --dry-run
    python -m tools.alert_runner --data data --monitors .artifacts/monitors --interval 300
    python -m tools.alert_runner --monitor slow_region --repeat

Posting needs the webhook URL: the monitor's ``webhook.url_env`` variable
(e.g. INSIGHT_ALERT_WEBHOOK_URL), or --webhook. In --dry-run mode the payload
is printed and nothing is sent.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.analytics.sql_engine import DuckSession, unique_table_names  # noqa: E402
from insight.reports.artifacts import slugify  # noqa: E402

_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass
class Monitor:
    name: str
    query: str
    metric_column: Optional[str] = None
    op: str = ">"
    threshold: Any = None
    webhook_url_env: Optional[str] = None
    payload_template: str = "[Insight] {{ monitor_name }} returned {{ rows }} rows"

    @classmethod
    def from_dict(cls, raw: dict) -> "Monitor":
        threshold = raw.get("threshold") or {}
        webhook = raw.get("webhook") or {}
        template = (webhook.get("payload_template") or {}).get(
            "text", cls.payload_template
        )
        return cls(
            name=slugify(raw.get("name") or "insight_monitor"),
            query=(raw.get("query") or "").strip(),
            metric_column=threshold.get("metric_column"),
            op=str(threshold.get("op") or ">"),
            threshold=threshold.get("value"),
            webhook_url_env=webhook.get("url_env"),
            payload_template=template,
        )


@dataclass
class AlertOutcome:
    monitor: str
    rows: int = 0
    triggered: bool = False
    threshold_note: str = ""
    sample: Optional[pd.DataFrame] = None
    error: Optional[str] = None


@dataclass
class Runner:
    monitors: list[Monitor] = field(default_factory=list)
    datasets: dict = field(default_factory=dict)
    webhook_url: Optional[str] = None
    dry_run: bool = False
    repeat: bool = False
    notify_recovery: bool = False
    state: set = field(default_factory=set)
    posted: int = 0

    def evaluate(self, monitor: Monitor, session: DuckSession) -> AlertOutcome:
        if not monitor.query:
            return AlertOutcome(monitor=monitor.name, error="monitor has no query")
        try:
            frame = session.execute(monitor.query)
        except ValueError as exc:
            return AlertOutcome(monitor=monitor.name, error=str(exc))

        outcome = AlertOutcome(
            monitor=monitor.name,
            rows=len(frame),
            sample=frame.head(5),
        )
        if monitor.metric_column and monitor.threshold is not None:
            if monitor.metric_column not in frame.columns:
                outcome.error = (
                    f"threshold column '{monitor.metric_column}' not in result "
                    f"(columns: {list(frame.columns)})"
                )
                return outcome
            column = pd.to_numeric(frame[monitor.metric_column], errors="coerce")
            if column.empty or column.isna().all():
                outcome.error = f"threshold column '{monitor.metric_column}' has no value"
                return outcome
            value = float(column.iloc[-1])
            compare = _OPS.get(monitor.op)
            if compare is None:
                outcome.error = f"unsupported threshold op '{monitor.op}'"
                return outcome
            outcome.triggered = compare(value, monitor.threshold)
            outcome.threshold_note = (
                f"{monitor.metric_column}={value:g} {monitor.op} "
                f"{monitor.threshold:g} -> {outcome.triggered}"
            )
        else:
            outcome.triggered = len(frame) > 0
            outcome.threshold_note = f"{len(frame)} row(s) returned"
        return outcome

    def render_payload(self, monitor: Monitor, outcome: AlertOutcome, resolved: bool = False) -> dict:
        text = monitor.payload_template
        for key, value in (
            ("monitor_name", monitor.name),
            ("rows", outcome.rows),
            ("threshold", outcome.threshold_note or "n/a"),
            ("status", "resolved" if resolved else "triggered"),
        ):
            text = text.replace("{{ " + key + " }}", str(value))
            text = text.replace("{{" + key + "}}", str(value))
        return {"text": f"{text}"}

    def post(self, url: str, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status >= 300:
                raise urllib.error.URLError(f"webhook returned HTTP {response.status}")

    def deliver(self, monitor: Monitor, outcome: AlertOutcome, resolved: bool = False) -> bool:
        payload = self.render_payload(monitor, outcome, resolved=resolved)
        if self.dry_run:
            print(f"[dry-run] {monitor.name}: {payload['text']}")
            self.posted += 1
            return True
        url = self.webhook_url
        if not url and monitor.webhook_url_env:
            import os

            url = os.getenv(monitor.webhook_url_env)
        if not url:
            print(f"[skip] {monitor.name}: no webhook URL "
                  f"(set {monitor.webhook_url_env or '--webhook'})")
            return False
        try:
            self.post(url, payload)
        except (urllib.error.URLError, OSError) as exc:
            print(f"[error] {monitor.name}: webhook post failed: {exc}")
            return False
        print(f"[sent] {monitor.name}: {payload['text']}")
        self.posted += 1
        return True

    def cycle(self) -> None:
        with DuckSession(self.datasets) as session:
            for monitor in self.monitors:
                outcome = self.evaluate(monitor, session)
                if outcome.error:
                    print(f"[warn] {monitor.name}: {outcome.error}")
                    continue
                known = monitor.name in self.state
                if outcome.triggered and (self.repeat or not known):
                    self.deliver(monitor, outcome)
                    self.state.add(monitor.name)
                elif not outcome.triggered and known:
                    self.state.discard(monitor.name)
                    if self.notify_recovery:
                        self.deliver(monitor, outcome, resolved=True)
                elif outcome.triggered and known:
                    print(f"[hold] {monitor.name}: still triggered ({outcome.threshold_note})")


def load_monitors(path: Path, name_filter: Optional[str] = None) -> list[Monitor]:
    files: list[Path] = []
    if path.is_dir():
        files = sorted(path.glob("*.json"))
    elif path.is_file():
        files = [path]
    monitors: list[Monitor] = []
    for file in files:
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] skipping {file.name}: {exc}")
            continue
        monitor = Monitor.from_dict(raw)
        if name_filter and monitor.name != name_filter:
            continue
        monitors.append(monitor)
    return monitors


def load_datasets(path: Path) -> dict:
    frames: dict = {}
    files: list[Path] = []
    if path.is_dir():
        files = sorted(p for p in path.glob("*.csv"))
    elif path.is_file():
        files = [path]
    for file in files:
        try:
            frames[file.name] = pd.read_csv(file)
        except Exception as exc:
            print(f"[warn] cannot read {file.name}: {exc}")
    return frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--monitors", default=".artifacts/monitors",
                        help="monitor JSON file or directory (default: .artifacts/monitors)")
    parser.add_argument("--data", default="data",
                        help="CSV file or directory of CSVs to register (default: data)")
    parser.add_argument("--monitor", default=None, help="run only this monitor name")
    parser.add_argument("--webhook", default=None, help="override webhook URL")
    parser.add_argument("--interval", type=float, default=None,
                        help="seconds between cycles (omit for a single run)")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="render payloads and print them; post nothing")
    parser.add_argument("--repeat", action="store_true",
                        help="post every cycle while triggered (default: on state change)")
    parser.add_argument("--notify-recovery", action="store_true",
                        help="also post when a triggered monitor recovers")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="stop after this many cycles")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    monitors = load_monitors(Path(args.monitors), args.monitor)
    if not monitors:
        print("no monitors found — export one from the webapp (Alert monitor button)")
        return 1

    datasets = load_datasets(Path(args.data))
    if not datasets:
        print(f"no datasets found at {args.data}")
        return 1

    print(f"alert-runner: {len(monitors)} monitor(s), tables: "
          f"{sorted(unique_table_names(list(datasets)).values())}")

    runner = Runner(
        monitors=monitors,
        datasets=datasets,
        webhook_url=args.webhook,
        dry_run=args.dry_run,
        repeat=args.repeat,
        notify_recovery=args.notify_recovery,
    )

    runs = 0
    try:
        while True:
            runs += 1
            runner.cycle()
            if args.once or args.interval is None:
                break
            if args.max_runs is not None and runs >= args.max_runs:
                break
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("\nstopped by user")
    print(f"alert-runner: {runner.posted} posting(s) over {runs} cycle(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
