"""Companion worker for exported Insight alert monitors.

The webapp's "Alert monitor" export is a JSON descriptor pairing a read-only
DuckDB query with a cron hint, a threshold and a Slack-ready webhook template.
This worker closes the loop: it registers the same CSVs as tables, runs every
monitor's query on a schedule through the identical sandboxed DuckSession
(validate_statement + interrupt watchdog + row cap), applies the threshold,
and posts the rendered payload when the condition changes state.

Usage:
    python -m tools.alert_runner --once --dry-run
    python -m tools.alert_runner --data data --monitors .artifacts/monitors
    python -m tools.alert_runner --data data --monitors .artifacts/monitors --interval 300
    python -m tools.alert_runner --monitor slow_region --repeat

With no ``--interval`` or ``--once``, monitors that carry a ``schedule_cron``
run on that five-field local-time schedule (minute hour day month weekday) and
the worker stays in the foreground. State-change state is persisted to
``--state`` so restarting does not re-post an already-triggering monitor.

Posting needs the webhook URL: the monitor's ``webhook.url_env`` variable
(e.g. INSIGHT_ALERT_WEBHOOK_URL), or --webhook. In --dry-run mode the payload
is printed and nothing is sent; state is not persisted unless --state is given
explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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

_CRON_FIELDS = ("minute", "hour", "day of month", "month", "day of week")


def _parse_cron_field(field: str, low: int, high: int, label: str) -> set[int]:
    values: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty {label} field")
        step = 1
        base = part
        if "/" in part:
            base, step_text = part.split("/", 1)
            try:
                step = int(step_text)
            except ValueError as exc:
                raise ValueError(f"invalid {label} step {step_text!r}") from exc
            if step <= 0:
                raise ValueError(f"{label} step must be positive")
        if base == "*":
            start, end = low, high
        elif "-" in base:
            start_text, end_text = base.split("-", 1)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise ValueError(f"invalid {label} range {base!r}") from exc
        else:
            try:
                start = int(base)
            except ValueError as exc:
                raise ValueError(f"invalid {label} value {base!r}") from exc
            end = high if "/" in part else start
        if start < low or end > high or start > end:
            raise ValueError(f"{label} {part!r} outside {low}-{high}")
        values.update(range(start, end + 1, step))
    return values


@dataclass(frozen=True)
class CronSchedule:
    """A validated five-field cron expression (local time).

    Supports ``*``, ``a``, ``a-b``, lists and ``/step``. Day names and month
    names are not supported; use numbers (0 or 7 is Sunday).
    """

    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]
    day_any: bool
    weekday_any: bool

    @classmethod
    def parse(cls, expression: str) -> "CronSchedule":
        fields = (expression or "").split()
        if len(fields) != 5:
            raise ValueError(
                "expected 5 fields (minute hour day month weekday), "
                f"got {len(fields)}"
            )
        for text, label in zip(fields, _CRON_FIELDS):
            if not text.strip():
                raise ValueError(f"empty {label} field")
        weekdays = {
            0 if value == 7 else value
            for value in _parse_cron_field(fields[4], 0, 7, "day of week")
        }
        schedule = cls(
            minutes=frozenset(_parse_cron_field(fields[0], 0, 59, "minute")),
            hours=frozenset(_parse_cron_field(fields[1], 0, 23, "hour")),
            days=frozenset(_parse_cron_field(fields[2], 1, 31, "day of month")),
            months=frozenset(_parse_cron_field(fields[3], 1, 12, "month")),
            weekdays=frozenset(weekdays),
            day_any=fields[2].strip() == "*",
            weekday_any=fields[4].strip() == "*",
        )
        try:
            schedule.next_after(datetime(2000, 1, 1))
        except ValueError as exc:
            raise ValueError(f"never matches: {exc}") from exc
        return schedule

    def _day_matches(self, moment: datetime) -> bool:
        cron_weekday = (moment.weekday() + 1) % 7
        day_match = moment.day in self.days
        weekday_match = cron_weekday in self.weekdays
        if self.day_any and self.weekday_any:
            return True
        if self.day_any:
            return weekday_match
        if self.weekday_any:
            return day_match
        # Standard cron: when both day fields are restricted, either may match.
        return day_match or weekday_match

    def matches(self, moment: datetime) -> bool:
        return (
            moment.minute in self.minutes
            and moment.hour in self.hours
            and moment.month in self.months
            and self._day_matches(moment)
        )

    def next_after(self, moment: datetime) -> datetime:
        """Return the first matching minute strictly after ``moment``."""
        candidate = moment.replace(second=0, microsecond=0) + timedelta(minutes=1)
        limit = candidate + timedelta(days=366 * 8)
        while candidate < limit:
            if candidate.month not in self.months:
                first_of_month = candidate.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
                candidate = (first_of_month + timedelta(days=32)).replace(day=1)
                continue
            if not self._day_matches(candidate):
                candidate = (candidate + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                continue
            if candidate.hour not in self.hours:
                candidate = (candidate + timedelta(hours=1)).replace(
                    minute=0, second=0, microsecond=0
                )
                continue
            if candidate.minute not in self.minutes:
                candidate += timedelta(minutes=1)
                continue
            return candidate
        raise ValueError(f"cron expression did not match within 8 years: {self}")


@dataclass
class Monitor:
    name: str
    query: str
    metric_column: Optional[str] = None
    op: str = ">"
    threshold: Any = None
    webhook_url_env: Optional[str] = None
    payload_template: str = "[Insight] {{ monitor_name }} returned {{ rows }} rows"
    schedule_cron: Optional[str] = None

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
            schedule_cron=(raw.get("schedule_cron") or "").strip() or None,
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
    state_path: Optional[Path] = None
    last_run: dict[str, float] = field(default_factory=dict)
    last_posted: dict[str, float] = field(default_factory=dict)
    _entries: dict[str, dict] = field(default_factory=dict, repr=False)

    def load_state(self) -> None:
        """Load fired-monitor state and last-run times from ``state_path``."""
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            version = raw.get("version")
            if version not in (None, 1):
                raise ValueError(f"unsupported state version {version!r}")
            entries = raw.get("monitors") or {}
            if not isinstance(entries, dict):
                raise ValueError("monitors is not an object")
            self._entries = {
                name: dict(entry)
                for name, entry in entries.items()
                if isinstance(entry, dict)
            }
            self.state = {
                name for name, entry in self._entries.items() if entry.get("triggered")
            }
            self.last_run = {
                name: float(entry["last_run_at"])
                for name, entry in self._entries.items()
                if entry.get("last_run_at") is not None
            }
            self.last_posted = {
                name: float(entry["last_posted_at"])
                for name, entry in self._entries.items()
                if entry.get("last_posted_at") is not None
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"[warn] cannot read alert state {self.state_path}: {exc}")
            self._entries = {}
            self.state = set()
            self.last_run = {}
            self.last_posted = {}

    def save_state(self) -> None:
        """Atomically persist fired-monitor state (no-op without ``state_path``)."""
        if self.state_path is None:
            return
        entries = {name: dict(entry) for name, entry in self._entries.items()}
        for monitor in self.monitors:
            entries[monitor.name] = {
                "triggered": monitor.name in self.state,
                "last_run_at": self.last_run.get(monitor.name),
                "last_posted_at": self.last_posted.get(monitor.name),
            }
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "monitors": entries,
        }
        tmp = self.state_path.with_name(
            f"{self.state_path.name}.{os.getpid()}.tmp"
        )
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, self.state_path)
            self._entries = entries
        except OSError as exc:
            print(f"[warn] cannot save alert state {self.state_path}: {exc}")
            try:
                tmp.unlink()
            except OSError:
                pass

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
            self.last_posted[monitor.name] = time.time()
            return True
        url = self.webhook_url
        if not url and monitor.webhook_url_env:
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
        self.last_posted[monitor.name] = time.time()
        return True

    def cycle(self, monitors: Optional[list[Monitor]] = None) -> None:
        active = list(self.monitors if monitors is None else monitors)
        with DuckSession(self.datasets) as session:
            for monitor in active:
                outcome = self.evaluate(monitor, session)
                self.last_run[monitor.name] = time.time()
                if outcome.error:
                    print(f"[warn] {monitor.name}: {outcome.error}")
                    self.save_state()
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
                self.save_state()


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


def _due_monitors(
    monitors: list[Monitor],
    schedules: dict[str, CronSchedule],
    last_run: dict[str, float],
    now: float,
) -> list[Monitor]:
    """Monitors whose cron schedule has come due (or that have no schedule)."""
    due: list[Monitor] = []
    for monitor in monitors:
        schedule = schedules.get(monitor.name)
        previous = last_run.get(monitor.name)
        if schedule is None or previous is None:
            due.append(monitor)
            continue
        try:
            next_at = schedule.next_after(datetime.fromtimestamp(previous))
        except (OverflowError, OSError, ValueError):
            due.append(monitor)
            continue
        if next_at.timestamp() <= now:
            due.append(monitor)
    return due


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--monitors", default=".artifacts/monitors",
                        help="monitor JSON file or directory (default: .artifacts/monitors)")
    parser.add_argument("--data", default="data",
                        help="CSV file or directory of CSVs to register (default: data)")
    parser.add_argument("--monitor", default=None, help="run only this monitor name")
    parser.add_argument("--webhook", default=None, help="override webhook URL")
    parser.add_argument("--interval", type=float, default=None,
                        help="seconds between cycles; overrides schedule_cron (omit for schedule mode)")
    parser.add_argument("--poll", type=float, default=30.0,
                        help="seconds between due checks in schedule mode (default: 30)")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="render payloads and print them; post nothing")
    parser.add_argument("--repeat", action="store_true",
                        help="post every cycle while triggered (default: on state change)")
    parser.add_argument("--notify-recovery", action="store_true",
                        help="also post when a triggered monitor recovers")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="stop after this many cycles")
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument(
        "--state",
        default=None,
        help="state file for fired monitors and last-run times "
             "(default: .artifacts/alert_state.json; dry-run defaults to off)",
    )
    state_group.add_argument(
        "--no-state",
        action="store_true",
        help="do not persist fired-monitor state; a restart re-posts active alerts",
    )
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

    schedules: dict[str, CronSchedule] = {}
    for monitor in monitors:
        if not monitor.schedule_cron:
            continue
        try:
            schedules[monitor.name] = CronSchedule.parse(monitor.schedule_cron)
        except ValueError as exc:
            print(f"[warn] {monitor.name}: invalid schedule_cron "
                  f"{monitor.schedule_cron!r}: {exc} (running every cycle)")

    if args.no_state:
        state_path = None
    elif args.state:
        state_path = Path(args.state)
    elif args.dry_run:
        state_path = None
    else:
        state_path = Path(".artifacts/alert_state.json")

    runner = Runner(
        monitors=monitors,
        datasets=datasets,
        webhook_url=args.webhook,
        dry_run=args.dry_run,
        repeat=args.repeat,
        notify_recovery=args.notify_recovery,
        state_path=state_path,
    )
    if state_path is not None:
        runner.load_state()
        print(f"alert-runner: state {state_path}")

    runs = 0
    try:
        if args.once or (args.interval is None and not schedules):
            runs = 1
            runner.cycle()
        elif args.interval is not None:
            while True:
                runs += 1
                runner.cycle()
                if args.max_runs is not None and runs >= args.max_runs:
                    break
                time.sleep(max(1.0, args.interval))
        else:
            described = ", ".join(
                f"{monitor.name}={monitor.schedule_cron}"
                for monitor in monitors
                if monitor.name in schedules
            )
            print(f"alert-runner: cron schedule active ({described}); "
                  f"polling every {args.poll:g}s")
            while True:
                due = _due_monitors(
                    monitors, schedules, runner.last_run, time.time()
                )
                if due:
                    runs += 1
                    runner.cycle(due)
                    if args.max_runs is not None and runs >= args.max_runs:
                        break
                time.sleep(max(1.0, args.poll))
    except KeyboardInterrupt:
        print("\nstopped by user")
    print(f"alert-runner: {runner.posted} posting(s) over {runs} cycle(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
