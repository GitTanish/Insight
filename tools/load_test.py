"""HTTP load smoke for the webapp.

Default mode exercises upload -> parse -> profile -> briefing with concurrent
clients and never calls an LLM, so it is safe to run against any deployment.
``--query`` adds the full question path (needs provider keys on the server).

    python tools/load_test.py --count 20 --workers 8
    python tools/load_test.py --query --question "Which region leads revenue?" --count 8

Exit code 1 when any request fails.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


def _csv_bytes(task_id: int, rows: int) -> bytes:
    regions = ("North", "South", "East", "West")
    lines = ["region,product,revenue,units"]
    for i in range(rows):
        region = regions[(task_id + i) % len(regions)]
        lines.append(f"{region},P{i % 7},{100 + (i * 7) % 900},{1 + i % 5}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _run_task(args, task_id: int) -> tuple[int, dict[str, int], float]:
    started = time.perf_counter()
    headers = {}
    if args.auth_token:
        headers["Authorization"] = f"Bearer {args.auth_token}"
    with httpx.Client(
        base_url=args.base_url, timeout=args.timeout, headers=headers
    ) as client:
        upload = client.post(
            "/upload",
            files={"file": (f"load_{task_id}.csv", _csv_bytes(task_id, args.rows), "text/csv")},
        )
        if not args.query:
            return upload.status_code, {}, time.perf_counter() - started
        query = client.post(
            "/api/query",
            json={"question": args.question, "agentic": bool(args.agentic)},
        )
        return query.status_code, {"upload": upload.status_code}, time.perf_counter() - started


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Concurrent HTTP load smoke")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=20, help="total requests")
    parser.add_argument("--workers", type=int, default=8, help="concurrent clients")
    parser.add_argument("--rows", type=int, default=200, help="rows per synthetic CSV")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--query", action="store_true", help="run full questions (LLM)")
    parser.add_argument("--question", default="Which region has the highest revenue?")
    parser.add_argument("--agentic", action="store_true")
    parser.add_argument("--auth-token", default="")
    args = parser.parse_args(argv)

    if args.count < 1 or args.workers < 1 or args.rows < 2:
        parser.error("count/workers must be >= 1 and rows >= 2")

    started = time.perf_counter()
    durations: list[float] = []
    statuses: Counter[int] = Counter()
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_task, args, task_id): task_id
            for task_id in range(args.count)
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                status, extra, elapsed = future.result()
                durations.append(elapsed)
                statuses[status] += 1
                ok = status == 200 if args.query else status == 303
                if not ok:
                    failures.append(f"task {task_id}: status {status} {extra}")
            except Exception as exc:  # noqa: BLE001 - report every failure kind
                statuses[0] += 1
                failures.append(f"task {task_id}: {type(exc).__name__}: {exc}")

    wall = time.perf_counter() - started
    summary = {
        "mode": "query" if args.query else "upload",
        "base_url": args.base_url,
        "requests": args.count,
        "workers": args.workers,
        "wall_s": round(wall, 2),
        "ok": len(failures) == 0,
        "status_counts": {str(k): v for k, v in sorted(statuses.items())},
        "latency_ms": {
            "p50": round(statistics.median(durations) * 1000, 1),
            "p95": round(_percentile(durations, 0.95) * 1000, 1),
            "max": round(max(durations) * 1000, 1),
        },
        "failures": failures[:5],
    }
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
