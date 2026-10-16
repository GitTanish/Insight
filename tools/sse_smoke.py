"""Live SSE smoke for the webapp's agentic path.

Uploads evaluation/data/sales.csv, asks one agentic question over POST /query,
and checks the stream carries a tool_call -> tool_result pair (in order), a
streamed answer that cites the right region, and no error frame.

Requires a running server: uvicorn webapp.app:app --port 8899
No secrets are read or printed; the server uses its own .env providers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8899"


def main() -> int:
    csv_path = ROOT / "evaluation" / "data" / "sales.csv"
    with httpx.Client(base_url=BASE, timeout=300.0, follow_redirects=False) as client:
        print("healthz:", client.get("/healthz").text.strip()[:60])
        with csv_path.open("rb") as fh:
            up = client.post("/upload", files={"file": ("sales.csv", fh, "text/csv")})
        print("upload status:", up.status_code)
        if up.status_code not in (200, 303):
            print(up.text[:400])
            return 1

        counts: dict = {}
        events: list = []
        error_payload = None
        west_hits = 0
        event = None
        with client.stream("POST", "/query", data={
            "question": "Which region generates the highest total revenue?",
            "agentic": "1",
        }) as resp:
            print("query status:", resp.status_code)
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                    events.append(event)
                elif line.startswith("data:"):
                    counts[event] = counts.get(event, 0) + 1
                    payload = line[5:].strip()
                    if "West" in payload:
                        west_hits += 1
                    if event == "error":
                        error_payload = payload[:300]

    print("event counts:", counts)
    failures = []
    if not (events.count("tool_call") >= 1 and events.count("tool_result") >= 1):
        failures.append("no tool_call/tool_result frames")
    if events.count("tool_call") and "tool_result" in events and (
        events.index("tool_call") > events.index("tool_result")
    ):
        failures.append("tool_result before tool_call")
    if error_payload:
        failures.append(f"error frame: {error_payload}")
    if not west_hits:
        failures.append("answer html never mentions West")
    if "delta" not in counts:
        failures.append("no streamed text deltas")
    if failures:
        print("FAIL:", "; ".join(failures))
        return 1
    print("PASS: live tool trail + streamed answer over SSE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
