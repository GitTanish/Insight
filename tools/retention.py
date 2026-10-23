"""Preview or apply artifact retention.

    python tools/retention.py --days 14 --dry-run
    python tools/retention.py --days 14
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from insight.retention import sweep_artifacts  # noqa: E402
from insight.settings import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=settings.artifact_retention_days)
    parser.add_argument("--root", default=str(settings.artifacts_dir))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = sweep_artifacts(args.root, args.days, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
