import os
import time

import pytest

from insight.retention import sweep_artifacts


def _make_artifact(root, name, age_days, now):
    directory = root / "web" / name
    directory.mkdir(parents=True)
    chart = directory / "chart.png"
    chart.write_bytes(b"x" * 10)
    stamp = now - age_days * 86400
    os.utime(chart, (stamp, stamp))
    os.utime(directory, (stamp, stamp))
    return directory


def test_sweep_removes_only_expired_artifact_dirs(tmp_path):
    now = time.time()
    old = _make_artifact(tmp_path, "old", 10, now)
    fresh = _make_artifact(tmp_path, "fresh", 1, now)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "keep.json").write_text("{}", encoding="utf-8")

    report = sweep_artifacts(tmp_path, 7, now=now)

    assert report["scanned"] == 2
    assert report["removed"] == 1
    assert report["bytes_freed"] == 10
    assert not old.exists()
    assert fresh.exists()
    assert (tmp_path / "cache" / "keep.json").exists()


def test_dry_run_reports_without_deleting(tmp_path):
    now = time.time()
    old = _make_artifact(tmp_path, "old", 30, now)

    report = sweep_artifacts(tmp_path, 7, now=now, dry_run=True)

    assert report["removed"] == 1 and report["dry_run"] is True
    assert old.exists()


def test_missing_root_is_a_noop(tmp_path):
    report = sweep_artifacts(tmp_path / "does-not-exist", 7)
    assert report == {
        "root": str(tmp_path / "does-not-exist"),
        "scanned": 0,
        "removed": 0,
        "bytes_freed": 0,
        "dry_run": False,
        "removed_paths": [],
        "failed": [],
    }


def test_negative_age_rejected(tmp_path):
    with pytest.raises(ValueError):
        sweep_artifacts(tmp_path, -1)


def test_symlinks_are_not_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    web = tmp_path / "web"
    web.mkdir()
    link = web / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")

    old = time.time() - 30 * 86400
    os.utime(outside / "keep.txt", (old, old))

    report = sweep_artifacts(tmp_path, 7)

    assert report["removed"] == 0
    assert (outside / "keep.txt").exists()
