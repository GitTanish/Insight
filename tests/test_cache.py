from insight import cache as result_cache
from insight.domain.query import AnalysisRequest
from insight.domain.visualization import AnalysisResponse
from insight.orchestrator import run_analysis_sync

from tests.conftest import EXPLANATION, VALID_PLAN


class _FakeSettings:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = artifacts_dir
        self.query_cache_enabled = True
        self.query_cache_ttl_days = 14
        self.query_cache_max_mb = 512
        self.temperature = 0.0
        self.default_model_id = "groq/x"


def _enable_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "insight.cache.get_settings", lambda: _FakeSettings(tmp_path)
    )


def test_cache_round_trip_serves_second_call_without_llm(
    stub_chain_factory, sales_df, profile, tmp_path, monkeypatch
):
    _enable_cache(monkeypatch, tmp_path)
    stub_chain_factory([VALID_PLAN, EXPLANATION])
    first = run_analysis_sync(
        sales_df, profile,
        AnalysisRequest(question="Which region has the highest revenue?"),
        artifacts_dir=tmp_path / "a",
    )
    assert first.error is None and first.meta.get("cache_hit") is False

    stub_chain_factory([])
    second = run_analysis_sync(
        sales_df, profile,
        AnalysisRequest(question="  which REGION has the highest revenue? "),
        artifacts_dir=tmp_path / "b",
    )
    assert second.answer == first.answer
    assert second.meta["cache_hit"] is True
    assert len(second.plots) == len(first.plots)
    for plot in second.plots:
        assert str(tmp_path / "b") in plot.path


def test_use_cache_false_bypasses_cache(
    stub_chain_factory, sales_df, profile, tmp_path, monkeypatch
):
    _enable_cache(monkeypatch, tmp_path)
    stub_chain_factory([VALID_PLAN, EXPLANATION])
    run_analysis_sync(
        sales_df, profile,
        AnalysisRequest(question="Which region has the highest revenue?"),
        artifacts_dir=tmp_path,
    )

    llm = stub_chain_factory([VALID_PLAN, "**Fresh answer.**"])
    again = run_analysis_sync(
        sales_df, profile,
        AnalysisRequest(question="Which region has the highest revenue?"),
        artifacts_dir=tmp_path,
        use_cache=False,
    )
    assert again.answer == "**Fresh answer.**"
    assert again.meta.get("cache_hit", False) is False
    assert len(llm.requests) == 2


def test_failed_plan_is_not_cached(stub_chain_factory, sales_df, profile, tmp_path, monkeypatch):
    _enable_cache(monkeypatch, tmp_path)
    stub_chain_factory(["no json {", "still no {"])
    response = run_analysis_sync(
        sales_df, profile,
        AnalysisRequest(question="Which region has the highest revenue?"),
        artifacts_dir=tmp_path,
    )
    assert response.error == "plan_validation_failed"
    keys = list((tmp_path / "cache").glob("*.json")) if (tmp_path / "cache").exists() else []
    assert keys == []


def test_cache_disabled_flag_skips_lookup(
    stub_chain_factory, sales_df, profile, tmp_path, monkeypatch
):
    def _off_settings():
        settings = _FakeSettings(tmp_path)
        settings.query_cache_enabled = False
        return settings

    monkeypatch.setattr("insight.cache.get_settings", _off_settings)
    stub_chain_factory([VALID_PLAN, EXPLANATION])
    first = run_analysis_sync(
        sales_df, profile,
        AnalysisRequest(question="Which region has the highest revenue?"),
        artifacts_dir=tmp_path,
    )
    assert result_cache.cache_key_for(profile, AnalysisRequest(question="x")) is None
    assert first.meta.get("cache_hit") is None

    llm = stub_chain_factory([VALID_PLAN, "**Second answer.**"])
    second = run_analysis_sync(
        sales_df, profile,
        AnalysisRequest(question="Which region has the highest revenue?"),
        artifacts_dir=tmp_path,
    )
    assert second.answer == "**Second answer.**"
    assert len(llm.requests) == 2


def test_sweep_cache_enforces_ttl(tmp_path, monkeypatch):
    import os
    import time as _time

    old_key = tmp_path / "cache" / "aaaa"
    new_key = tmp_path / "cache" / "bbbb"
    for key in (old_key, new_key):
        key.mkdir(parents=True)
        (key / "k.json").write_text("{}", encoding="utf-8")

    now = _time.time()
    os.utime(old_key, (now - 3 * 86_400, now - 3 * 86_400))
    os.utime(new_key, (now, now))

    class _TTLSettings(_FakeSettings):
        def __init__(self):
            super().__init__(tmp_path)
            self.query_cache_ttl_days = 1

    monkeypatch.setattr("insight.cache.get_settings", lambda: _TTLSettings())

    removed = result_cache.sweep_cache()

    assert removed == 1
    assert not old_key.exists()
    assert new_key.exists()


def test_sweep_cache_enforces_size_budget(tmp_path, monkeypatch):
    import os
    import time as _time

    older = tmp_path / "cache" / "aaaa"
    newer = tmp_path / "cache" / "bbbb"
    for key in (older, newer):
        key.mkdir(parents=True)
        (key / "blob.bin").write_bytes(b"x" * 700_000)

    now = _time.time()
    os.utime(older, (now - 120, now - 120))
    os.utime(newer, (now - 60, now - 60))

    class _SizeSettings(_FakeSettings):
        def __init__(self):
            super().__init__(tmp_path)
            self.query_cache_ttl_days = 3650
            self.query_cache_max_mb = 1

    monkeypatch.setattr("insight.cache.get_settings", lambda: _SizeSettings())

    removed = result_cache.sweep_cache()

    assert removed == 1
    assert not older.exists()
    assert newer.exists()


def test_store_runs_sweep_before_write(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("insight.cache.sweep_cache", lambda: calls.append(1) or 0)

    class _On(_FakeSettings):
        pass

    monkeypatch.setattr("insight.cache.get_settings", lambda: _On(tmp_path))

    response = AnalysisResponse(question="q", answer="a")
    result_cache.store_cached_response("testkey", response)

    assert calls == [1]
    assert (tmp_path / "cache" / "testkey.json").exists()
