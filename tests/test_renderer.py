from pathlib import Path

import pytest

from insight.domain.analysis import ChartSpec, ChartType
from insight.domain.execution import DataTable
from insight.visualization.renderer import render_chart, render_charts
from insight.visualization.themes import DARK_THEME, EDITORIAL_THEME


TABLE = DataTable(
    step_id=0,
    name="grouped",
    columns=["region", "total"],
    rows=[
        ["North", 100.0],
        ["South", 300.0],
        ["East", 200.5],
        ["West", 250.0],
    ],
    total_rows=4,
)


def _spec(chart_type, **kw):
    defaults = dict(
        chart_type=chart_type, title="Test Chart", source_step=0,
        x="region", y="total",
    )
    defaults.update(kw)
    return ChartSpec(**defaults)


def _render(tmp_path, spec, theme=EDITORIAL_THEME):
    return render_chart(spec, TABLE, tmp_path, theme)


def test_bar_chart(tmp_path):
    artifact = _render(tmp_path, _spec(ChartType.bar))
    path = Path(artifact.path)
    assert path.exists() and path.stat().st_size > 5000
    assert artifact.chart_type == "bar" and artifact.step_id == 0


def test_line_pie_hist_render(tmp_path):
    for chart_type in (ChartType.line, ChartType.pie, ChartType.hist):
        artifact = _render(tmp_path, _spec(chart_type))
        assert Path(artifact.path).stat().st_size > 5000


def test_dark_theme_renders(tmp_path):
    artifact = _render(tmp_path, _spec(ChartType.bar), DARK_THEME)
    assert Path(artifact.path).exists()


def test_missing_axis_column_raises(tmp_path):
    spec = _spec(ChartType.bar, y="nope")
    with pytest.raises(Exception):
        _render(tmp_path, spec)


def test_render_charts_skips_invalid(tmp_path):
    specs = [
        _spec(ChartType.bar),
        ChartSpec(chart_type=ChartType.bar, title="Broken", source_step=7,
                  x="a", y="b"),
    ]
    artifacts, warnings = render_charts(specs, {0: TABLE}, tmp_path)
    assert len(artifacts) == 1
    assert len(warnings) == 1 and "source step 7" in warnings[0]
