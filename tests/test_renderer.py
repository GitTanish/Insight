from pathlib import Path

import numpy as np
import pytest
from matplotlib import image as mpimg
from matplotlib.colors import to_rgb

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

SINGLE_TABLE = DataTable(
    step_id=0,
    name="grouped",
    columns=["region", "total"],
    rows=[["North", 214.0]],
    total_rows=1,
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


def test_single_row_bar_is_a_metric_card(tmp_path):
    artifact = render_chart(_spec(ChartType.bar), SINGLE_TABLE, tmp_path)
    path = Path(artifact.path)
    assert path.exists() and path.stat().st_size > 5000

    image = mpimg.imread(path)
    rgb = image[..., :3]
    foreground = np.array(to_rgb(EDITORIAL_THEME.foreground))
    near_foreground = np.all(np.abs(rgb - foreground) < 0.03, axis=2)
    assert near_foreground.mean() < 0.15

    accent = np.array(to_rgb(EDITORIAL_THEME.accent))
    near_accent = np.all(np.abs(rgb - accent) < 0.05, axis=2)
    assert near_accent.sum() > 500


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
