from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from insight.domain.analysis import ChartSpec, ChartType  # noqa: E402
from insight.domain.execution import DataTable  # noqa: E402
from insight.domain.visualization import PlotArtifact  # noqa: E402
from insight.visualization.themes import EditorialTheme, EDITORIAL_THEME  # noqa: E402


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "chart"


def _sanitize_title(spec: ChartSpec, df: pd.DataFrame, y_series: pd.Series | None = None) -> str:
    """Fix 'Top N' claims that the data cannot support."""
    match = re.search(r"[Tt]op[- ]?(\d+)\b", spec.title or "")
    if not match:
        return spec.title
    claimed = int(match.group(1))
    available_rows = len(df)
    if available_rows >= claimed:
        return spec.title

    focus_col = spec.x or (df.columns[0] if len(df.columns) else "column")
    distinct = None
    if focus_col in df.columns:
        distinct = df[focus_col].nunique()
        if distinct is not None and distinct <= 3:
            return f"Class Balance of {focus_col}"
    target = spec.y or focus_col
    return f"Distribution of {target}"


def _with_other_bucket(
    labels: pd.Series, values: pd.Series, top_n: int
) -> tuple[pd.Series, pd.Series]:
    """Fold rows beyond top_n into an 'Other' slice so totals stay visible."""
    if len(values) <= top_n:
        return labels, values
    kept_idx = values.index[:top_n]
    dropped_idx = values.index[top_n:]
    other_total = float(values.loc[dropped_idx].sum())
    kept_labels = labels.loc[kept_idx]
    kept_values = values.loc[kept_idx]
    return (
        pd.concat([kept_labels, pd.Series(["Other"])], ignore_index=True),
        pd.concat([kept_values, pd.Series([other_total])], ignore_index=True),
    )


def _apply_theme(theme: EditorialTheme) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": theme.background,
            "axes.facecolor": theme.background,
            "savefig.facecolor": theme.background,
            "text.color": theme.foreground,
            "axes.edgecolor": theme.foreground,
            "axes.labelcolor": theme.foreground,
            "xtick.color": theme.foreground,
            "ytick.color": theme.foreground,
            "font.family": theme.font_family,
            "font.serif": theme.font_serif_candidates,
            "axes.grid": True,
            "grid.linestyle": theme.grid_style,
            "grid.alpha": theme.grid_alpha,
            "grid.color": theme.foreground,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _numeric_column(df: pd.DataFrame, name: str) -> pd.Series:
    series = pd.to_numeric(df[name], errors="coerce")
    if series.notna().sum() == 0:
        raise ValueError(f"column '{name}' has no numeric values")
    return series


def _human_number(v: float) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "?"
    magnitude = abs(v)
    if magnitude >= 1_000_000_000:
        return f"{v / 1_000_000_000:.1f}B"
    if magnitude >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if magnitude >= 10_000:
        return f"{v / 1_000:.1f}K"
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.4g}"


def _tick_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda val, _: _human_number(val))


def _style_axes(ax, spec: ChartSpec, theme: EditorialTheme, title_override: str | None = None) -> None:
    ax.set_title(
        title_override or spec.title,
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=12,
        color=theme.foreground,
    )
    if spec.x_label:
        ax.set_xlabel(spec.x_label)
    if spec.y_label:
        ax.set_ylabel(spec.y_label)


def _render_bar_or_pie(
    ax, df: pd.DataFrame, spec: ChartSpec, theme: EditorialTheme, pie: bool
) -> None:
    data = df.copy()
    data["_y"] = _numeric_column(data, spec.y)
    title = _sanitize_title(spec, data, data["_y"])

    orderable = pd.to_numeric(data[spec.x], errors="coerce").notna().mean() < 0.95
    if orderable and spec.resolved_sort_desc():
        data = data.sort_values("_y", ascending=False)

    top_n = spec.resolved_top_n()
    labels_series = data[spec.x].astype(str).str.slice(0, 28)
    values_series = data["_y"]

    if pie:
        kept_labels, kept_values = _with_other_bucket(
            labels_series, values_series, top_n
        )
        labels = kept_labels
        values = kept_values.reset_index(drop=True)
        colors = (theme.palette * 4)[: len(values)]
        wedges, _, autotexts = ax.pie(
            values,
            labels=None,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": theme.background, "linewidth": 1.5},
            textprops={"color": theme.foreground, "fontsize": 8},
        )
        for autotext in autotexts:
            autotext.set_color(theme.background)
            autotext.set_fontweight("bold")
        ax.legend(
            wedges,
            labels,
            loc="center left",
            bbox_to_anchor=(0.98, 0.5),
            frameon=False,
            labelcolor=theme.foreground,
            fontsize=8,
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.grid(False)
        return

    kept_labels, kept_values = _with_other_bucket(
        labels_series, values_series, top_n
    )
    labels = kept_labels.reset_index(drop=True)
    values = kept_values.reset_index(drop=True)

    max_label_len = max((len(str(l)) for l in labels), default=0)
    use_horizontal = len(labels) > 6 and max_label_len > 12

    bars_color = theme.accent
    if use_horizontal:
        y_positions = range(len(values))[::-1]
        ax.barh(
            y_positions, values.values, color=bars_color,
            edgecolor=theme.foreground, linewidth=0.6,
        )
        ax.set_yticks(list(y_positions))
        ax.set_yticklabels([str(l)[:34] for l in labels], fontsize=8)
        ax.xaxis.set_major_formatter(_tick_formatter())
        ax.grid(axis="y", visible=False)
        max_idx = int(values.idxmax()) if len(values) else None
        for i, container in enumerate(ax.containers):
            for j, rect in enumerate(container):
                if j == max_idx:
                    rect.set_color(theme.foreground)
        _style_axes(ax, spec, theme, title_override=title)
        return

    bars = ax.bar(
        range(len(values)), values, color=bars_color,
        edgecolor=theme.foreground, linewidth=0.6,
    )
    max_idx = int(values.idxmax()) if len(values) else None
    for i, bar in enumerate(bars):
        if i == max_idx:
            bar.set_color(theme.foreground)
    rotation = 45 if max_label_len > 12 else 30
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(_tick_formatter())
    _style_axes(ax, spec, theme, title_override=title)


def _render_line(ax, df: pd.DataFrame, spec: ChartSpec, theme: EditorialTheme) -> None:
    if len(df) < 2:
        raise ValueError("line chart needs at least two points")
    y = _numeric_column(df, spec.y)
    ax.plot(
        range(len(y)), y.values, color=theme.accent, linewidth=2.0,
        marker="o", markersize=3.5, markerfacecolor=theme.foreground,
        markeredgecolor=theme.background,
    )
    step = max(1, len(df) // 12)
    ticks = list(range(0, len(df), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [str(v)[:16] for v in df[spec.x].iloc[ticks]], rotation=30,
        ha="right", fontsize=8,
    )
    ax.yaxis.set_major_formatter(_tick_formatter())
    _style_axes(ax, spec, theme)


def _render_scatter(ax, df: pd.DataFrame, spec: ChartSpec, theme: EditorialTheme) -> None:
    x = _numeric_column(df, spec.x)
    y = _numeric_column(df, spec.y)
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 2:
        raise ValueError("scatter chart needs at least two complete pairs")
    if len(x) > 2000:
        positions = np.linspace(0, len(x) - 1, 2000).astype(int)
        x, y = x.iloc[positions], y.iloc[positions]
    ax.scatter(x, y, s=18, alpha=0.65, color=theme.muted, edgecolors="none")
    ax.xaxis.set_major_formatter(_tick_formatter())
    ax.yaxis.set_major_formatter(_tick_formatter())
    _style_axes(ax, spec, theme)


def _render_hist(ax, df: pd.DataFrame, spec: ChartSpec, theme: EditorialTheme) -> None:
    count_col = next(
        (c for c in df.columns if c.lower() in {"count", "row_count"}),
        spec.y if spec.y else df.columns[-1],
    )
    counts = _numeric_column(df, count_col)
    bins = df[spec.x].astype(str) if spec.x else df.iloc[:, 0].astype(str)
    ax.bar(
        range(len(counts)), counts.values, color=theme.muted,
        edgecolor=theme.foreground, linewidth=0.6,
    )
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([b[:18] for b in bins], rotation=35, ha="right", fontsize=7.5)
    ax.yaxis.set_major_formatter(_tick_formatter())
    _style_axes(ax, spec, theme)


_RENDERERS = {
    ChartType.bar: lambda ax, df, spec, t: _render_bar_or_pie(ax, df, spec, t, pie=False),
    ChartType.pie: lambda ax, df, spec, t: _render_bar_or_pie(ax, df, spec, t, pie=True),
    ChartType.line: _render_line,
    ChartType.scatter: _render_scatter,
    ChartType.hist: _render_hist,
}


def render_chart(
    spec: ChartSpec,
    table: DataTable,
    out_dir,
    theme: EditorialTheme = EDITORIAL_THEME,
    index: int = 0,
) -> PlotArtifact:
    df = table.to_dataframe()
    if df.empty:
        raise ValueError(f"no data to plot for '{spec.title}'")

    available = set(df.columns)
    needed = {axis for axis in (spec.x, spec.y) if axis}
    missing = needed - available
    if missing:
        raise ValueError(
            f"chart columns {sorted(missing)} not in step {table.step_id} output "
            f"(available: {sorted(available)})"
        )

    _apply_theme(theme)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    try:
        _RENDERERS[spec.chart_type](ax, df, spec, theme)
        fig.tight_layout()
        filename = (
            f"chart_s{table.step_id}_{index}_{_slugify(spec.title)}.png"
        )
        path = out_dir / filename
        fig.savefig(path, dpi=theme.dpi, bbox_inches="tight", facecolor=theme.background)
    finally:
        plt.close(fig)

    return PlotArtifact(
        path=str(path),
        title=spec.title,
        chart_type=spec.chart_type.value,
        step_id=table.step_id,
    )


def render_charts(
    specs: list[ChartSpec],
    tables_by_step: dict[int, DataTable],
    out_dir,
    theme: EditorialTheme = EDITORIAL_THEME,
) -> tuple[list[PlotArtifact], list[str]]:
    artifacts: list[PlotArtifact] = []
    warnings: list[str] = []

    for i, spec in enumerate(specs):
        table = tables_by_step.get(spec.source_step)
        if table is None:
            warnings.append(
                f"skipped chart '{spec.title}': source step {spec.source_step} produced no table"
            )
            continue
        try:
            artifacts.append(render_chart(spec, table, out_dir, theme, index=i))
        except Exception as exc:
            warnings.append(f"skipped chart '{spec.title}': {exc}")

    return artifacts, warnings
