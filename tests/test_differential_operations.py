"""Differential tests: deterministic operations vs independent references.

These tests deliberately avoid importing the verifier. They compare operation
outputs against separate NumPy/SciPy/pandas computations so a shared bug in the
verification path cannot make a wrong result look correct.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from insight.analytics.operations import OPERATIONS, apply_filters
from insight.domain.analysis import AnalysisPlan, Filter, FilterOperator, PlanStep
from insight.execution.executor import execute_plan_sync

SEEDS = (11, 29, 47)


def _frame(seed: int, n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    value = np.round(rng.lognormal(mean=4.6, sigma=0.55, size=n), 2)
    units = rng.integers(1, 25, size=n)
    score = np.round(0.9 * value + rng.normal(0.0, 8.0, size=n), 2)
    x1 = rng.normal(0.0, 1.2, size=n)
    x2 = rng.normal(0.0, 0.8, size=n)
    y = 3.0 + 1.6 * x1 - 0.7 * x2 + rng.normal(0.0, 0.6, size=n)
    frame = pd.DataFrame(
        {
            "group_a": rng.choice(["North", "South", "East", "West"], size=n),
            "group_b": rng.choice(["core", "growth", "other"], size=n),
            "value": value,
            "units": units,
            "score": score,
            "x1": x1,
            "x2": x2,
            "y": y,
            "flag": rng.integers(0, 2, size=n),
        }
    )
    frame.loc[frame.index % 23 == 0, "x1"] = np.nan
    frame.loc[frame.index % 19 == 0, "y"] = np.nan
    frame.loc[frame.index % 31 == 0, "value"] = np.nan
    return frame


def _plan(operation: str, params: dict) -> AnalysisPlan:
    return AnalysisPlan(
        objective="differential",
        steps=[PlanStep(step_id=0, operation=operation, params=params)],
        charts=[],
    )


def _execute(frame: pd.DataFrame, operation: str, params: dict):
    return execute_plan_sync(frame, _plan(operation, params))


def _table(result):
    output = result.steps[0].output
    assert output is not None and output.table is not None, result.failed_messages()
    return output.table.to_dataframe()


def _calcs(result) -> dict:
    output = result.steps[0].output
    assert output is not None
    return {calculation.label: calculation.value for calculation in output.calculations}


def _assert_close(got, expected, *, rel: float = 1e-6, abs_: float = 1e-6) -> None:
    assert float(got) == pytest.approx(float(expected), rel=rel, abs=abs_), (
        got,
        expected,
    )


def _complete(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.dropna()


@pytest.mark.parametrize("seed", SEEDS)
def test_group_aggregate_matches_pandas_reference(seed: int) -> None:
    frame = _frame(seed)
    result = _execute(
        frame,
        "group_aggregate",
        {
            "group_by": ["group_a", "group_b"],
            "metrics": [
                {"column": "value", "agg": "sum", "alias": "total"},
                {"column": "units", "agg": "mean", "alias": "mean_units"},
                {"agg": "count", "alias": "n"},
            ],
            "sort_by": "total",
            "top_n": 6,
        },
    )
    table = _table(result)
    grouped = frame.groupby(["group_a", "group_b"], dropna=False, observed=True)
    expected = pd.DataFrame(
        {
            "total": grouped["value"].sum(),
            "mean_units": grouped["units"].mean(),
            "n": grouped.size(),
        }
    ).reset_index()
    expected = expected.sort_values("total", ascending=False).head(6)

    assert list(table.columns) == ["group_a", "group_b", "total", "mean_units", "n"]
    assert len(table) == len(expected)
    for position in range(len(expected)):
        row = table.iloc[position]
        reference = expected.iloc[position]
        assert row["group_a"] == reference["group_a"]
        assert row["group_b"] == reference["group_b"]
        assert int(row["n"]) == int(reference["n"])
        _assert_close(row["total"], reference["total"])
        _assert_close(row["mean_units"], reference["mean_units"])


@pytest.mark.parametrize("seed", SEEDS)
def test_top_n_matches_stable_sort_reference(seed: int) -> None:
    frame = _frame(seed)
    result = _execute(
        frame,
        "top_n",
        {"sort_by": "score", "n": 12, "columns": ["group_a", "score"]},
    )
    table = _table(result)
    expected = frame.sort_values(
        "score", ascending=False, na_position="last", kind="stable"
    ).head(12)[["group_a", "score"]]
    assert list(table.columns) == ["group_a", "score"]
    assert list(table["group_a"]) == list(expected["group_a"])
    for got, want in zip(table["score"], expected["score"]):
        _assert_close(got, want)


@pytest.mark.parametrize("seed", SEEDS)
def test_value_counts_matches_pandas_reference(seed: int) -> None:
    frame = _frame(seed)
    result = _execute(
        frame,
        "value_counts",
        {"column": "group_a", "normalize": True, "top_n": 10},
    )
    table = _table(result)
    counts = frame["group_a"].value_counts().head(10)
    valid_total = int(frame["group_a"].notna().sum())
    assert len(table) == len(counts)
    for position, (label, count) in enumerate(counts.items()):
        assert table.iloc[position]["group_a"] == label
        assert int(table.iloc[position]["count"]) == int(count)
        _assert_close(
            table.iloc[position]["proportion"], round(count / valid_total, 4), abs_=1e-6
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_correlation_matches_numpy_pairwise_reference(seed: int) -> None:
    frame = _frame(seed)
    columns = ["value", "units", "score"]
    result = _execute(frame, "correlation", {"columns": columns, "method": "pearson"})
    table = _table(result)
    expected = {}
    for left in columns:
        for right in columns:
            if left == right:
                expected[(left, right)] = 1.0
                continue
            pair = _complete(frame, [left, right])
            expected[(left, right)] = round(
                float(np.corrcoef(pair[left], pair[right])[0, 1]), 4
            )

    for row in table.itertuples(index=False):
        row_label = row[0]
        for column_label in table.columns[1:]:
            got = getattr(row, column_label)
            _assert_close(got, expected[(row_label, column_label)], abs_=1e-9)

    calcs = _calcs(result)
    pairs = [
        (abs(expected[(left, right)]), left, right, expected[(left, right)])
        for i, left in enumerate(columns)
        for right in columns[i + 1 :]
    ]
    pairs.sort(reverse=True)
    if pairs:
        _assert_close(calcs["strongest_correlation"], round(pairs[0][3], 6), abs_=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_compare_subsets_matches_numpy_reference(seed: int) -> None:
    frame = _frame(seed)
    result = _execute(
        frame,
        "compare_subsets",
        {
            "metric": {"column": "value", "agg": "mean"},
            "subset_a": [{"column": "flag", "op": "eq", "value": 1}],
            "subset_b": [{"column": "flag", "op": "eq", "value": 0}],
            "label_a": "treated",
            "label_b": "control",
        },
    )
    table = _table(result)
    calcs = _calcs(result)
    expected = {}
    for label, flag in (("treated", 1), ("control", 0)):
        subset = frame.loc[frame["flag"] == flag]
        values = _complete(subset, ["value"])["value"].to_numpy(dtype=float)
        expected[label] = {
            "rows": int(len(subset)),
            "value": float(np.mean(values)),
            "half": 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values)),
        }

    assert list(table["subset"]) == ["treated", "control"]
    for row in table.itertuples(index=False):
        reference = expected[row.subset]
        assert int(row.rows) == reference["rows"]
        _assert_close(row.value, reference["value"])
    _assert_close(
        calcs["absolute_delta"],
        expected["control"]["value"] - expected["treated"]["value"],
    )
    for label in ("treated", "control"):
        _assert_close(
            calcs[f"ci95_low_{label}"],
            expected[label]["value"] - expected[label]["half"],
        )
        _assert_close(
            calcs[f"ci95_high_{label}"],
            expected[label]["value"] + expected[label]["half"],
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_anova_matches_scipy_reference(seed: int) -> None:
    frame = _frame(seed)
    result = _execute(
        frame,
        "anova_test",
        {"group_column": "group_a", "value_column": "y"},
    )
    table = _table(result)
    calcs = _calcs(result)

    complete = frame[["group_a", "y"]].apply(pd.to_numeric, errors="coerce")
    complete["group_a"] = frame["group_a"].astype(str)
    complete = complete.dropna()
    groups = {
        level: group["y"].to_numpy(dtype=float)
        for level, group in complete.groupby("group_a")
    }
    statistic, p_value = sps.f_oneway(*groups.values())
    all_values = np.concatenate(list(groups.values()))
    grand_mean = float(all_values.mean())
    ss_between = sum(
        len(values) * (float(values.mean()) - grand_mean) ** 2
        for values in groups.values()
    )
    ss_total = float(((all_values - grand_mean) ** 2).sum())
    eta_squared = ss_between / ss_total

    expected = complete.groupby("group_a")["y"].agg(["count", "mean", "std"])
    expected = expected.sort_values("mean", ascending=False)
    assert len(table) == len(expected)
    for position, (level, reference) in enumerate(expected.iterrows()):
        row = table.iloc[position]
        assert row["group_a"] == level
        assert int(row["n"]) == int(reference["count"])
        _assert_close(row["mean"], reference["mean"])
        _assert_close(row["std"], reference["std"])

    _assert_close(calcs["statistic"], round(float(statistic), 6), abs_=1e-6)
    _assert_close(calcs["p_value"], round(float(p_value), 6), abs_=1e-6)
    _assert_close(calcs["effect_size"], round(float(eta_squared), 4), abs_=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_regression_matches_scipy_linregress_reference(seed: int) -> None:
    frame = _frame(seed)
    result = _execute(
        frame,
        "regression",
        {"target_column": "y", "feature_columns": ["x1"]},
    )
    table = _table(result)
    calcs = _calcs(result)

    data = _complete(frame, ["x1", "y"])
    reference = sps.linregress(data["x1"], data["y"])
    expected = {
        "intercept": (reference.intercept, reference.intercept_stderr),
        "x1": (reference.slope, reference.stderr),
    }
    rows = {row["term"]: row for row in table.to_dict("records")}
    for term, (coefficient, std_error) in expected.items():
        row = rows[term]
        _assert_close(row["coefficient"], coefficient)
        _assert_close(row["std_error"], std_error)
        expected_t = coefficient / std_error
        _assert_close(row["t_stat"], expected_t)
        expected_p = 2.0 * sps.t.sf(abs(expected_t), len(data) - 2)
        _assert_close(row["p_value"], expected_p)

    _assert_close(
        rows["x1"]["standardized_beta"],
        reference.slope * float(data["x1"].std(ddof=1)) / float(data["y"].std(ddof=1)),
    )
    _assert_close(calcs["r_squared"], reference.rvalue**2)
    _assert_close(calcs["n_observations"], len(data), abs_=0)
    _assert_close(calcs["rows_excluded_incomplete"], len(frame) - len(data), abs_=0)


@pytest.mark.parametrize("seed", SEEDS)
def test_group_aggregate_with_filters_matches_pandas_reference(seed: int) -> None:
    frame = _frame(seed)
    result = _execute(
        frame,
        "group_aggregate",
        {
            "group_by": "group_a",
            "metrics": [{"column": "value", "agg": "mean", "alias": "avg"}],
            "filters": [{"column": "flag", "op": "eq", "value": 1}],
            "sort_by": "avg",
        },
    )
    table = _table(result)
    filtered = apply_filters(
        frame, [Filter(column="flag", op=FilterOperator.eq, value=1)]
    )
    expected = (
        filtered.groupby("group_a", observed=True)["value"]
        .mean()
        .sort_values(ascending=False)
    )
    assert list(table["group_a"]) == list(expected.index)
    for got, want in zip(table["avg"], expected):
        _assert_close(got, want)
