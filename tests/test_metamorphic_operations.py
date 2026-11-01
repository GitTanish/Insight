"""Metamorphic tests: transformations with predictable effects.

Each test executes the same operation on a base frame and a transformed frame,
then checks the relation the transformation must satisfy. These properties catch
systematic errors that a single hand-checked example can miss.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from insight.domain.analysis import AnalysisPlan, PlanStep
from insight.execution.executor import execute_plan_sync

SEEDS = (11, 29)
SCALE = 4.0
SHIFT = 10.0


def _frame(seed: int, n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    value = np.round(rng.lognormal(mean=4.6, sigma=0.5, size=n), 2)
    units = rng.integers(1, 25, size=n)
    score = np.round(np.arange(n, dtype=float) * 1.7 + rng.normal(0.0, 0.01, n), 4)
    x1 = rng.normal(0.0, 1.2, size=n)
    x2 = rng.normal(0.0, 0.8, size=n)
    y = 3.0 + 1.6 * x1 - 0.7 * x2 + rng.normal(0.0, 0.6, size=n)
    return pd.DataFrame(
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


def _execute(frame: pd.DataFrame, operation: str, params: dict):
    plan = AnalysisPlan(
        objective="metamorphic",
        steps=[PlanStep(step_id=0, operation=operation, params=params)],
        charts=[],
    )
    result = execute_plan_sync(frame, plan)
    assert result.success, result.failed_messages()
    return result


def _table(result) -> pd.DataFrame:
    output = result.steps[0].output
    assert output is not None and output.table is not None
    return output.table.to_dataframe()


def _calcs(result) -> dict:
    output = result.steps[0].output
    assert output is not None
    return {calculation.label: calculation.value for calculation in output.calculations}


def _close(got, expected, *, rel: float = 1e-6, abs_: float = 1e-6) -> None:
    got_value = float(got)
    expected_value = float(expected)
    if math.isnan(got_value) or math.isnan(expected_value):
        assert math.isnan(got_value) and math.isnan(expected_value), (got, expected)
        return
    assert got_value == pytest.approx(expected_value, rel=rel, abs=abs_), (
        got,
        expected,
    )


def _keyed(table: pd.DataFrame, keys: list[str]) -> dict:
    if len(keys) == 1:
        return {row[keys[0]]: row for _, row in table.iterrows()}
    return {tuple(row[key] for key in keys): row for _, row in table.iterrows()}


def _group_plan() -> dict:
    return {
        "group_by": "group_a",
        "metrics": [
            {"column": "value", "agg": "sum", "alias": "total"},
            {"column": "value", "agg": "mean", "alias": "avg"},
            {"agg": "count", "alias": "n"},
        ],
        "sort_by": "total",
    }


def _anova_plan(value_column: str = "y") -> dict:
    return {"group_column": "group_a", "value_column": value_column}


def _regression_plan(target: str = "y") -> dict:
    return {"target_column": target, "feature_columns": ["x1", "x2"]}


def _correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = _execute(
        frame, "correlation", {"columns": ["value", "units", "score"]}
    )
    return _table(result)


@pytest.mark.parametrize("seed", SEEDS)
def test_row_permutation_preserves_results(seed: int) -> None:
    frame = _frame(seed)
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    base = _keyed(_table(_execute(frame, "group_aggregate", _group_plan())), ["group_a"])
    moved = _keyed(
        _table(_execute(shuffled, "group_aggregate", _group_plan())), ["group_a"]
    )
    assert set(base) == set(moved)
    for key in base:
        for column in ("total", "avg", "n"):
            _close(moved[key][column], base[key][column])

    base_top = _table(_execute(frame, "top_n", {"sort_by": "score", "n": 15}))
    moved_top = _table(_execute(shuffled, "top_n", {"sort_by": "score", "n": 15}))
    assert list(moved_top["score"]) == list(base_top["score"])
    assert list(moved_top["group_a"]) == list(base_top["group_a"])

    base_counts = _table(
        _execute(frame, "value_counts", {"column": "group_a", "normalize": True})
    )
    moved_counts = _table(
        _execute(shuffled, "value_counts", {"column": "group_a", "normalize": True})
    )
    assert _keyed(moved_counts, ["group_a"]).keys() == _keyed(
        base_counts, ["group_a"]
    ).keys()
    for key, row in _keyed(base_counts, ["group_a"]).items():
        _close(moved_counts.set_index("group_a").loc[key, "count"], row["count"])
        _close(moved_counts.set_index("group_a").loc[key, "proportion"], row["proportion"])

    base_corr = _correlation_table(frame)
    moved_corr = _correlation_table(shuffled)
    for column in base_corr.columns[1:]:
        for base_value, moved_value in zip(base_corr[column], moved_corr[column]):
            _close(moved_value, base_value, abs_=1e-9)

    base_anova = _table(_execute(frame, "anova_test", _anova_plan()))
    moved_anova = _table(_execute(shuffled, "anova_test", _anova_plan()))
    base_rows = _keyed(base_anova, ["group_a"])
    moved_rows = _keyed(moved_anova, ["group_a"])
    assert set(base_rows) == set(moved_rows)
    for key in base_rows:
        for column in ("n", "mean", "std"):
            _close(moved_rows[key][column], base_rows[key][column])

    base_reg = _table(_execute(frame, "regression", _regression_plan()))
    moved_reg = _table(_execute(shuffled, "regression", _regression_plan()))
    base_terms = _keyed(base_reg, ["term"])
    moved_terms = _keyed(moved_reg, ["term"])
    assert set(base_terms) == set(moved_terms)
    for term in base_terms:
        for column in ("coefficient", "std_error", "t_stat", "p_value"):
            _close(moved_terms[term][column], base_terms[term][column])

    base_cmp = _table(
        _execute(
            frame,
            "compare_subsets",
            {
                "metric": {"column": "value", "agg": "mean"},
                "subset_a": [{"column": "flag", "op": "eq", "value": 1}],
                "subset_b": [{"column": "flag", "op": "eq", "value": 0}],
            },
        )
    )
    moved_cmp = _table(
        _execute(
            shuffled,
            "compare_subsets",
            {
                "metric": {"column": "value", "agg": "mean"},
                "subset_a": [{"column": "flag", "op": "eq", "value": 1}],
                "subset_b": [{"column": "flag", "op": "eq", "value": 0}],
            },
        )
    )
    assert list(moved_cmp["subset"]) == list(base_cmp["subset"])
    for column in ("rows", "value"):
        for base_value, moved_value in zip(base_cmp[column], moved_cmp[column]):
            _close(moved_value, base_value)


@pytest.mark.parametrize("seed", SEEDS)
def test_duplicate_rows_relations(seed: int) -> None:
    frame = _frame(seed)
    duplicated = pd.concat([frame, frame], ignore_index=True)

    base = _keyed(_table(_execute(frame, "group_aggregate", _group_plan())), ["group_a"])
    doubled = _keyed(
        _table(_execute(duplicated, "group_aggregate", _group_plan())), ["group_a"]
    )
    assert set(base) == set(doubled)
    for key in base:
        _close(doubled[key]["total"], base[key]["total"] * 2)
        _close(doubled[key]["avg"], base[key]["avg"])
        assert int(doubled[key]["n"]) == int(base[key]["n"]) * 2

    base_counts = _table(
        _execute(frame, "value_counts", {"column": "group_a", "normalize": True})
    )
    doubled_counts = _table(
        _execute(duplicated, "value_counts", {"column": "group_a", "normalize": True})
    )
    base_map = _keyed(base_counts, ["group_a"])
    doubled_map = _keyed(doubled_counts, ["group_a"])
    for key in base_map:
        assert int(doubled_map[key]["count"]) == int(base_map[key]["count"]) * 2
        _close(doubled_map[key]["proportion"], base_map[key]["proportion"])

    base_top = _table(_execute(frame, "top_n", {"sort_by": "score", "n": 15}))
    doubled_top = _table(
        _execute(duplicated, "top_n", {"sort_by": "score", "n": 30})
    )
    expected_scores = [score for score in base_top["score"] for _ in range(2)]
    assert list(doubled_top["score"]) == expected_scores

    base_corr = _correlation_table(frame)
    doubled_corr = _correlation_table(duplicated)
    for column in base_corr.columns[1:]:
        for base_value, doubled_value in zip(base_corr[column], doubled_corr[column]):
            _close(doubled_value, base_value, abs_=1e-9)

    base_anova = _table(_execute(frame, "anova_test", _anova_plan()))
    doubled_anova = _table(_execute(duplicated, "anova_test", _anova_plan()))
    base_rows = _keyed(base_anova, ["group_a"])
    doubled_rows = _keyed(doubled_anova, ["group_a"])
    for key in base_rows:
        base_n = int(base_rows[key]["n"])
        assert int(doubled_rows[key]["n"]) == base_n * 2
        _close(doubled_rows[key]["mean"], base_rows[key]["mean"])
        # Sample std uses ddof=1, so duplicating every row changes the
        # denominator from n-1 to 2n-1 rather than leaving the value fixed.
        _close(
            doubled_rows[key]["std"],
            base_rows[key]["std"] * math.sqrt(2 * (base_n - 1) / (2 * base_n - 1)),
        )
    base_calcs = _calcs(_execute(frame, "anova_test", _anova_plan()))
    doubled_calcs = _calcs(_execute(duplicated, "anova_test", _anova_plan()))
    # Classic ANOVA's within-group df is N-k, which becomes 2N-k after
    # duplication rather than 2(N-k); F therefore scales by (2N-k)/(N-k).
    total_n = sum(int(row["n"]) for row in base_rows.values())
    group_count = len(base_rows)
    _close(
        doubled_calcs["statistic"],
        base_calcs["statistic"] * (2 * total_n - group_count) / (total_n - group_count),
    )
    _close(doubled_calcs["effect_size"], base_calcs["effect_size"])

    base_reg = _table(_execute(frame, "regression", _regression_plan()))
    doubled_reg = _table(_execute(duplicated, "regression", _regression_plan()))
    base_terms = _keyed(base_reg, ["term"])
    doubled_terms = _keyed(doubled_reg, ["term"])
    for term in base_terms:
        _close(doubled_terms[term]["coefficient"], base_terms[term]["coefficient"])
        _close(
            doubled_terms[term]["standardized_beta"],
            base_terms[term]["standardized_beta"],
        )
    base_calcs = _calcs(_execute(frame, "regression", _regression_plan()))
    doubled_calcs = _calcs(_execute(duplicated, "regression", _regression_plan()))
    _close(doubled_calcs["r_squared"], base_calcs["r_squared"])
    _close(
        doubled_calcs["n_observations"],
        float(base_calcs["n_observations"]) * 2,
        abs_=0,
    )

    base_cmp = _table(
        _execute(
            frame,
            "compare_subsets",
            {
                "metric": {"column": "value", "agg": "mean"},
                "subset_a": [{"column": "flag", "op": "eq", "value": 1}],
                "subset_b": [{"column": "flag", "op": "eq", "value": 0}],
            },
        )
    )
    doubled_cmp = _table(
        _execute(
            duplicated,
            "compare_subsets",
            {
                "metric": {"column": "value", "agg": "mean"},
                "subset_a": [{"column": "flag", "op": "eq", "value": 1}],
                "subset_b": [{"column": "flag", "op": "eq", "value": 0}],
            },
        )
    )
    for base_row, doubled_row in zip(
        base_cmp.itertuples(index=False), doubled_cmp.itertuples(index=False)
    ):
        assert int(doubled_row.rows) == int(base_row.rows) * 2
        _close(doubled_row.value, base_row.value)


@pytest.mark.parametrize("seed", SEEDS)
def test_positive_scale_relations(seed: int) -> None:
    frame = _frame(seed)
    scaled = frame.copy()
    scaled["value"] = scaled["value"] * SCALE
    scaled_y = frame.copy()
    scaled_y["y"] = scaled_y["y"] * SCALE

    base = _keyed(_table(_execute(frame, "group_aggregate", _group_plan())), ["group_a"])
    scaled_rows = _keyed(
        _table(_execute(scaled, "group_aggregate", _group_plan())), ["group_a"]
    )
    for key in base:
        _close(scaled_rows[key]["total"], base[key]["total"] * SCALE)
        _close(scaled_rows[key]["avg"], base[key]["avg"] * SCALE)
        assert int(scaled_rows[key]["n"]) == int(base[key]["n"])

    base_top = _table(_execute(frame, "top_n", {"sort_by": "value", "n": 15}))
    scaled_top = _table(_execute(scaled, "top_n", {"sort_by": "value", "n": 15}))
    assert list(scaled_top["group_a"]) == list(base_top["group_a"])
    for base_value, scaled_value in zip(base_top["value"], scaled_top["value"]):
        _close(scaled_value, base_value * SCALE)

    base_corr = _correlation_table(frame)
    scaled_corr = _correlation_table(scaled)
    for column in base_corr.columns[1:]:
        for base_value, scaled_value in zip(base_corr[column], scaled_corr[column]):
            _close(scaled_value, base_value, abs_=1e-9)

    base_anova = _table(_execute(frame, "anova_test", _anova_plan()))
    scaled_anova = _table(_execute(scaled_y, "anova_test", _anova_plan()))
    for base_row, scaled_row in zip(
        base_anova.itertuples(index=False), scaled_anova.itertuples(index=False)
    ):
        _close(scaled_row.mean, base_row.mean * SCALE)
        _close(scaled_row.std, base_row.std * SCALE)
        assert int(scaled_row.n) == int(base_row.n)
    base_calcs = _calcs(_execute(frame, "anova_test", _anova_plan()))
    scaled_calcs = _calcs(_execute(scaled_y, "anova_test", _anova_plan()))
    for label in ("statistic", "p_value", "effect_size"):
        _close(scaled_calcs[label], base_calcs[label])

    base_reg = _table(_execute(frame, "regression", _regression_plan()))
    scaled_reg = _table(_execute(scaled_y, "regression", _regression_plan()))
    base_terms = _keyed(base_reg, ["term"])
    scaled_terms = _keyed(scaled_reg, ["term"])
    for term, row in base_terms.items():
        _close(
            scaled_terms[term]["coefficient"],
            row["coefficient"] * SCALE,
            abs_=1e-5,
        )
        _close(scaled_terms[term]["std_error"], row["std_error"] * SCALE, abs_=1e-5)
        _close(scaled_terms[term]["t_stat"], row["t_stat"])
        _close(scaled_terms[term]["p_value"], row["p_value"])
        _close(
            scaled_terms[term]["standardized_beta"], row["standardized_beta"]
        )
    base_calcs = _calcs(_execute(frame, "regression", _regression_plan()))
    scaled_calcs = _calcs(_execute(scaled_y, "regression", _regression_plan()))
    _close(scaled_calcs["r_squared"], base_calcs["r_squared"])

    base_cmp = _table(
        _execute(
            frame,
            "compare_subsets",
            {
                "metric": {"column": "value", "agg": "mean"},
                "subset_a": [{"column": "flag", "op": "eq", "value": 1}],
                "subset_b": [{"column": "flag", "op": "eq", "value": 0}],
            },
        )
    )
    scaled_cmp = _table(
        _execute(
            scaled,
            "compare_subsets",
            {
                "metric": {"column": "value", "agg": "mean"},
                "subset_a": [{"column": "flag", "op": "eq", "value": 1}],
                "subset_b": [{"column": "flag", "op": "eq", "value": 0}],
            },
        )
    )
    for base_row, scaled_row in zip(
        base_cmp.itertuples(index=False), scaled_cmp.itertuples(index=False)
    ):
        assert int(scaled_row.rows) == int(base_row.rows)
        _close(scaled_row.value, base_row.value * SCALE)
    base_calcs = _calcs(
        _execute(
            frame,
            "compare_subsets",
            {
                "metric": {"column": "value", "agg": "mean"},
                "subset_a": [{"column": "flag", "op": "eq", "value": 1}],
                "subset_b": [{"column": "flag", "op": "eq", "value": 0}],
            },
        )
    )
    scaled_calcs = _calcs(
        _execute(
            scaled,
            "compare_subsets",
            {
                "metric": {"column": "value", "agg": "mean"},
                "subset_a": [{"column": "flag", "op": "eq", "value": 1}],
                "subset_b": [{"column": "flag", "op": "eq", "value": 0}],
            },
        )
    )
    _close(scaled_calcs["absolute_delta"], base_calcs["absolute_delta"] * SCALE)
    for label in ("ci95_low_a", "ci95_low_b", "ci95_high_a", "ci95_high_b"):
        if label in base_calcs and label in scaled_calcs:
            _close(scaled_calcs[label], base_calcs[label] * SCALE)


@pytest.mark.parametrize("seed", SEEDS)
def test_additive_shift_relations(seed: int) -> None:
    frame = _frame(seed)
    shifted = frame.copy()
    shifted["value"] = shifted["value"] + SHIFT
    shifted_y = frame.copy()
    shifted_y["y"] = shifted_y["y"] + SHIFT

    base = _keyed(_table(_execute(frame, "group_aggregate", _group_plan())), ["group_a"])
    shifted_rows = _keyed(
        _table(_execute(shifted, "group_aggregate", _group_plan())), ["group_a"]
    )
    for key in base:
        _close(
            shifted_rows[key]["total"],
            base[key]["total"] + SHIFT * int(base[key]["n"]),
        )
        _close(shifted_rows[key]["avg"], base[key]["avg"] + SHIFT)
        assert int(shifted_rows[key]["n"]) == int(base[key]["n"])

    base_top = _table(_execute(frame, "top_n", {"sort_by": "value", "n": 15}))
    shifted_top = _table(_execute(shifted, "top_n", {"sort_by": "value", "n": 15}))
    assert list(shifted_top["group_a"]) == list(base_top["group_a"])
    for base_value, shifted_value in zip(base_top["value"], shifted_top["value"]):
        _close(shifted_value, base_value + SHIFT)

    base_corr = _correlation_table(frame)
    shifted_corr = _correlation_table(shifted)
    for column in base_corr.columns[1:]:
        for base_value, shifted_value in zip(base_corr[column], shifted_corr[column]):
            _close(shifted_value, base_value, abs_=1e-9)

    base_anova = _table(_execute(frame, "anova_test", _anova_plan()))
    shifted_anova = _table(_execute(shifted_y, "anova_test", _anova_plan()))
    for base_row, shifted_row in zip(
        base_anova.itertuples(index=False), shifted_anova.itertuples(index=False)
    ):
        _close(shifted_row.mean, base_row.mean + SHIFT)
        _close(shifted_row.std, base_row.std)
        assert int(shifted_row.n) == int(base_row.n)
    base_calcs = _calcs(_execute(frame, "anova_test", _anova_plan()))
    shifted_calcs = _calcs(_execute(shifted_y, "anova_test", _anova_plan()))
    for label in ("statistic", "p_value", "effect_size"):
        _close(shifted_calcs[label], base_calcs[label])

    base_reg = _table(_execute(frame, "regression", _regression_plan()))
    shifted_reg = _table(_execute(shifted_y, "regression", _regression_plan()))
    base_terms = _keyed(base_reg, ["term"])
    shifted_terms = _keyed(shifted_reg, ["term"])
    _close(
        shifted_terms["intercept"]["coefficient"],
        base_terms["intercept"]["coefficient"] + SHIFT,
        abs_=1e-5,
    )
    for term in ("x1", "x2"):
        _close(shifted_terms[term]["coefficient"], base_terms[term]["coefficient"])
        _close(shifted_terms[term]["t_stat"], base_terms[term]["t_stat"])
        _close(shifted_terms[term]["p_value"], base_terms[term]["p_value"])
        _close(
            shifted_terms[term]["standardized_beta"],
            base_terms[term]["standardized_beta"],
        )
    base_calcs = _calcs(_execute(frame, "regression", _regression_plan()))
    shifted_calcs = _calcs(_execute(shifted_y, "regression", _regression_plan()))
    for label in ("r_squared", "adj_r_squared"):
        _close(shifted_calcs[label], base_calcs[label])

    base_cmp = _table(
        _execute(
            frame,
            "compare_subsets",
            {
                "metric": {"column": "value", "agg": "mean"},
                "subset_a": [{"column": "flag", "op": "eq", "value": 1}],
                "subset_b": [{"column": "flag", "op": "eq", "value": 0}],
            },
        )
    )
    shifted_cmp = _table(
        _execute(
            shifted,
            "compare_subsets",
            {
                "metric": {"column": "value", "agg": "mean"},
                "subset_a": [{"column": "flag", "op": "eq", "value": 1}],
                "subset_b": [{"column": "flag", "op": "eq", "value": 0}],
            },
        )
    )
    for base_row, shifted_row in zip(
        base_cmp.itertuples(index=False), shifted_cmp.itertuples(index=False)
    ):
        assert int(shifted_row.rows) == int(base_row.rows)
        _close(shifted_row.value, base_row.value + SHIFT)
    base_calcs = _calcs(
        _execute(
            frame,
            "compare_subsets",
            {
                "metric": {"column": "value", "agg": "mean"},
                "subset_a": [{"column": "flag", "op": "eq", "value": 1}],
                "subset_b": [{"column": "flag", "op": "eq", "value": 0}],
            },
        )
    )
    shifted_calcs = _calcs(
        _execute(
            shifted,
            "compare_subsets",
            {
                "metric": {"column": "value", "agg": "mean"},
                "subset_a": [{"column": "flag", "op": "eq", "value": 1}],
                "subset_b": [{"column": "flag", "op": "eq", "value": 0}],
            },
        )
    )
    _close(shifted_calcs["absolute_delta"], base_calcs["absolute_delta"])
    for label in ("ci95_low_a", "ci95_low_b", "ci95_high_a", "ci95_high_b"):
        if label in base_calcs and label in shifted_calcs:
            _close(shifted_calcs[label], base_calcs[label] + SHIFT)
