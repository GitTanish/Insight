import numpy as np
import pandas as pd
import pytest

from insight.analytics.operations import (
    OPERATIONS,
    OperationError,
)
from insight.analytics.statistics import route_and_run, run_group_comparison


@pytest.fixture
def ab_df():
    rng = np.random.default_rng(7)
    n = 120
    flag = rng.integers(0, 2, n)
    return pd.DataFrame({
        "treated": flag,
        "spend": np.where(flag == 1, 100 + rng.normal(0, 10, n), 130 + rng.normal(0, 10, n)),
        "segment": rng.choice(["retail", "wholesale"], n),
        "score": rng.normal(50, 5, n),
    })


def test_binary_vs_numeric_routes_to_group_comparison(ab_df):
    route, outcome = route_and_run(ab_df, "treated", "spend")
    assert route == "group_comparison"
    assert outcome.test in ("Welch's two-sample t-test", "Mann-Whitney U")
    assert "%" in outcome.interpretation or "averages" in outcome.interpretation
    assert outcome.p_value is not None and outcome.p_value < 0.05


def test_small_sample_uses_mann_whitney():
    df = pd.DataFrame({
        "g": [0] * 8 + [1] * 8,
        "v": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24],
    })
    outcome = run_group_comparison(df, 'g', 'v')
    assert outcome.test == "Mann-Whitney U"


def test_categorical_pair_routes_to_chi_square():
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({
        "channel": rng.choice(["web", "store"], n, p=[0.7, 0.3]),
        "device": rng.choice(["ios", "android"], n, p=[0.7, 0.3]),
    })
    route, outcome = route_and_run(df, "channel", "device")
    assert route == "chi_square"
    assert outcome.test == "Chi-square independence"
    assert 0 <= outcome.effect_size <= 1


def test_continuous_pair_routes_to_correlation():
    rng = np.random.default_rng(11)
    x = rng.normal(size=200)
    df = pd.DataFrame({"a": x, "b": 2 * x + rng.normal(scale=0.1, size=200)})
    route, outcome = route_and_run(df, "a", "b")
    assert route == "correlation:pearson"
    assert abs(outcome.statistic) > 0.99


def test_binary_numeric_vs_categorical_routes_to_chi_square():
    rng = np.random.default_rng(31)
    n = 400
    df = pd.DataFrame({
        "device": rng.choice(["mobile", "desktop"], n, p=[0.6, 0.4]),
        "converted": rng.binomial(1, np.where(
            rng.choice(["mobile", "desktop"], n, p=[0.6, 0.4]) == "mobile", 0.3, 0.12
        )),
    })
    for col_a, col_b in (("device", "converted"), ("converted", "device")):
        route, outcome = route_and_run(df, col_a, col_b)
        assert route == "chi_square"
        assert outcome.p_value is not None
        assert 0 <= outcome.effect_size <= 1


def test_statistical_test_handles_string_column_order(ab_df):
    out = OPERATIONS["statistical_test"].run(
        ab_df,
        OPERATIONS["statistical_test"].params_model(column_a="spend", column_b="segment"),
        step_id=0,
    )
    labels = {c.label: c.value for c in out.calculations}
    assert labels["test"] in (
        "Welch's two-sample t-test",
        "Mann-Whitney U",
        "Chi-square independence",
    )


def test_chi_square_rejects_low_expected_counts():
    df = pd.DataFrame({
        "a": ["rare_a"] * 95 + ["b"] * 5,
        "c": ["x"] * 98 + ["y"] * 2,
    })
    with pytest.raises(ValueError):
        route_and_run(df, "a", "c")


def test_statistical_test_operation_registers(ab_df):
    out = OPERATIONS["statistical_test"].run(
        ab_df,
        OPERATIONS["statistical_test"].params_model(column_a="treated", column_b="spend"),
        step_id=0,
    )
    labels = {c.label: c.value for c in out.calculations}
    assert labels["test"] in ("Welch's two-sample t-test", "Mann-Whitney U")
    assert labels["p_value"] is not None and labels["p_value"] < 0.05
    assert "statistical_test" in str(out.notes) or "routed" in str(out.notes)


def test_adaptive_bins_respect_small_dataset():
    rng = np.random.default_rng(5)
    values = pd.Series(rng.normal(50, 10, 100))
    from insight.analytics.operations import _adaptive_bins

    k = _adaptive_bins(values, None)
    assert 5 <= k <= 30

    out = OPERATIONS["distribution"].run(
        pd.DataFrame({"v": values}),
        OPERATIONS["distribution"].params_model(column="v"),
        step_id=0,
    )
    counts = [r[1] for r in out.table.rows]
    assert sum(counts) == 100
    assert min(counts) >= max(5, int(np.ceil(100 * 0.05))) - 0 or len(counts) <= 4
    if any(c < 10 for c in counts):
        assert "n<10" in (out.notes or "")


def test_bin_labels_are_compact():
    import pandas as pd

    values = pd.Series([120.628 + i * 1.137 for i in range(80)])
    out = OPERATIONS["distribution"].run(
        pd.DataFrame({"v": values}),
        OPERATIONS["distribution"].params_model(column="v"),
        step_id=0,
    )
    for row in out.table.rows:
        label = row[0]
        assert len(label) <= 20, f"label too long: {label!r}"
        digits = "".join(ch for ch in label.split(",")[0].lstrip("([ ") if ch.isdigit())
        assert len(digits) <= 7, f"excess precision: {label!r}"


def test_correlation_excludes_binary_columns(ab_df):
    out = OPERATIONS["correlation"].run(
        ab_df, OPERATIONS["correlation"].params_model(), step_id=0
    )
    table = out.table.to_dataframe()
    assert "treated" not in table["column"].values
    excluded = next(c for c in out.calculations if c.label == "excluded_binary_columns")
    assert "treated" in (excluded.detail or "")
    assert "statistical_test" in (excluded.detail or "")


def test_title_sanitizer_fixes_top_n_claim(tmp_path=None):
    from insight.domain.analysis import ChartSpec, ChartType
    from insight.visualization.renderer import _sanitize_title

    df = pd.DataFrame({"region": ["North", "South"], "total": [10.0, 20.0]})
    spec = ChartSpec(chart_type=ChartType.bar, title="Top 3 Regions by Sales",
                     source_step=0, x="region", y="total")
    fixed = _sanitize_title(spec, df)
    assert "Top 3" not in fixed
    assert fixed.startswith(("Class Balance of region", "Distribution of"))


def test_other_bucket_folds_overflow(tmp_path=None):
    from insight.visualization.renderer import _with_other_bucket

    labels = pd.Series([f"c{i}" for i in range(15)])
    values = pd.Series([float(i + 1) for i in range(15)])
    kept_l, kept_v = _with_other_bucket(labels, values, 10)
    assert list(kept_v)[-1] == float(sum(range(11, 16)))
    assert kept_l.iloc[-1] == "Other"
    assert len(kept_v) == 11


def test_multi_group_numeric_routes_to_anova():
    rng = np.random.default_rng(21)
    n_per = 40
    df = pd.DataFrame({
        "plan": (["basic"] * n_per) + (["plus"] * n_per) + (["premium"] * n_per),
        "spend": np.concatenate([
            rng.normal(10, 2, n_per),
            rng.normal(14, 2, n_per),
            rng.normal(18, 2, n_per),
        ]),
    })
    route, outcome = route_and_run(df, "plan", "spend")
    assert route == "anova_test"
    assert outcome.test == "One-way ANOVA"
    assert outcome.p_value is not None and outcome.p_value < 0.05
    assert outcome.effect_size > 0.14
    assert "premium" in outcome.interpretation


def test_two_groups_still_route_to_group_comparison():
    rng = np.random.default_rng(22)
    n = 40
    df = pd.DataFrame({
        "tier": ["a"] * n + ["b"] * n,
        "score": np.concatenate([rng.normal(5, 1, n), rng.normal(7, 1, n)]),
    })
    route, outcome = route_and_run(df, "tier", "score")
    assert route == "group_comparison"


def test_anova_rejects_fewer_than_three_groups(ab_df):
    from insight.analytics.statistics import run_anova

    with pytest.raises(ValueError):
        run_anova(ab_df, "segment", "spend")


def test_anova_operation_registers_and_emits_table():
    rng = np.random.default_rng(23)
    rows = []
    means = {"east": 12.0, "west": 15.0, "central": 20.0}
    for region, mean in means.items():
        for value in rng.normal(mean, 2.0, 45):
            rows.append({"region": region, "delivery_days": value})
    df = pd.DataFrame(rows)

    out = OPERATIONS["anova_test"].run(
        df,
        OPERATIONS["anova_test"].params_model(group_column="region", value_column="delivery_days"),
        step_id=0,
    )
    table = out.table.to_dataframe()
    assert set(table["region"]) == {"east", "west", "central"}
    assert list(table["mean"]) == sorted(table["mean"], reverse=True)
    labels = {c.label: c.value for c in out.calculations}
    assert labels["test"] == "One-way ANOVA"
    assert labels["p_value"] < 0.05
    assert labels["effect_size"] >= 0.14
    assert "assumptions" in str(out.notes)


def test_regression_recovers_known_coefficients():
    rng = np.random.default_rng(42)
    n = 150
    x1 = rng.normal(0, 2, n)
    x2 = rng.normal(0, 1.5, n)
    y = 5 + 3 * x1 - 2 * x2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    out = OPERATIONS["regression"].run(
        df,
        OPERATIONS["regression"].params_model(target_column="y", feature_columns=["x1", "x2"]),
        step_id=0,
    )
    table = out.table.to_dataframe()
    coefs = dict(zip(table["term"], [float(v) for v in table["coefficient"]]))
    assert abs(coefs["intercept"] - 5) < 0.25
    assert abs(coefs["x1"] - 3) < 0.15
    assert abs(coefs["x2"] - (-2)) < 0.15

    labels = {c.label: c.value for c in out.calculations}
    assert labels["r_squared"] > 0.98
    assert labels["n_observations"] == n
    p_values = dict(zip(table["term"], [float(v) for v in table["p_value"]]))
    assert p_values["x1"] < 0.001 and p_values["x2"] < 0.001

    std_betas = dict(zip(table["term"], [float(v) for v in table["standardized_beta"]]))
    assert std_betas["x1"] > 0 and std_betas["x2"] < 0
    strongest = next(c for c in out.calculations if c.label == "strongest_predictor")
    predictor_betas = [v for term, v in std_betas.items() if term != "intercept"]
    assert abs(std_betas[strongest.value]) == max(abs(v) for v in predictor_betas)


def test_regression_rejects_insufficient_rows():
    df = pd.DataFrame({
        "x": [1.0, 2.0],
        "y": [3.0, 6.0],
    })
    with pytest.raises(OperationError):
        OPERATIONS["regression"].run(
            df,
            OPERATIONS["regression"].params_model(target_column="y", feature_columns=["x"]),
            step_id=0,
        )


def test_compare_subsets_adds_ci_for_large_means(ab_df):
    metric = {"column": "spend", "agg": "mean"}
    out = OPERATIONS["compare_subsets"].run(
        ab_df,
        OPERATIONS["compare_subsets"].params_model(
            metric=metric,
            subset_a=[{"column": "treated", "op": "eq", "value": 0}],
            subset_b=[{"column": "treated", "op": "eq", "value": 1}],
            label_a="control",
            label_b="treatment",
        ),
        step_id=0,
    )
    labels = {c.label for c in out.calculations}
    assert "ci95_low_control" in labels and "ci95_high_control" in labels
    assert "ci95_low_treatment" in labels and "ci95_high_treatment" in labels
    low = next(c for c in out.calculations if c.label == "ci95_low_control").value
    high = next(c for c in out.calculations if c.label == "ci95_high_control").value
    control_mean = next(c for c in out.calculations if c.label == "control").value
    assert low < float(control_mean) < float(high)
    assert "normal approximation" in (out.notes or "")


def test_compare_subsets_skips_ci_for_small_samples(sales_df):
    metric = {"column": "revenue", "agg": "mean"}
    out = OPERATIONS["compare_subsets"].run(
        sales_df.head(8),
        OPERATIONS["compare_subsets"].params_model(
            metric=metric,
            subset_a=[{"column": "region", "op": "eq", "value": "North"}],
            subset_b=[{"column": "region", "op": "eq", "value": "South"}],
        ),
        step_id=0,
    )
    labels = {c.label for c in out.calculations}
    assert not any(label.startswith("ci95_") for label in labels)
    assert "n>=30" in (out.notes or "")


def test_isotonic_fit_is_monotone_and_pools_violations():
    from insight.analytics.statistics import isotonic_fit

    increasing = isotonic_fit(np.array([0.1, 0.4, 0.2, 0.7, 0.9]))
    assert list(increasing) == sorted(increasing)
    assert abs(increasing[1] - 0.3) < 1e-9
    assert abs(increasing[2] - 0.3) < 1e-9

    decreasing = isotonic_fit(np.array([9.0, 7.0, 5.0, 3.0]))
    assert np.allclose(decreasing, 6.0)


def test_isotonic_calibration_detects_miscalibration():
    rng = np.random.default_rng(9)
    n = 800
    scores = rng.uniform(0.05, 0.95, n)
    inflated = 0.5 + 0.5 * scores
    outcomes = rng.binomial(1, scores)
    df = pd.DataFrame({"score": inflated, "converted": outcomes})

    out = OPERATIONS["isotonic_calibration"].run(
        df,
        OPERATIONS["isotonic_calibration"].params_model(
            score_column="score", outcome_column="converted"
        ),
        step_id=0,
    )
    table = out.table.to_dataframe()
    assert set(table.columns) == {"bin", "n", "avg_score", "observed_rate"}
    assert table["observed_rate"].iloc[-1] < table["avg_score"].iloc[-1]

    labels = {c.label: c.value for c in out.calculations}
    assert labels["brier_calibrated"] <= labels["brier_raw"] + 1e-6
    assert labels["ece"] > 0.02
    assert "over-confident" in labels["interpretation"]
    rates = [float(v) for v in table["observed_rate"]]
    rho = pd.Series(table["avg_score"]).corr(pd.Series(rates), method="spearman")
    assert rho > 0.95


def test_isotonic_calibration_accepts_well_calibrated_scores():
    rng = np.random.default_rng(10)
    n = 1000
    scores = rng.uniform(0.05, 0.95, n)
    outcomes = rng.binomial(1, scores)
    df = pd.DataFrame({"p": scores, "y": outcomes})

    out = OPERATIONS["isotonic_calibration"].run(
        df,
        OPERATIONS["isotonic_calibration"].params_model(score_column="p", outcome_column="y"),
        step_id=0,
    )
    labels = {c.label: c.value for c in out.calculations}
    assert labels["ece"] < 0.06
    assert "well calibrated" in labels["interpretation"]


def test_isotonic_calibration_rejects_non_binary_outcome():
    df = pd.DataFrame({
        "score": list(range(40)),
        "outcome": [i % 3 for i in range(40)],
    })
    with pytest.raises(OperationError):
        OPERATIONS["isotonic_calibration"].run(
            df,
            OPERATIONS["isotonic_calibration"].params_model(
                score_column="score", outcome_column="outcome"
            ),
            step_id=0,
        )


def test_isotonic_calibration_rejects_small_samples(ab_df):
    with pytest.raises(OperationError):
        OPERATIONS["isotonic_calibration"].run(
            ab_df.head(10),
            OPERATIONS["isotonic_calibration"].params_model(
                score_column="spend", outcome_column="treated"
            ),
            step_id=0,
        )


def test_group_aggregate_reports_mixed_count_bases():
    df = pd.DataFrame({
        "region": ["North", "North", "North", "South", "South"],
        "revenue": [100.0, np.nan, 300.0, 50.0, 70.0],
    })
    out = OPERATIONS["group_aggregate"].run(
        df,
        OPERATIONS["group_aggregate"].params_model(
            group_by="region",
            metrics=[
                {"column": "revenue", "agg": "count", "alias": "rows"},
                {"column": "revenue", "agg": "mean", "alias": "mean_revenue"},
            ],
        ),
        step_id=0,
    )
    table = out.table.to_dataframe()
    north = table[table["region"] == "North"].iloc[0]
    assert int(north["rows"]) == 3
    assert abs(float(north["mean_revenue"]) - 200.0) < 1e-9

    gap = next(c for c in out.calculations if c.label == "excluded_missing_mean_revenue")
    assert gap.value == 1
    assert "row_count" in gap.detail and "excluded" in gap.detail


def test_group_aggregate_no_gap_no_calc():
    df = pd.DataFrame({
        "region": ["A", "A", "B"],
        "revenue": [10.0, 20.0, 30.0],
    })
    out = OPERATIONS["group_aggregate"].run(
        df,
        OPERATIONS["group_aggregate"].params_model(
            group_by="region",
            metrics=[{"column": "revenue", "agg": "sum", "alias": "total"}],
        ),
        step_id=0,
    )
    assert not any(c.label.startswith("excluded_missing_") for c in out.calculations)
