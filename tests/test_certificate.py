import json

import pytest

from insight.toolkit import AnalysisToolkit
from insight.verify_cli import main

SALES_CSV = (
    "region,revenue,units\n"
    "North,100,2\n"
    "South,40,1\n"
    "East,60,3\n"
    "North,50,2\n"
    "West,30,1\n"
)


@pytest.fixture
def tk():
    toolkit = AnalysisToolkit()
    toolkit.load_table("sales", content=SALES_CSV)
    return toolkit


def _group_agg(toolkit):
    return toolkit.execute_operation(
        "sales",
        "group_aggregate",
        {
            "group_by": "region",
            "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}],
        },
    )


def test_certificate_grounds_numbers_with_spans_and_evidence_ids(tk):
    run = _group_agg(tk)
    result = tk.verify_claims(
        "North leads with 150 in revenue across 5 rows.",
        run_id=run["run_id"],
    )
    certificate = result["certificate"]

    assert certificate["schema_version"] == "1.0"
    assert certificate["attestation_level"] == "recomputed"
    assert certificate["verdict"]["grounded"] is True
    assert certificate["verdict"]["recomputed"] == "pass"
    assert certificate["tables"][0]["name"] == "grouped"
    assert certificate["tables"][0]["rows"] == 4
    assert certificate["tables"][0]["digest"]

    grounded = [
        claim
        for claim in certificate["claims"]
        if claim["kind"] == "numeric" and claim["status"] == "grounded"
    ]
    assert grounded
    for claim in grounded:
        assert len(claim["span"]) == 2
        assert claim["evidence_ids"]
        assert claim["rule"] in {
            "exact_numeric_match",
            "percentage_scale_match",
            "verbatim_token_match",
        }


def test_certificate_flags_ungrounded_number(tk):
    run = _group_agg(tk)
    result = tk.verify_claims(
        "North leads with 999 in revenue.",
        run_id=run["run_id"],
    )
    certificate = result["certificate"]

    ungrounded = [
        claim
        for claim in certificate["claims"]
        if claim["kind"] == "numeric" and claim["status"] == "ungrounded"
    ]
    assert ungrounded
    assert ungrounded[0]["text"] == "999"
    assert certificate["checks"]["numeric_grounding"]["status"] == "fail"
    assert certificate["verdict"]["overall"] == "fail"
    assert result["ungrounded_numbers"] == ["999"]


def test_certificate_recomputation_detects_tampering(tk):
    run = _group_agg(tk)
    record = tk._runs[run["run_id"]]
    record.execution.steps[0].output.table.rows[0][1] = 123456

    result = tk.verify_claims(
        "North leads with 150 in revenue.",
        run_id=run["run_id"],
    )
    certificate = result["certificate"]

    check = certificate["checks"]["independent_recomputation"]
    assert check["status"] == "fail"
    assert check["issues"][0]["code"] == "RECOMPUTE_MISMATCH"
    assert certificate["verdict"]["overall"] == "fail"
    assert result["verified"] is False


def test_certificate_marks_covered_aggregates(tk):
    run = _group_agg(tk)
    result = tk.verify_claims("North leads with 150.", run_id=run["run_id"])
    aggregates = result["certificate"]["aggregates"]

    assert aggregates
    assert aggregates[0]["operation"] == "group_aggregate"
    assert aggregates[0]["recomputation"] == "verified"


def test_certificate_wires_small_sample_warning():
    tk = AnalysisToolkit()
    tk.load_table("tiny", content="group,value\na,1\nb,2\n")
    run = tk.execute_operation(
        "tiny",
        "group_aggregate",
        {
            "group_by": "group",
            "metrics": [{"column": "value", "agg": "count", "alias": "count"}],
        },
    )
    result = tk.verify_claims("Group a has 1 row.", run_id=run["run_id"])
    certificate = result["certificate"]

    check = certificate["checks"]["statistical_validity"]
    assert check["status"] == "warn"
    assert any(issue["code"] == "SMALL_SAMPLE" for issue in check["issues"])
    assert certificate["verdict"]["overall"] == "pass_with_warnings"


def test_certificate_flags_causal_language_without_design_marker(tk):
    run = _group_agg(tk)
    result = tk.verify_claims(
        "North's 150 in revenue drove the regional lead.",
        run_id=run["run_id"],
    )
    certificate = result["certificate"]

    causal = [claim for claim in certificate["claims"] if claim["kind"] == "causal"]
    assert causal
    assert causal[0]["status"] == "flagged"
    assert causal[0]["rule"] == "CAUSAL_LANGUAGE_WITHOUT_IDENTIFICATION"
    assert certificate["verdict"]["causal_overreach"] == "warn"
    assert certificate["verdict"]["overall"] == "pass_with_warnings"


def test_certificate_does_not_claim_sql_was_recomputed(tk):
    run = tk.run_sql(
        "SELECT region, SUM(revenue) AS total FROM sales GROUP BY region",
        tables=["sales"],
    )
    result = tk.verify_claims("North leads with 150.", run_id=run["run_id"])
    certificate = result["certificate"]

    check = certificate["checks"]["independent_recomputation"]
    assert check["status"] == "not_checked"
    assert check["checked"] == 0
    assert certificate["verdict"]["recomputed"] == "not_checked"
    assert all(
        aggregate["recomputation"] in {"not_covered", "not_checked"}
        for aggregate in certificate["aggregates"]
    )


def test_certificate_grounding_only_is_explicitly_labeled():
    tk = AnalysisToolkit()
    result = tk.verify_claims(
        "Revenue was 77.",
        evidence_numbers=[77],
    )
    certificate = result["certificate"]

    assert certificate["attestation_level"] == "grounding"
    assert certificate["checks"]["independent_recomputation"]["status"] == "not_checked"
    assert certificate["verdict"]["recomputed"] == "not_checked"
    assert certificate["verdict"]["overall"] == "pass_with_warnings"
    assert result["run_id"] is None


def test_certificate_is_json_serializable(tk):
    run = _group_agg(tk)
    result = tk.verify_claims("North leads with 150.", run_id=run["run_id"])
    json.dumps(result["certificate"])


def test_committed_json_schema_matches_model():
    from pathlib import Path

    from insight.verification import VerificationCertificate

    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "verification_certificate.schema.json"
    )
    assert json.loads(schema_path.read_text(encoding="utf-8")) == (
        VerificationCertificate.model_json_schema()
    )


def _grounded(answer, calcs):
    from insight.validation.grounding import ground_numeric_claims

    return ground_numeric_claims(answer, calcs, [])


def test_scientific_notation_is_one_numeric_token():
    from insight.domain.execution import Calculation

    claims = _grounded(
        "The result is significant at p=4.794e-16.",
        [Calculation(label="p_value", value=4.794e-16)],
    )
    numeric = [claim for claim in claims if claim.raw != "4.794e-16"]
    assert numeric == []
    claim = claims[0]
    assert claim.raw == "4.794e-16"
    assert claim.value == 4.794e-16
    assert claim.status == "grounded"
    assert claim.evidence_ids == ["calc:0"]
    assert claim.rule == "exact_numeric_match"


def test_tiny_values_do_not_ground_against_rounded_zero():
    from insight.domain.execution import Calculation

    claims = _grounded(
        "The result is significant at p=1e-9.",
        [Calculation(label="p_value", value=0.0)],
    )
    assert claims[0].status == "ungrounded"

    zero = _grounded(
        "The reported p-value is 0.0.",
        [Calculation(label="p_value", value=0.0)],
    )
    assert zero[0].status == "grounded"


def test_rounding_matches_only_at_the_claimed_precision():
    from insight.domain.execution import Calculation

    calcs = [
        Calculation(label="statistic", value=8.554537),
        Calculation(label="absolute_delta", value=-683.411797),
    ]
    claims = {
        claim.raw: claim
        for claim in _grounded("t=8.55, a difference of 683 g.", calcs)
    }
    assert claims["8.55"].status == "grounded"
    assert claims["8.55"].rule == "rounded_numeric_match"
    assert claims["683"].status == "grounded"
    assert claims["683"].rule == "rounded_magnitude_match"

    bad = _grounded("t=8.56", calcs)
    assert bad[0].status == "ungrounded"


def test_p_value_inequality_is_evaluated_against_p_value_evidence():
    from insight.domain.execution import Calculation

    calcs = [Calculation(label="p_value", value=4.794e-16)]

    grounded = _grounded("Significant at p < 0.001.", calcs)
    assert grounded[0].raw == "0.001"
    assert grounded[0].status == "grounded"
    assert grounded[0].rule == "inequality_threshold_match"
    assert grounded[0].evidence_ids == ["calc:0"]

    violated = _grounded("Significant at p > 0.05.", calcs)
    assert violated[0].status == "ungrounded"
    assert violated[0].reason == "inequality_threshold_violated"

    unchecked = _grounded(
        "Significant at p < 0.001.",
        [Calculation(label="statistic", value=1.0)],
    )
    assert unchecked[0].status == "not_checked"
    assert unchecked[0].reason == "inequality_threshold_unverified"


def test_unverified_threshold_makes_certificate_warn():
    tk = AnalysisToolkit()
    result = tk.verify_claims("Significant at p < 0.001.", evidence_numbers=[0.05])
    certificate = result["certificate"]

    assert certificate["checks"]["numeric_grounding"]["status"] == "warn"
    assert certificate["verdict"]["overall"] == "pass_with_warnings"
    assert result["grounded"] is True


def test_statistical_test_is_independently_recomputed():
    tk = AnalysisToolkit()
    tk.load_table(
        "groups",
        content=(
            "group,value\n"
            "a,1\na,2\na,3\na,4\na,5\n"
            "b,10\nb,11\nb,12\nb,13\nb,14\n"
        ),
    )
    run = tk.execute_operation(
        "groups",
        "statistical_test",
        {"column_a": "group", "column_b": "value"},
    )
    result = tk.verify_claims(
        "The difference is significant at p < 0.01.",
        run_id=run["run_id"],
    )
    certificate = result["certificate"]

    check = certificate["checks"]["independent_recomputation"]
    assert check["status"] == "pass"
    assert check["checked"] == 1
    assert certificate["verdict"]["recomputed"] == "pass"
    assert any(
        aggregate["recomputation"] == "verified"
        for aggregate in certificate["aggregates"]
    )
    assert all(
        aggregate["recomputation"] in {"verified", "not_checked"}
        for aggregate in certificate["aggregates"]
    )
    numeric = [
        claim
        for claim in certificate["claims"]
        if claim["kind"] == "numeric"
    ]
    assert numeric[0]["status"] == "grounded"

    record = tk._runs[run["run_id"]]
    for calc in record.execution.steps[0].output.calculations:
        if calc.label == "p_value":
            calc.value = 0.9
    tampered = tk.verify_claims(
        "The difference is significant at p < 0.01.",
        run_id=run["run_id"],
    )
    tampered_check = tampered["certificate"]["checks"]["independent_recomputation"]
    assert tampered_check["status"] == "fail"
    assert any(
        issue["code"] == "RECOMPUTE_MISMATCH"
        for issue in tampered_check["issues"]
    )
    aggregates = {
        aggregate["label"]: aggregate["recomputation"]
        for aggregate in tampered["certificate"]["aggregates"]
    }
    assert aggregates["p_value"] == "mismatch"
    assert aggregates["statistic"] == "verified"
    assert aggregates["interpretation"] == "not_checked"


def test_cli_evidence_mode(tmp_path, capsys):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"numbers": [77]}), encoding="utf-8")

    code = main(
        [
            "--answer",
            "Revenue was 77.",
            "--evidence",
            str(evidence),
            "--compact",
        ]
    )

    assert code == 0
    certificate = json.loads(capsys.readouterr().out)
    assert certificate["attestation_level"] == "grounding"
    assert certificate["verdict"]["grounded"] is True


def test_cli_fails_on_ungrounded_number(tmp_path, capsys):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"numbers": [77]}), encoding="utf-8")

    code = main(
        [
            "--answer",
            "Revenue was 999.",
            "--evidence",
            str(evidence),
            "--compact",
        ]
    )

    assert code == 1
    certificate = json.loads(capsys.readouterr().out)
    assert certificate["verdict"]["overall"] == "fail"


def test_cli_executes_operation(tmp_path, capsys):
    data = tmp_path / "sales.csv"
    data.write_text(SALES_CSV, encoding="utf-8")

    code = main(
        [
            "--answer",
            "North leads with 150.",
            "--data",
            str(data),
            "--operation",
            "group_aggregate",
            "--params",
            json.dumps(
                {
                    "group_by": "region",
                    "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}],
                }
            ),
            "--compact",
        ]
    )

    assert code == 0
    certificate = json.loads(capsys.readouterr().out)
    assert certificate["attestation_level"] == "recomputed"
    assert certificate["verdict"]["recomputed"] == "pass"
