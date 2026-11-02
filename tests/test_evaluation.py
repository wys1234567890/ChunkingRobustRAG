import json

import pytest

from scripts.evaluate_answers import main
from src.evaluation import evaluate_answers, score_answer


def query(qid):
    return {"query_id": qid, "question": f"Question {qid}?", "ground_truth": ["The Beatles", "Beatles"]}


def result(qid, answer="Beatles", status="success"):
    return {"query_id": qid, "question": f"Question {qid}?", "model_output": answer,
            "api_status": status, "retriever": "none"}


def test_normalization_and_best_reference():
    assert score_answer("THE Beatles!", ["Elvis", "Beatles"]) == {"em": 1, "token_f1": 1}
    assert score_answer("July 2, 1776", ["July 2 , 1776"])["em"] == 1
    assert score_answer("10", ["10"])["em"] == 1
    assert score_answer("", ["Beatles"])["token_f1"] == 0


def test_f1_counts_repeated_tokens_and_does_not_treat_substrings_as_answers():
    assert score_answer("red red blue", ["red blue blue"])["token_f1"] == pytest.approx(2 / 3)
    assert score_answer("The answer is Beatles", ["Beatles"])["em"] == 0
    assert score_answer("Unionized", ["Union"])["token_f1"] == 0


def test_errors_and_missing_are_zero_in_fixed_denominator():
    summary, details = evaluate_answers([query("1"), query("2"), query("3")],
                                        [result("1"), result("2", status="error")])
    assert summary["em"] == pytest.approx(1 / 3)
    assert summary["successful_only_em"] == 1
    assert summary["missing_queries"] == summary["error_queries"] == 1
    assert details[2]["api_status"] == "missing"


@pytest.mark.parametrize("records,match", [
    ([result("1"), result("1")], "duplicate"),
    ([result("2")], "unknown"),
    ([dict(result("1"), question="different")], "question mismatch"),
    ([dict(result("1"), api_status="unknown")], "invalid api_status"),
])
def test_invalid_alignment_is_rejected(records, match):
    with pytest.raises(ValueError, match=match):
        evaluate_answers([query("1")], records)


def test_mixed_runs_rejected():
    with pytest.raises(ValueError, match="mixed"):
        evaluate_answers([query("1"), query("2")], [result("1"), dict(result("2"), retriever="e5")])


def test_missing_references_rejected():
    with pytest.raises(ValueError, match="ground_truth"):
        evaluate_answers([dict(query("1"), ground_truth=[])], [])


def test_cli_writes_summary_and_details(tmp_path, capsys):
    queries = tmp_path / "queries.jsonl"
    results = tmp_path / "results.jsonl"
    summary = tmp_path / "summary.json"
    details = tmp_path / "details.jsonl"
    queries.write_text(json.dumps(query("1")) + "\n", encoding="utf-8")
    results.write_text(json.dumps(result("1")) + "\n", encoding="utf-8")
    assert main(["--queries", str(queries), "--results", str(results),
                 "--output", str(summary), "--details", str(details)]) == 0
    assert json.loads(summary.read_text(encoding="utf-8"))["em"] == 1
    assert json.loads(details.read_text(encoding="utf-8"))["query_id"] == "1"
    assert "EM=100.00%" in capsys.readouterr().out


def test_cli_cannot_overwrite_results(tmp_path):
    with pytest.raises(SystemExit):
        main(["--queries", str(tmp_path / "q"), "--results", str(tmp_path / "r"),
              "--output", str(tmp_path / "r")])
