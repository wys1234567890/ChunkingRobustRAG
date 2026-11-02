import json

import pytest

from scripts import run_experiment
from src.api_client import GenerationResult
from src.config import load_settings


def test_no_rag_cli_never_loads_chunks_or_retriever(tmp_path, monkeypatch):
    queries = tmp_path / "q.jsonl"
    output = tmp_path / "out.jsonl"
    queries.write_text(json.dumps({"query_id": "q1", "question": "Who?",
                                   "ground_truth": ["Beatles"], "dataset": "nq"}), encoding="utf-8")
    settings = load_settings(environ={"LLM_API_KEY": "fake", "LLM_MODEL": "fake"}, dotenv_path=None)
    monkeypatch.setattr(run_experiment, "load_settings", lambda _: settings)

    def forbidden(*args, **kwargs):
        raise AssertionError("no-RAG must not load retrieval dependencies")

    monkeypatch.setattr(run_experiment, "load_chunks", forbidden)
    monkeypatch.setattr(run_experiment, "build_retriever", forbidden)

    class FakeApi:
        def __init__(self, settings):
            self.settings = settings

        def generate(self, messages, *, request_id):
            return GenerationResult("Beatles", "success", "fake", request_id, False, 0, 1,
                                    8, 1, None, None)

    monkeypatch.setattr(run_experiment, "ApiClient", FakeApi)
    args = ["--no-rag", "--queries", str(queries), "--output", str(output)]
    assert run_experiment.main(args) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["dataset"] == "nq"
    assert record["prompt_version"] == "no-rag-v1"
    assert record["retriever"] == "none"
    assert run_experiment.main(args) == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
    output.write_text(json.dumps(dict(record, retriever="e5")), encoding="utf-8")
    with pytest.raises(SystemExit):
        run_experiment.main(args)


@pytest.mark.parametrize("options", [[], ["--no-rag", "--retriever", "e5"],
                                    ["--no-rag", "--chunks", "unused.jsonl"]])
def test_cli_rejects_conflicting_or_missing_mode(options):
    with pytest.raises(SystemExit):
        run_experiment.main(["--queries", "unused.jsonl", "--output", "unused-out.jsonl", *options])
