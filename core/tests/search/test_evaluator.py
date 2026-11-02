import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def evaluator():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_pdf_search.py"
    spec = importlib.util.spec_from_file_location("pdf_search_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_span_offsets_preserve_original_ligature_text() -> None:
    module = evaluator()
    text = "The special classiﬁcation token is CLS."
    passage = SimpleNamespace(id="p1", text=text, location={"page": 1})
    report = SimpleNamespace(passages=[passage])
    labels = module.expected_spans(report, [{"id": "q1", "quote": "classification token"}])
    span = labels["q1"][0][0]
    assert text[span["start"] : span["end"]] == "classiﬁcation token"


def test_multiblock_evidence_rejects_a_hit_that_only_covers_half_the_answer() -> None:
    module = evaluator()
    report = SimpleNamespace(
        passages=[
            SimpleNamespace(id="p1", text="approximately 40", location={"page": 1}),
            SimpleNamespace(id="p2", text="epochs over the corpus", location={"page": 1}),
        ]
    )
    labels = module.expected_spans(
        report, [{"id": "q1", "quotes": ["approximately 40", "epochs over the corpus"]}]
    )

    def hit(identity, start=None, end=None):
        return SimpleNamespace(
            mappings=[SimpleNamespace(passage_id=identity, start_offset=start, end_offset=end)]
        )

    assert module.coverage([hit("p1")], labels["q1"], report) == (False, False)
    assert module.coverage([hit("p1"), hit("p2")], labels["q1"], report) == (True, True)
    assert module.coverage([hit("p1", 0, 5), hit("p2")], labels["q1"], report) == (True, False)
    with pytest.raises(ValueError, match="expected evidence not found"):
        module.expected_spans(report, [{"id": "wrong", "quote": "not in either passage"}])
