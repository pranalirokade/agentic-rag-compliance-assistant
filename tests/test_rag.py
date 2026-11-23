from rag.bm25 import BM25
from rag.chunking import chunk_document, chunk_text
from rag.cli import build_index
from rag.index import rrf
from rag.llm import ExtractiveLLM
from rag.loaders import Document
from rag.pipeline import Config, Context, run_sequential
from rag.rerank import LexicalReranker
from rag.verify import verify_answer

DOCS = "data/sample_docs"


def _ctx(**cfg):
    idx = build_index(DOCS)
    return Context(idx, LexicalReranker(idx.bm25), ExtractiveLLM(), Config(**cfg))


def test_chunking_respects_size_and_headings():
    doc = Document("x.md", "# A\n" + "word " * 400 + "\n# B\nshort section")
    chunks = chunk_document(doc, size=300, overlap=30)
    assert all(len(c.text) <= 340 for c in chunks)
    assert {c.section for c in chunks} == {"A", "B"}
    assert len(chunk_text("tiny", 300, 30)) == 1


def test_bm25_ranks_exact_term_first():
    bm = BM25(["encryption at rest uses aes", "lunch menu for friday", "password length policy"])
    s = bm.scores("aes encryption")
    assert s.index(max(s)) == 0


def test_rrf_prefers_items_ranked_by_both_lists():
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    assert [i for i, _ in fused][:2] == [1, 3]


def test_verify_catches_fabricated_number_and_bad_citation():
    ctx = ["Critical vulnerabilities must be patched within 7 days of release."]
    good = verify_answer("Critical vulnerabilities must be patched within 7 days [1].", ctx)
    assert good.ok
    wrong = verify_answer("Critical vulnerabilities must be patched within 90 days quarterly [1].", ctx)
    assert not wrong.ok
    assert not verify_answer("Patching takes 7 days [3].", ctx).ok
    assert not verify_answer("Critical vulnerabilities are patched fast.", ctx).ok  # uncited


def test_end_to_end_answers_with_citation():
    out = run_sequential("How long are financial records retained?", _ctx())
    assert out["grounded"] and "8 years" in out["answer"]
    assert out["citations"][0]["source"] == "data_retention_policy.md"
    assert out["trace"][-1] == "finalize"


def test_out_of_scope_question_abstains():
    out = run_sequential("What is the company parental leave policy?", _ctx())
    assert out["grounded"] is False and "abstain" == out["trace"][-1]


def test_low_relevance_triggers_one_rewrite_before_abstaining():
    out = run_sequential("What is the CEO salary?", _ctx())
    assert out["trace"].count("rewrite") <= 1 and out["trace"][-1] == "abstain"


def test_regeneration_then_abstain_when_answer_never_verifies():
    class Liar(ExtractiveLLM):
        def answer(self, q, chunks, strict=False):
            return "Records are kept for 99 years across seven galaxies [1]."
    c = _ctx(); c.llm = Liar()
    out = run_sequential("How long are financial records retained?", c)
    assert out["grounded"] is False and out["trace"].count("generate") == 2
