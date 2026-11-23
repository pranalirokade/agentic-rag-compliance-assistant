"""Reproducible retrieval + abstention evaluation on eval/golden.jsonl (no API keys needed).

    PYTHONPATH=src python eval/eval_retrieval.py [--embedder st]

A retrieved chunk counts as a hit when it comes from the expected source AND contains the
expected key phrase. Reports Hit@k and MRR for bm25 / dense / hybrid, plus end-to-end
answer-grounding and abstention behaviour of the full agent.
"""
import argparse
import json
from pathlib import Path

from rag.agent import ask
from rag.cli import build_index
from rag.llm import make_llm
from rag.pipeline import Config, Context
from rag.rerank import make_reranker

ap = argparse.ArgumentParser()
ap.add_argument("--embedder", default="lsa"); ap.add_argument("--reranker", default="lexical")
ap.add_argument("--llm", default="extractive"); ap.add_argument("--docs", default="data/sample_docs")
ap.add_argument("--golden", default="eval/golden.jsonl"); ap.add_argument("--out", default="eval/results.md")
args = ap.parse_args()

rows = [json.loads(l) for l in Path(args.golden).read_text().splitlines() if l.strip()]
answerable = [r for r in rows if r["answerable"]]
unanswerable = [r for r in rows if not r["answerable"]]
index = build_index(args.docs, args.embedder)
hit = lambda chunk, r: chunk.source == r["source"] and r["must_contain"] in chunk.text.lower()


def retrieval_metrics(mode: str, rerank: bool = False):
    h1 = h3 = h5 = mrr = 0.0
    rr = make_reranker(args.reranker, index)
    for r in answerable:
        cands = [c for c, _ in index.search(r["question"], 20, mode)]
        ranked = [c for c, _ in rr.rerank(r["question"], cands, 5)] if rerank else cands[:5]
        pos = next((i for i, c in enumerate(ranked, 1) if hit(c, r)), None)
        h1 += pos == 1; h3 += bool(pos and pos <= 3); h5 += bool(pos and pos <= 5); mrr += 1 / pos if pos else 0
    n = len(answerable)
    return [round(x / n, 3) for x in (h1, h3, h5, mrr)]


table = [("BM25 only", *retrieval_metrics("bm25")), ("Dense only", *retrieval_metrics("dense")),
         ("Hybrid (RRF)", *retrieval_metrics("hybrid")), ("Hybrid + rerank", *retrieval_metrics("hybrid", True))]

ctx = Context(index, make_reranker(args.reranker, index), make_llm(args.llm), Config())
grounded = correct = 0
for r in answerable:
    out = ask(r["question"], ctx, use_langgraph=False)
    grounded += out["grounded"]
    correct += out["grounded"] and r["must_contain"] in out["answer"].lower()
abstained = sum(not ask(r["question"], ctx, use_langgraph=False)["grounded"] for r in unanswerable)

md = [f"## Retrieval eval ({len(answerable)} answerable questions; embedder={args.embedder}, "
      f"reranker={args.reranker}, llm={args.llm})\n",
      "| Method | Hit@1 | Hit@3 | Hit@5 | MRR |", "|---|---|---|---|---|"]
md += [f"| {n} | {a} | {b} | {c} | {d} |" for n, a, b, c, d in table]
md += ["", f"**End-to-end:** grounded answers {grounded}/{len(answerable)}; answers containing the expected "
       f"fact {correct}/{len(answerable)}; correct abstentions on out-of-scope questions "
       f"{abstained}/{len(unanswerable)}.", ""]
Path(args.out).write_text("\n".join(md))
print("\n".join(md))
