"""RAGAS faithfulness / context-recall using a real LLM judge (needs an API key).

    pip install ragas datasets langchain-openai
    export OPENAI_API_KEY=...
    PYTHONPATH=src python eval/eval_ragas.py --llm openai --embedder st --reranker cross-encoder

NOTE: RAGAS changes its API between releases; this was written for the `ragas` 0.2.x
`evaluate(Dataset, metrics=[...])` interface. Pin the version in requirements if it breaks.
Ground-truth answers are the `must_contain` key phrases, so context_recall here is a coarse proxy.
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, faithfulness

from rag.agent import ask
from rag.cli import build_index
from rag.llm import make_llm
from rag.pipeline import Config, Context
from rag.rerank import make_reranker

ap = argparse.ArgumentParser()
ap.add_argument("--llm", default="openai"); ap.add_argument("--embedder", default="st")
ap.add_argument("--reranker", default="cross-encoder"); ap.add_argument("--docs", default="data/sample_docs")
args = ap.parse_args()

index = build_index(args.docs, args.embedder)
ctx = Context(index, make_reranker(args.reranker, index), make_llm(args.llm), Config())
rows = [json.loads(l) for l in Path("eval/golden.jsonl").read_text().splitlines() if l.strip()]
data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
for r in filter(lambda r: r["answerable"], rows):
    cands = [c for c, _ in index.search(r["question"], 20)]
    top = [c.text for c, _ in ctx.reranker.rerank(r["question"], cands, ctx.cfg.top_k)]
    data["question"].append(r["question"]); data["contexts"].append(top)
    data["answer"].append(ask(r["question"], ctx)["answer"]); data["ground_truth"].append(r["must_contain"])
print(evaluate(Dataset.from_dict(data), metrics=[faithfulness, context_recall]))
