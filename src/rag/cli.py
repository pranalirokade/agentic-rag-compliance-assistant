"""CLI:  python -m rag.cli ingest --docs data/sample_docs --out index
        python -m rag.cli ask "How long are financial records retained?" """
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .agent import ask
from .chunking import chunk_document
from .embeddings import make_embedder
from .index import HybridIndex
from .llm import make_llm
from .loaders import load_directory
from .pipeline import Config, Context
from .rerank import make_reranker


def build_index(docs_dir: str, embedder: str = "lsa", backend: str = "auto",
                chunk_size: int = 500, overlap: int = 60) -> HybridIndex:
    chunks = [c for d in load_directory(docs_dir) for c in chunk_document(d, chunk_size, overlap)]
    return HybridIndex(make_embedder(embedder), backend).build(chunks)


def load_context(index_dir: str, llm: str = "extractive", reranker: str = "lexical", cfg: Config | None = None):
    index = joblib.load(Path(index_dir) / "index.joblib")
    return Context(index, make_reranker(reranker, index), make_llm(llm), cfg or Config())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--docs", default="data/sample_docs"); i.add_argument("--out", default="index")
    i.add_argument("--embedder", default="lsa", choices=["lsa", "st"])
    i.add_argument("--backend", default="auto", choices=["auto", "numpy", "faiss", "qdrant"])
    a = sub.add_parser("ask")
    a.add_argument("question"); a.add_argument("--index", default="index")
    a.add_argument("--llm", default="extractive", choices=["extractive", "openai", "gemini"])
    a.add_argument("--reranker", default="lexical", choices=["lexical", "cross-encoder"])
    args = ap.parse_args()
    if args.cmd == "ingest":
        idx = build_index(args.docs, args.embedder, args.backend)
        Path(args.out).mkdir(exist_ok=True)
        joblib.dump(idx, Path(args.out) / "index.joblib")
        print(f"Indexed {len(idx.chunks)} chunks from {len({c.source for c in idx.chunks})} documents -> {args.out}/")
    else:
        print(json.dumps(ask(args.question, load_context(args.index, args.llm, args.reranker)), indent=2))


if __name__ == "__main__":
    main()
