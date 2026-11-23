# Enterprise Multi-Agent Knowledge & Compliance Assistant (Agentic RAG)

An **agentic Retrieval-Augmented Generation** system for policy / SOP / audit documents:
heading-aware chunking → **hybrid dense + BM25 retrieval (RRF)** → **cross-encoder re-ranking** →
a **LangGraph agent** that grades relevance, rewrites weak queries, verifies citations, and
**abstains** instead of guessing.

```
question ─► retrieve (dense ⊕ BM25 → RRF) ─► rerank ─► relevance OK? ──no──► rewrite query ─► (retry once) ─► abstain
                                                          │yes
                                                          ▼
                                   generate (cited answer) ─► verify citations ─► ok? ──► finalize
                                                          ▲                          │no
                                                          └──── tighten prompt ◄─────┘  (once, then abstain)
```

## What is in the box
| Component | File | Notes |
|---|---|---|
| Multi-format loaders | `loaders.py` | `.md .txt .pdf .docx` |
| Recursive, heading-aware chunking | `chunking.py` | keeps section titles for citations |
| Hybrid retrieval | `index.py`, `bm25.py` | dense (FAISS / numpy / **Qdrant**) + BM25, fused with Reciprocal Rank Fusion |
| Re-ranking | `rerank.py` | `CrossEncoderReranker` (MS MARCO MiniLM) or offline `LexicalReranker` |
| Agent graph | `pipeline.py`, `agent.py` | routing logic defined once; runs on **LangGraph** or a dependency-free runner |
| Citation verification | `verify.py` | cited chunk must exist, support the claim, and contain every number the claim states |
| LLM adapters | `llm.py` | OpenAI, Gemini, or offline `ExtractiveLLM` for tests |
| Evaluation | `eval/` | retrieval Hit@k/MRR + abstention (offline); RAGAS faithfulness/context-recall (needs API key) |

## Run it
```bash
pip install -r requirements.txt          # offline demo + tests
make ingest && make test && make eval
python -m rag.cli ask "How long are financial records retained?"
```
Production stack (real embeddings, cross-encoder, LangGraph, FAISS/Qdrant, OpenAI/Gemini, RAGAS):
```bash
pip install -r requirements-full.txt
export OPENAI_API_KEY=...            # or GEMINI_API_KEY
python -m rag.cli ingest --embedder st --backend faiss
python -m rag.cli ask "..." --llm openai --reranker cross-encoder
python eval/eval_ragas.py            # RAGAS faithfulness + context recall
```

## Evaluation (honest scope)
The corpus is **6 short fictional policy documents (24 chunks)** and the golden set has 14 answerable +
3 out-of-scope questions (`eval/golden.jsonl`). Result of `make eval` in *offline mode*
(LSA embeddings, lexical reranker, extractive LLM — no API keys):

| Method | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| BM25 only | 0.857 | 0.929 | 1.0 | 0.911 |
| Dense only | 0.929 | 0.929 | 0.929 | 0.929 |
| Hybrid (RRF) | 1.0 | 1.0 | 1.0 | 1.0 |
| Hybrid + lexical rerank | 0.857 | 1.0 | 1.0 | 0.929 |

End-to-end: 12/14 grounded answers, 3/3 correct abstentions on out-of-scope questions.

**Read these numbers with care:** 14 questions over 24 chunks is a smoke test, not a benchmark; the
lexical reranker is a fallback and *hurts* Hit@1 here. To measure the real stack, run
`eval/eval_ragas.py` with your own corpus, a cross-encoder and an LLM, and record your own
faithfulness / context-recall numbers (and the hallucination-rate change versus a no-verification
baseline) below. The pipeline supports scaling to large corpora (batch embedding + FAISS/Qdrant),
but this repo ships only the small sample corpus.

<!-- YOUR OWN RAGAS RESULTS HERE -->

## Design decisions
- **Hybrid > either alone:** BM25 nails exact terms (`AES-256`, `SOC 2 Type II`); dense handles paraphrase. RRF fuses ranks without score calibration.
- **Abstain by design:** unanswerable questions are a first-class test case; the agent returns "insufficient support" rather than a plausible guess.
- **Numeric guard:** in compliance, a silently changed number (7 → 90 days) is the worst failure, so verification requires numbers to appear verbatim in the cited chunk.
- **Limits:** the offline verifier is lexical. For paraphrase-level hallucinations add an LLM/NLI judge.

MIT licensed.
