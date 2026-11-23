## Retrieval eval (14 answerable questions; embedder=lsa, reranker=lexical, llm=extractive)

| Method | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| BM25 only | 0.857 | 0.929 | 1.0 | 0.911 |
| Dense only | 0.929 | 0.929 | 0.929 | 0.929 |
| Hybrid (RRF) | 1.0 | 1.0 | 1.0 | 1.0 |
| Hybrid + rerank | 0.857 | 1.0 | 1.0 | 0.929 |

**End-to-end:** grounded answers 12/14; answers containing the expected fact 12/14; correct abstentions on out-of-scope questions 3/3.
