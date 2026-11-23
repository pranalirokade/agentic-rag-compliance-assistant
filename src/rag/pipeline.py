"""Agent nodes and routing. Defined once; executed either by LangGraph (agent.py) or by the
dependency-free `run_sequential` below, so routing logic is unit-testable offline."""
from __future__ import annotations

from dataclasses import dataclass, field

from .llm import INSUFFICIENT
from .verify import CITE, verify_answer

ABSTAIN_MSG = "I couldn't find sufficient support for that in the indexed documents."


@dataclass
class Config:
    candidates: int = 20          # fused candidates sent to the reranker
    top_k: int = 4                # chunks given to the LLM
    min_relevance: float = 0.45   # rerank score needed to attempt an answer
    max_rewrites: int = 1
    max_regens: int = 1
    min_support: float = 0.6      # citation-verification threshold
    mode: str = "hybrid"          # hybrid | dense | bm25


@dataclass
class Context:
    index: object
    reranker: object
    llm: object
    cfg: Config = field(default_factory=Config)


def n_retrieve(s, c):
    q = s.get("rewritten") or s["question"]
    return {"candidates": [h for h, _ in c.index.search(q, c.cfg.candidates, c.cfg.mode)]}


def n_rerank(s, c):
    q = s.get("rewritten") or s["question"]
    docs = c.reranker.rerank(q, s["candidates"], c.cfg.top_k)
    return {"docs": docs, "top_rel": docs[0][1] if docs else 0.0}


def n_rewrite(s, c):
    return {"rewritten": c.llm.rewrite(s["question"]), "rewrites": s.get("rewrites", 0) + 1}


def n_tighten(s, c):
    return {"strict": True}


def n_generate(s, c):
    chunks = [d for d, _ in s["docs"]]
    ans = c.llm.answer(s["question"], chunks, strict=s.get("strict", False))
    return {"answer": ans, "gens": s.get("gens", 0) + 1}


def n_verify(s, c):
    texts = [d.text for d, _ in s["docs"]]
    if INSUFFICIENT in s["answer"]:
        return {"verification": None}
    return {"verification": verify_answer(s["answer"], texts, c.cfg.min_support)}


def n_finalize(s, c):
    used = {int(n) for n in CITE.findall(s["answer"])}
    cites = [{"n": i, "source": d.source, "section": d.section, "relevance": round(r, 3)}
             for i, (d, r) in enumerate(s["docs"], 1) if i in used]
    return {"final": {"answer": s["answer"], "citations": cites, "grounded": True}}


def n_abstain(s, c):
    return {"final": {"answer": ABSTAIN_MSG, "citations": [], "grounded": False,
                      "reason": (s.get("verification").problems if s.get("verification") else "low relevance or no answer")}}


def r_after_rerank(s, c):
    if s["top_rel"] >= c.cfg.min_relevance:
        return "generate"
    return "rewrite" if s.get("rewrites", 0) < c.cfg.max_rewrites else "abstain"


def r_after_verify(s, c):
    v = s.get("verification")
    if v is None:
        return "abstain"
    if v.ok:
        return "finalize"
    return "tighten" if s.get("gens", 0) <= c.cfg.max_regens else "abstain"


NODES = {"retrieve": n_retrieve, "rerank": n_rerank, "rewrite": n_rewrite, "tighten": n_tighten,
         "generate": n_generate, "verify": n_verify, "finalize": n_finalize, "abstain": n_abstain}
EDGES = {"retrieve": "rerank", "rewrite": "retrieve", "tighten": "generate", "generate": "verify"}
CONDITIONAL = {"rerank": (r_after_rerank, ["generate", "rewrite", "abstain"]),
               "verify": (r_after_verify, ["finalize", "tighten", "abstain"])}
ENTRY, TERMINAL = "retrieve", {"finalize", "abstain"}


def run_sequential(question: str, ctx: Context, max_steps: int = 20) -> dict:
    state, node, trace = {"question": question}, ENTRY, []
    for _ in range(max_steps):
        state.update(NODES[node](state, ctx))
        trace.append(node)
        if node in TERMINAL:
            break
        node = CONDITIONAL[node][0](state, ctx) if node in CONDITIONAL else EDGES[node]
    out = dict(state["final"])
    out["trace"] = trace
    return out
