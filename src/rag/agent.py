"""LangGraph wiring over the node/edge definitions in pipeline.py."""
from __future__ import annotations

from typing import Any, TypedDict

from .pipeline import CONDITIONAL, EDGES, ENTRY, NODES, TERMINAL, Context, run_sequential


class AgentState(TypedDict, total=False):
    question: str
    rewritten: str
    candidates: list
    docs: list
    top_rel: float
    answer: str
    verification: Any
    strict: bool
    rewrites: int
    gens: int
    final: dict


def build_graph(ctx: Context):
    from langgraph.graph import END, START, StateGraph
    g = StateGraph(AgentState)
    for name, fn in NODES.items():
        g.add_node(name, lambda state, fn=fn: fn(state, ctx))
    g.add_edge(START, ENTRY)
    for src, dst in EDGES.items():
        g.add_edge(src, dst)
    for src, (router, targets) in CONDITIONAL.items():
        g.add_conditional_edges(src, lambda state, r=router: r(state, ctx), {t: t for t in targets})
    for t in TERMINAL:
        g.add_edge(t, END)
    return g.compile()


def ask(question: str, ctx: Context, use_langgraph: bool = True) -> dict:
    """Run the agent. Falls back to the sequential runner if langgraph isn't installed."""
    if use_langgraph:
        try:
            graph = build_graph(ctx)
        except ImportError:  # langgraph not installed
            graph = None
        if graph is not None:
            return graph.invoke({"question": question})["final"]
    return run_sequential(question, ctx)
