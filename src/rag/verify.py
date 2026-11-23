"""Citation verification: every claim must cite a real context chunk that supports it.

This offline check is lexical (content-word coverage) plus a strict numeric-consistency rule. It
catches fabricated citations and numbers that don't appear in the cited chunk; for subtler paraphrase errors, add an LLM/NLI
judge on top (see README)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import tokens

CITE = re.compile(r"\[(\d+)\]")
NUM = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass
class Verification:
    ok: bool
    problems: list[str] = field(default_factory=list)
    support: list[float] = field(default_factory=list)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[\d])|\n+", text) if s.strip()]


def verify_answer(answer: str, context_texts: list[str], min_support: float = 0.6) -> Verification:
    """`context_texts[i]` is the chunk cited as [i+1]."""
    problems, supports = [], []
    for sent in split_sentences(answer):
        ids = [int(x) for x in CITE.findall(sent)]
        body = tokens(CITE.sub("", sent))
        if len(body) < 3:
            continue
        if not ids:
            problems.append(f"uncited claim: {sent[:80]}")
            continue
        bad = [i for i in ids if not 1 <= i <= len(context_texts)]
        if bad:
            problems.append(f"invalid citation {bad} in: {sent[:80]}")
            continue
        claim = set(body)
        best = max(len(claim & set(tokens(context_texts[i - 1]))) / len(claim) for i in ids)
        supports.append(best)
        # Numbers, durations and thresholds must appear verbatim in a cited chunk.
        cited_nums = {n for i in ids for n in NUM.findall(context_texts[i - 1])}
        missing = {n for n in NUM.findall(CITE.sub("", sent))} - cited_nums
        if missing:
            problems.append(f"number(s) {sorted(missing)} not found in cited context: {sent[:80]}")
        elif best < min_support:
            problems.append(f"weak support ({best:.2f}) for: {sent[:80]}")
    return Verification(ok=not problems and bool(supports), problems=problems, support=supports)
