"""LLM adapters. `ExtractiveLLM` needs no API and is used for tests/offline demos; it selects
and cites sentences from context rather than generating text."""
from __future__ import annotations

import os
import re

from .text import STOP, tokens
from .verify import split_sentences

INSUFFICIENT = "INSUFFICIENT_CONTEXT"

ANSWER_PROMPT = """You are a compliance assistant. Answer ONLY from the numbered context below.
Cite the supporting context number like [1] after every sentence. Copy numbers, durations and
thresholds exactly. If the context does not contain the answer, reply exactly: {flag}
{strict}
Context:
{context}

Question: {question}
Answer:"""

REWRITE_PROMPT = ("Rewrite this question as a concise keyword search query for a policy document "
                  "search engine. Return only the query.\nQuestion: {q}")


def format_context(chunks) -> str:
    return "\n".join(f"[{i}] (source: {c.source} > {c.section}) {c.text}" for i, c in enumerate(chunks, 1))


class ChatLLM:
    """Base class: subclasses implement `_complete(prompt) -> str`."""

    def _complete(self, prompt: str) -> str:
        raise NotImplementedError

    def answer(self, question: str, chunks, strict: bool = False) -> str:
        strict_txt = "Be extremely literal: do not add any fact that is not stated in the context." if strict else ""
        return self._complete(ANSWER_PROMPT.format(
            flag=INSUFFICIENT, strict=strict_txt, context=format_context(chunks), question=question)).strip()

    def rewrite(self, question: str) -> str:
        return self._complete(REWRITE_PROMPT.format(q=question)).strip()


class OpenAILLM(ChatLLM):
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client, self.model = OpenAI(), model

    def _complete(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""


class GeminiLLM(ChatLLM):
    def __init__(self, model: str = "gemini-2.0-flash"):
        from google import genai
        self.client, self.model = genai.Client(api_key=os.environ["GEMINI_API_KEY"]), model

    def _complete(self, prompt: str) -> str:
        return self.client.models.generate_content(model=self.model, contents=prompt).text or ""


class ExtractiveLLM:
    """Offline stand-in: returns the best-matching sentence(s) from context with citations."""

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def answer(self, question: str, chunks, strict: bool = False) -> str:
        q = set(tokens(question))
        scored = []
        for i, c in enumerate(chunks, 1):
            for s in split_sentences(c.text.replace("\n", " ")):
                s = re.sub(r"^[-*\d.\s]+", "", s)
                if len(tokens(s)) >= 3:
                    scored.append((len(q & set(tokens(s))) / (len(q) or 1), i, s))
        scored.sort(key=lambda x: -x[0])
        picked = [t for t in scored[: self.max_sentences] if t[0] > 0]
        if not picked:
            return INSUFFICIENT
        return " ".join(f"{s.rstrip('.')} [{i}]." for _, i, s in picked)

    def rewrite(self, question: str) -> str:
        return " ".join(w for w in re.findall(r"[A-Za-z0-9\-]+", question) if w.lower() not in STOP)


def make_llm(kind: str):
    return {"openai": OpenAILLM, "gemini": GeminiLLM}.get(kind, ExtractiveLLM)()
