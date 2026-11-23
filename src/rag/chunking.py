"""Heading-aware recursive chunking with overlap."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Document

SEPS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    id: str
    source: str
    section: str
    text: str


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i:i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    out, cur = [], ""
    for part in text.split(sep):
        cand = part if not cur else cur + sep + part
        if len(cand) <= size:
            cur = cand
            continue
        if cur:
            out.append(cur)
        if len(part) > size:
            out.extend(_split(part, size, rest))
            cur = ""
        else:
            cur = part
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, size: int = 500, overlap: int = 60) -> list[str]:
    base = [c.strip() for c in _split(text, size, SEPS) if c.strip()]
    if overlap <= 0 or len(base) < 2:
        return base
    out = [base[0]]
    for prev, cur in zip(base, base[1:]):
        tail = prev[-overlap:]
        tail = tail[tail.find(" ") + 1:] if " " in tail else ""
        out.append((tail + " " + cur).strip())
    return out


def chunk_document(doc: Document, size: int = 500, overlap: int = 60) -> list[Chunk]:
    """Split on markdown headings first so every chunk keeps its section title."""
    sections, title, buf = [], "Introduction", []
    for line in doc.text.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                sections.append((title, "\n".join(buf)))
            title, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((title, "\n".join(buf)))
    chunks = []
    for title, body in sections:
        for j, piece in enumerate(chunk_text(body, size, overlap)):
            chunks.append(Chunk(f"{doc.source}::{len(chunks)}", doc.source, title, piece))
    return chunks
