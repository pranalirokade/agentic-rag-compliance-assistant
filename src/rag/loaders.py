"""Multi-format document loading: .md .txt .pdf .docx (pdf/docx need optional deps)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    source: str
    text: str
    meta: dict = field(default_factory=dict)


def _read_pdf(p: Path) -> str:
    from pypdf import PdfReader
    return "\n\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)


def _read_docx(p: Path) -> str:
    import docx
    return "\n\n".join(par.text for par in docx.Document(str(p)).paragraphs)


READERS = {
    ".md": lambda p: p.read_text(encoding="utf-8", errors="ignore"),
    ".txt": lambda p: p.read_text(encoding="utf-8", errors="ignore"),
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


def load_directory(root: str) -> list[Document]:
    docs = []
    for p in sorted(Path(root).rglob("*")):
        reader = READERS.get(p.suffix.lower())
        if p.is_file() and reader:
            text = reader(p).strip()
            if text:
                docs.append(Document(source=p.name, text=text, meta={"path": str(p)}))
    return docs
