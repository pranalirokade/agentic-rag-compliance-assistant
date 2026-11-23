"""Shared tokenisation: lowercase, stopword removal, light suffix stemming."""
import re

STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being this that
these those it its we our us you your they their there here do does did can could should would may might must
shall will what which who whom when where why how any all each per than then so not no into about over under
up out also such""".split())


def stem(w: str) -> str:
    for suf in ("ings", "ing", "edly", "ed", "ies", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    ws = re.findall(r"[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]", text.lower())
    return [stem(w) for w in ws if keep_stop or w not in STOP]
