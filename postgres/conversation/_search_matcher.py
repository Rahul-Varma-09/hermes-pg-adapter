"""The store's search query language and its matching rules, evaluated over candidate rows.

The sanitizer, the query grammar (adjacent phrases, ``NOT``, ``AND``, ``OR``), word matching (letters, digits and
marks; case and Latin diacritics folded; ``_`` splits words), substring matching (one-to-one case folding), the LIKE
scans (ASCII case folding, text up to the first NUL) and the snippet writers, as ``persistence.conversation.search``
specifies them. The repository narrows rows with SQL filters first; these rules decide which candidates match.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from persistence.conversation.search import MAX_SEARCH_QUERY_CHARS

_TRIGRAM_EXCLUDED_SOURCES = ("cron", "subagent")
_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x3000, 0x303F), (0x3040, 0x309F), (0x30A0, 0x30FF), (0xAC00, 0xD7AF))
_OPERATORS = frozenset({"AND", "OR", "NOT"})
_SPECIAL_RE = re.compile(r"""[+{}():"^@/#&|~\[\]<>,;!?$=\\']""")
_QUOTED_RE = re.compile(r'"[^"]*"')
_UNIT_RE = re.compile(r'"[^"]+"|\S+')
_SNIPPET_TOKENS = 40


def _is_cjk(char: str) -> bool:
    return any(lo <= ord(char) <= hi for lo, hi in _CJK_RANGES)


def _count_cjk(text: str) -> int:
    return sum(1 for char in text if _is_cjk(char))


def _non_operator_tokens(text: str) -> list[str]:
    return [token for token in text.split() if token.upper() not in _OPERATORS]


def _quote_tokens(text: str) -> str:
    """Each non-operator word quoted, the operator words kept."""
    return " ".join(t if t.upper() in _OPERATORS else '"' + t.replace('"', '""') + '"' for t in text.split())


def _strip_cjk_wildcards(text: str) -> str:
    """Trailing ``*`` dropped from each non-operator word unless it is all stars (``_strip_cjk_wildcards``)."""
    if "*" not in text:
        return text
    return " ".join(t if t.upper() in _OPERATORS else (t.rstrip("*") or t) for t in text.split()) or text


def sanitize_query(query: str) -> str:
    """The store's query sanitizer (``SessionSearchMixin._sanitize_fts5_query``)."""
    query = query[:MAX_SEARCH_QUERY_CHARS]
    held: list[str] = []

    def _hold(match: re.Match[str]) -> str:
        held.append(match.group(0))
        return f"\x00Q{len(held) - 1}\x00"

    text = _QUOTED_RE.sub(_hold, query).replace('"', " ")
    text = _SPECIAL_RE.sub(" ", text)
    if "%" in text and not any(_is_cjk(char) for char in text):
        text = text.replace("%", " ")
    text = re.sub(r"\*+", "*", text)
    text = re.sub(r"(^|\s)\*", r"\1", text)
    text = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", text.strip())
    text = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", text.strip())
    text = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', text)
    for index, quoted in enumerate(held):
        text = text.replace(f"\x00Q{index}\x00", quoted)
    return text.strip()


def _or_relaxed(query: str) -> Optional[str]:
    """The implicit-AND query with its units ORed, or None for fewer than two units or an explicit OR / NOT (``_or_relaxed_query``)."""
    units: list[str] = []
    for token in _UNIT_RE.findall(query):
        if token.upper() in {"OR", "NOT"}:
            return None
        if token.upper() != "AND":
            units.append(token)
    return " OR ".join(units) if len(units) >= 2 else None


def _word_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in "LNM" or category == "Co"


def _fold_word(word: str) -> str:
    """One word as the word index stores it: combining marks dropped (also from characters that decompose into one), lowercased."""
    out = []
    for char in word:
        parts = unicodedata.normalize("NFD", char)
        if any(unicodedata.category(part) == "Mn" for part in parts):
            char = "".join(part for part in parts if unicodedata.category(part) != "Mn")
        out.append(char)
    return "".join(out).lower()


def words(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, folded)`` for each word of ``text`` (maximal runs of letters, digits and marks)."""
    out: list[tuple[int, int, str]] = []
    start: Optional[int] = None
    for index, char in enumerate(text + " "):
        if index < len(text) and _word_char(char):
            start = index if start is None else start
        elif start is not None:
            out.append((start, index, _fold_word(text[start:index])))
            start = None
    return [item for item in out if item[2]]


def fold_char(char: str) -> str:
    """Case folding one character to one character, as the substring index does."""
    mapped = char.casefold()
    if len(mapped) != 1:
        mapped = char.lower()
    return mapped if len(mapped) == 1 else char


def fold_chars(text: str) -> str:
    return "".join(fold_char(char) for char in text)


def like_text(text: Optional[str]) -> str:
    """A stored text as the LIKE scans read it: up to its first NUL, empty for NULL."""
    return (text or "").split("\x00", 1)[0]


def _ascii_lower(value: str) -> str:
    return re.sub(r"[A-Z]+", lambda match: match.group().lower(), value)


def like_contains(text: Optional[str], term: str) -> bool:
    """SQL ``LIKE '%term%'``: ASCII case folding only."""
    return _ascii_lower(term) in _ascii_lower(like_text(text))


class QuerySyntaxError(ValueError):
    """Sanitized text the store's query grammar still rejects."""


@dataclass(frozen=True)
class _Phrase:
    text: str
    prefix: bool


@dataclass(frozen=True)
class _Node:
    op: str
    left: Any
    right: Any


def _bareword_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in "_\x1a") or not char.isascii()


def _lex(text: str) -> list[Any]:
    """Phrases and operator words of a sanitized query; any other character is a syntax error."""
    out: list[Any] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in " \t\n\r":
            i += 1
            continue
        if char == '"':
            end, parts = i + 1, []
            while True:
                close = text.find('"', end)
                if close < 0:
                    raise QuerySyntaxError("unterminated phrase")
                parts.append(text[end:close])
                if text.startswith('""', close):
                    parts.append('"')
                    end = close + 2
                    continue
                i = close + 1
                break
            phrase = "".join(parts)
        elif _bareword_char(char):
            start = i
            while i < len(text) and _bareword_char(text[i]):
                i += 1
            phrase = text[start:i]
            if phrase in _OPERATORS and not text.startswith("*", i):
                out.append(phrase)
                continue
        else:
            raise QuerySyntaxError(f"unexpected {char!r}")
        prefix = text.startswith("*", i)
        i += 1 if prefix else 0
        out.append(_Phrase(phrase, prefix))
    return out


def parse_query(text: str) -> Any:
    """The query tree: adjacent phrases bind tightest, then ``NOT``, ``AND``, ``OR``; all left-associative."""
    items = _lex(text)
    position = 0

    def level(operator: Optional[str]) -> Any:
        nonlocal position
        if operator is None:
            node = None
            while position < len(items) and isinstance(items[position], _Phrase):
                node = items[position] if node is None else _Node("AND", node, items[position])
                position += 1
            if node is None:
                raise QuerySyntaxError("operand expected")
            return node
        lower = {"OR": "AND", "AND": "NOT", "NOT": None}[operator]
        node = level(lower)
        while position < len(items) and items[position] == operator:
            position += 1
            node = _Node(operator, node, level(lower))
        return node

    if not items:
        raise QuerySyntaxError("empty query")
    tree = level("OR")
    if position != len(items):
        raise QuerySyntaxError("operator out of place")
    return tree


def _phrase_words(phrase: _Phrase) -> list[str]:
    return [word for _, _, word in words(phrase.text)]


def _word_hits(phrase: _Phrase, text: str) -> list[tuple[int, int]]:
    """Character spans of ``phrase`` in ``text`` under word matching (the last word as a prefix with ``*``)."""
    needle = _phrase_words(phrase)
    found = words(text)
    spans: list[tuple[int, int]] = []
    for i in range(len(found) - len(needle) + 1 if needle else 0):
        run = [word for _, _, word in found[i : i + len(needle)]]
        last_ok = run[-1].startswith(needle[-1]) if phrase.prefix else run[-1] == needle[-1]
        if run[:-1] == needle[:-1] and last_ok:
            spans.append((found[i][0], found[i + len(needle) - 1][1]))
    return spans


def _substring_hits(phrase: _Phrase, text: str) -> list[tuple[int, int]]:
    """Spans of ``phrase`` in ``text`` as a case-folded substring; a phrase under three characters matches without a span."""
    if len(phrase.text) < 3:
        return [(0, 0)]
    needle, haystack = fold_chars(phrase.text), fold_chars(text)
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle), haystack)]


def matched_phrases(tree: Any, fields: Sequence[str], hits: Callable[[_Phrase, str], list[tuple[int, int]]]) -> list[_Phrase]:
    """The phrases of the branches of ``tree`` a row with searchable columns ``fields`` matches (empty when the row does not match); a phrase matches inside one column."""
    if isinstance(tree, _Phrase):
        return [tree] if any(hits(tree, text) for text in fields) else []
    left = matched_phrases(tree.left, fields, hits)
    if tree.op == "NOT":
        return left if left and not matched_phrases(tree.right, fields, hits) else []
    right = matched_phrases(tree.right, fields, hits)
    if tree.op == "AND":
        return left + right if left and right else []
    return left + right


def _marked(text: str, spans: Sequence[tuple[int, int]], start: int, end: int) -> str:
    """``text[start:end]`` with every span inside it (overlaps merged) between ``>>>`` and ``<<<``; the store's snippet writer drops plain text from a NUL to the next match."""
    merged: list[list[int]] = []
    for lo, hi in sorted(span for span in spans if span[0] < span[1] and span[0] >= start and span[1] <= end):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    parts, cursor = [], start
    for lo, hi in merged:
        parts += [text[cursor:lo].split("\x00", 1)[0], ">>>", text[lo:hi], "<<<"]
        cursor = hi
    parts.append(text[cursor:end].split("\x00", 1)[0])
    return "".join(parts)


def word_snippet(columns: Sequence[str], phrases: Sequence[_Phrase]) -> str:
    """An excerpt of the first column holding a match of ``phrases``: up to 40 words around the first one, matches marked, ``...`` where cut."""
    for text in columns:
        spans = [span for phrase in phrases for span in _word_hits(phrase, text)]
        if not spans:
            continue
        found = words(text)
        if len(found) <= _SNIPPET_TOKENS:
            return _marked(text, spans, 0, len(text))
        first = next(i for i, item in enumerate(found) if item[0] == min(spans)[0])
        lo = min(max(0, first - (_SNIPPET_TOKENS - 1) // 2), len(found) - _SNIPPET_TOKENS)
        hi = lo + _SNIPPET_TOKENS
        start = found[lo][0] if lo else 0
        end = found[hi - 1][1] if hi < len(found) else len(text)
        return ("..." if lo else "") + _marked(text, spans, start, end) + ("..." if hi < len(found) else "")
    return columns[0] if columns else ""


def substring_snippet(columns: Sequence[str], phrases: Sequence[_Phrase]) -> str:
    """The first column holding a substring match of ``phrases``, matches marked; long text cut to about 40 characters around the first."""
    for text in columns:
        spans = [span for phrase in phrases for span in _substring_hits(phrase, text) if span[1] > span[0]]
        if not spans:
            continue
        if len(text) <= _SNIPPET_TOKENS + 2:
            return _marked(text, spans, 0, len(text))
        lo = max(0, min(min(spans)[0] - _SNIPPET_TOKENS // 2, len(text) - _SNIPPET_TOKENS))
        hi = lo + _SNIPPET_TOKENS
        return ("..." if lo else "") + _marked(text, spans, lo, hi) + ("..." if hi < len(text) else "")
    return columns[0] if columns else ""
def like_snippet(content: Optional[str], term: str) -> Optional[str]:
    """``substr(content, max(1, instr(content, term) - 40), 120)``."""
    if content is None:
        return None
    start = max(1, content.find(term) + 1 - 40)
    return content[start - 1 : start - 1 + 120]


def like_boolean(query: str) -> tuple[list[list[tuple[str, bool]]], Optional[str]]:
    """The ``tool`` role route's predicate: OR groups of (word, negated), groups without a positive word dropped, and the snippet word (``_compile_like_boolean_query``)."""
    groups: list[list[tuple[str, bool]]] = [[]]
    negate = False
    for token in _UNIT_RE.findall(query):
        operator = token.upper()
        if operator == "OR":
            if groups[-1]:
                groups.append([])
            negate = False
        elif operator == "NOT":
            negate = True
        elif operator not in {"AND", "NEAR"}:
            term = token.strip('"').strip("*").strip()
            if term:
                groups[-1].append((term, negate))
                negate = False
    kept = [group for group in groups if any(not negated for _, negated in group)]
    first = next((term for group in kept for term, negated in group if not negated), None)
    return kept, first
