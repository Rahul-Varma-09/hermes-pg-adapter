"""SQLite dialect to Postgres, one statement at a time.

The translator tokenizes a statement (string literals, quoted identifiers, comments, numbers, words, parameters and
punctuation), rewrites the SQLite constructs Hermes' file stores use, and returns Postgres text with psycopg ``%s`` /
``%(name)s`` placeholders. String literals and quoted identifiers are never rewritten.

Rewrites: ``?`` / ``:name`` / ``@name`` / ``$name`` parameters; ``INSERT OR IGNORE`` and ``INSERT OR REPLACE`` /
``REPLACE INTO`` (as ``ON CONFLICT``); ``a IS b`` / ``a IS NOT b`` with a non-literal right side (null-safe
comparison); ``LIKE`` (ASCII case-insensitive in SQLite: ``ILIKE``); ``ifnull``; multi-argument ``max`` / ``min``
(``greatest`` / ``least``); ``julianday`` / ``datetime`` / ``date`` / ``time`` / ``strftime`` / ``unixepoch`` (the
compat schema's functions); ``LIMIT -1``; ``ORDER BY`` terms get SQLite's NULL placement; DDL types (``INTEGER`` to
``bigint``, ``REAL`` to ``double precision``, ``BLOB`` to ``bytea``, an ``INTEGER PRIMARY KEY`` to an identity),
``AUTOINCREMENT`` / ``WITHOUT ROWID`` / ``STRICT`` dropped, ``CURRENT_TIMESTAMP`` defaults as SQLite's text;
backtick and bracket identifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<comment>--[^\n]*|/\*.*?\*/)
  | (?P<str>'(?:[^']|'')*')
  | (?P<qid>"(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]*\])
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|0[xX][0-9a-fA-F]+)
  | (?P<param>\?\d*|[:@$][A-Za-z_][A-Za-z0-9_]*)
  | (?P<word>[A-Za-z_][A-Za-z0-9_$]*)
  | (?P<op><=|>=|<>|!=|==|\|\||<<|>>|::|[-+*/%<>=(),;.&|~])
""", re.VERBOSE | re.DOTALL)

_CONSTRAINT_WORDS = {"NOT", "NULL", "PRIMARY", "UNIQUE", "DEFAULT", "CHECK", "REFERENCES", "COLLATE", "CONSTRAINT",
                     "GENERATED", "AS"}
_TYPE_MAP = {"INTEGER": "bigint", "INT": "bigint", "TINYINT": "bigint", "SMALLINT": "bigint", "MEDIUMINT": "bigint",
             "BIGINT": "bigint", "UNSIGNED": "", "INT2": "bigint", "INT8": "bigint", "BOOLEAN": "bigint", "BOOL": "bigint",
             "REAL": "double precision", "DOUBLE": "double precision", "FLOAT": "double precision",
             "NUMERIC": "double precision", "DECIMAL": "double precision", "BLOB": "bytea", "DATETIME": "text",
             "TIMESTAMP": "text", "DATE": "text", "JSON": "text", "VARCHAR": "text", "CHAR": "text", "CLOB": "text"}
_TIME_FUNCTIONS = {"JULIANDAY": "hermes_julianday", "DATETIME": "hermes_datetime", "DATE": "hermes_date",
                   "TIME": "hermes_time", "STRFTIME": "hermes_strftime", "UNIXEPOCH": "hermes_unixepoch"}
# SQLite functions without a same-named Postgres equivalent, served by the compat schema's ``hermes_*`` functions.
_HELPER_FUNCTIONS = {"INSTR", "JSON_EXTRACT", "JSON_VALID", "TYPEOF"}
_CLAUSE_END = {"LIMIT", "OFFSET", "UNION", "EXCEPT", "INTERSECT", "FOR", "RETURNING", "WINDOW", "ON", "DO"}


@dataclass
class Token:
    kind: str
    text: str

    @property
    def upper(self) -> str:
        return self.text.upper() if self.kind == "word" else self.text


@dataclass
class Statement:
    """One translated statement: its Postgres text, its kind (the first keyword) and what the connection must do
    around it."""
    sql: str
    kind: str
    table: Optional[str] = None        # the object a CREATE/ALTER/DROP/INSERT names
    tbl_name: Optional[str] = None     # the table it belongs to (an index's table; a table's own name)
    ddl: Optional[str] = None          # the original SQLite text of a CREATE, kept for sqlite_master
    insert_or_replace: bool = False    # INSERT OR REPLACE / REPLACE INTO: the conflict target is the table's key
    replace_columns: Optional[list[str]] = None
    pragma: Optional[tuple[str, Optional[str]]] = None
    rename: Optional[tuple[str, str]] = None
    add_column: Optional[str] = None
    update_or: Optional[str] = None    # UPDATE OR IGNORE: the connection updates row by row, skipping conflicts


def tokenize(sql: str) -> list[Token]:
    tokens = []
    pos = 0
    while pos < len(sql):
        match = _TOKEN.match(sql, pos)
        if match is None:
            raise ValueError(f"cannot tokenize SQL near {sql[pos:pos + 40]!r}")
        tokens.append(Token(match.lastgroup or "op", match.group()))
        pos = match.end()
    return tokens


def split_statements(sql: str) -> list[str]:
    """The statements of a script, split at top-level semicolons (a trigger body's semicolons stay inside it)."""
    out, current, depth = [], [], 0
    for token in tokenize(sql):
        if token.kind == "word" and token.upper == "BEGIN" and any(t.upper == "TRIGGER" for t in current if t.kind == "word"):
            depth += 1
        elif token.kind == "word" and token.upper == "END" and depth:
            depth -= 1
        if token.text == ";" and depth == 0:
            text = "".join(t.text for t in current).strip()
            if text:
                out.append(text)
            current = []
            continue
        current.append(token)
    text = "".join(t.text for t in current).strip()
    if text and any(t.kind not in ("ws", "comment") for t in current):
        out.append(text)
    return out


def _identifier(token: Token) -> str:
    """A quoted identifier in Postgres form; a word as is."""
    if token.kind != "qid":
        return token.text
    text = token.text
    if text[0] == "`":
        body = text[1:-1].replace("``", "`")
    elif text[0] == "[":
        body = text[1:-1]
    else:
        return text
    return '"' + body.replace('"', '""') + '"'


def bare_name(token: Token) -> str:
    """An identifier's name (quotes removed, case kept for quoted names, lowered for words)."""
    if token.kind == "qid":
        text = token.text
        return text[1:-1].replace('""', '"').replace("``", "`") if text[0] in '"`' else text[1:-1]
    return token.text.lower()


def _significant(tokens: list[Token]) -> list[int]:
    return [i for i, t in enumerate(tokens) if t.kind not in ("ws", "comment")]


def translate(sql: str, *, param_style: Callable[[Token], str]) -> Statement:
    """Translate one SQLite statement. ``param_style`` renders each parameter token (``%s`` or ``%(name)s``)."""
    tokens = tokenize(sql)
    sig = _significant(tokens)
    if not sig:
        return Statement("", "EMPTY")
    words = [tokens[i].upper for i in sig]
    kind = words[0]
    stmt = Statement("", kind)

    if kind == "PRAGMA":
        return _pragma(tokens, sig, stmt)
    if kind in ("BEGIN", "COMMIT", "END", "ROLLBACK", "SAVEPOINT", "RELEASE"):
        stmt.sql = " ".join(w for w in words if w not in ("DEFERRED", "IMMEDIATE", "EXCLUSIVE", "TRANSACTION"))
        if kind == "END":
            stmt.kind, stmt.sql = "COMMIT", "COMMIT"
        return stmt
    if kind in ("VACUUM", "ANALYZE", "REINDEX", "DETACH", "ATTACH"):
        stmt.kind = "NOOP"
        return stmt

    # Words SQLite takes as names but Postgres reserves (a ``grant`` column) are quoted, as the lowercase name.
    out = [Token(t.kind, _identifier(t)) if t.kind == "qid" else
           Token("qid", f'"{t.text.lower()}"') if t.kind == "word" and t.upper in _PG_ONLY_RESERVED else
           Token(t.kind, t.text) for t in tokens]
    is_ddl = kind in ("CREATE", "ALTER")

    if kind == "CREATE":
        _create(out, sig, stmt, sql)
    elif kind == "ALTER":
        _alter(out, sig, stmt)
        if stmt.kind == "ALTER ADD":
            # The column's own SQLite text, which sqlite_master's CREATE TABLE gains as SQLite's does.
            add = next(n for n, w in enumerate(words) if w == "ADD")
            first = sig[add + 2] if words[add + 1:add + 2] == ["COLUMN"] else sig[add + 1]
            stmt.ddl = "".join(t.text for t in tokens[first:]).strip().rstrip(";").strip()
    elif kind == "DROP":
        target = next((i for i in sig[1:] if out[i].upper not in ("TABLE", "INDEX", "VIEW", "TRIGGER", "IF", "EXISTS")), None)
        if target is not None:
            stmt.table = bare_name(out[target])
    elif kind in ("INSERT", "REPLACE"):
        _insert(out, sig, stmt)
    elif kind == "UPDATE" and words[1:2] == ["OR"] and len(sig) > 3:
        # UPDATE OR <conflict mode>: the mode goes; IGNORE is carried out by the connection (connection._update_or).
        stmt.update_or = words[2]
        out[sig[1]] = out[sig[2]] = Token("ws", "")
        stmt.table = bare_name(out[sig[3]])
        where = next((i for n, i in enumerate(sig) if words[n] == "WHERE" and _depth_at(words, n) == 0), None)
        out.insert(where if where is not None else len(out), Token("ws", " /*hermes_where*/ "))

    _expressions(out, is_ddl)
    _typed_null_tests(out)
    _truthiness(out)
    _order_by(out)
    rendered = []
    for token in out:
        if token.kind == "param":
            rendered.append(param_style(token))
        elif token.kind == "str":
            rendered.append(token.text.replace("%", "%%"))
        elif token.kind == "comment":
            rendered.append(" ")
        elif token.text == "%":
            rendered.append("%%")
        else:
            rendered.append(token.text)
    stmt.sql = "".join(rendered).strip().rstrip(";")
    return stmt


def _pragma(tokens: list[Token], sig: list[int], stmt: Statement) -> Statement:
    rest = [tokens[i] for i in sig[1:]]
    name = rest[0].text.lower() if rest else ""
    if len(rest) > 2 and rest[1].text == ".":  # schema-qualified pragma
        name = rest[2].text.lower()
        rest = rest[2:]
    argument = None
    if len(rest) >= 3 and rest[1].text in ("(", "="):
        argument = bare_name(rest[2]) if rest[2].kind in ("word", "qid") else rest[2].text.strip("'")
    stmt.kind, stmt.pragma = "PRAGMA", (name, argument)
    return stmt


def _create(out: list[Token], sig: list[int], stmt: Statement, original: str) -> None:
    words = [out[i].upper for i in sig]
    stmt.ddl = original.strip().rstrip(";")
    obj = next((w for w in words[1:4] if w in ("TABLE", "INDEX", "VIEW", "TRIGGER")), None)
    stmt.kind = f"CREATE {obj}" if obj else "CREATE"
    names = [i for i in sig[1:] if out[i].upper not in ("UNIQUE", "TEMP", "TEMPORARY", "TABLE", "INDEX", "VIEW",
                                                         "TRIGGER", "IF", "NOT", "EXISTS")]
    if names:
        stmt.table = stmt.tbl_name = bare_name(out[names[0]])
    if obj == "INDEX":
        on = next((n for n, i in enumerate(sig) if out[i].upper == "ON"), None)
        if on is not None and on + 1 < len(sig):
            stmt.tbl_name = bare_name(out[sig[on + 1]])
    if obj == "TABLE":
        close, alias = _table_body(out, sig)
        if close is not None:
            # SQLite's implicit rowid, which queries may name: the INTEGER PRIMARY KEY's value, else its own counter.
            # Result rows leave it out unless a query asks for it (connection.Cursor).
            rowid = (f'rowid bigint GENERATED ALWAYS AS ("{alias}") STORED' if alias
                     else "rowid bigint GENERATED BY DEFAULT AS IDENTITY")
            out[close] = Token(out[close].kind, f", {rowid})")
            # Table options after the body: WITHOUT ROWID, STRICT.
            for i in sig:
                if i > close and out[i].upper in ("WITHOUT", "ROWID", "STRICT"):
                    out[i] = Token("ws", "")
                elif i > close and out[i].text == "," :
                    out[i] = Token("ws", "")


def _table_body(out: list[Token], sig: list[int]) -> tuple[Optional[int], Optional[str]]:
    """Map the column definitions of a CREATE TABLE body; returns the index of its closing paren and the name of its
    ``INTEGER PRIMARY KEY`` column, if any."""
    opening = next((n for n, i in enumerate(sig) if out[i].text == "("), None)
    if opening is None:
        return None, None
    depth, segment, alias = 0, [], None
    for n in range(opening, len(sig)):
        token = out[sig[n]]
        if token.text == "(":
            depth += 1
            if depth == 1:
                continue
        elif token.text == ")":
            depth -= 1
            if depth == 0:
                return sig[n], _column_definition(out, segment) or alias
        if depth == 1 and token.text == ",":
            alias = _column_definition(out, segment) or alias
            segment = []
            continue
        segment.append(sig[n])
    return None, None


def _column_definition(out: list[Token], segment: list[int]) -> Optional[str]:
    """Map one column definition (``name type constraints``); a table constraint is left alone. Returns the column's
    name when it is an ``INTEGER PRIMARY KEY``, SQLite's alias of the rowid."""
    if not segment or out[segment[0]].upper in ("CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN"):
        return None
    words = [out[i].upper for i in segment]
    for n, i in enumerate(segment):
        if words[n] == "AUTOINCREMENT":
            out[i] = Token("ws", "")
        elif words[n] == "CURRENT_TIMESTAMP" and n and words[n - 1] == "DEFAULT":
            out[i] = Token("word", "(to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'))")
    if len(segment) < 2 or words[1] in _CONSTRAINT_WORDS:
        # SQLite takes a column without a type (any value); Postgres needs one, and text holds what such tables keep.
        out[segment[0]] = Token(out[segment[0]].kind, out[segment[0]].text + " text")
        return None
    if out[segment[1]].kind != "word" or words[1] not in _TYPE_MAP:
        return None
    mapped = _TYPE_MAP[words[1]]
    alias = None
    if words[1] in ("INTEGER", "INT") and words[2:4] == ["PRIMARY", "KEY"]:
        mapped = "bigint GENERATED BY DEFAULT AS IDENTITY"
        alias = bare_name(out[segment[0]])
    out[segment[1]] = Token("word", mapped)
    # A type's size, as in VARCHAR(64), goes with it; a second type word (UNSIGNED BIG INT) too.
    n = 2
    while n < len(segment) and words[n] in ("UNSIGNED", "BIG", "INT", "PRECISION", "VARYING"):
        out[segment[n]] = Token("ws", "")
        n += 1
    if n < len(segment) and out[segment[n]].text == "(":
        depth = 0
        for i in segment[n:]:
            if out[i].text == "(":
                depth += 1
            elif out[i].text == ")":
                depth -= 1
            out[i] = Token("ws", "")
            if depth == 0:
                break
    return alias


def _alter(out: list[Token], sig: list[int], stmt: Statement) -> None:
    words = [out[i].upper for i in sig]
    if len(sig) > 2:
        stmt.table = stmt.tbl_name = bare_name(out[sig[2]])
    if "RENAME" in words and "TO" in words and "COLUMN" not in words:
        to = words.index("TO")
        stmt.kind, stmt.rename = "ALTER RENAME", (stmt.table or "", bare_name(out[sig[to + 1]]))
    elif "ADD" in words:
        add = words.index("ADD")
        column = add + 1 + (1 if words[add + 1:add + 2] == ["COLUMN"] else 0)
        stmt.kind, stmt.add_column = "ALTER ADD", bare_name(out[sig[column]])
        _column_definition(out, sig[column:])


def _insert(out: list[Token], sig: list[int], stmt: Statement) -> None:
    words = [out[i].upper for i in sig]
    replace = words[0] == "REPLACE" or words[1:3] == ["OR", "REPLACE"]
    ignore = words[1:3] == ["OR", "IGNORE"]
    if words[0] == "REPLACE":
        out[sig[0]] = Token("word", "INSERT")
    if words[1] == "OR":
        out[sig[1]] = Token("ws", "")
        out[sig[2]] = Token("ws", "")
    into = words.index("INTO") if "INTO" in words else None
    positional = None
    if into is not None:
        stmt.table = bare_name(out[sig[into + 1]])
        after = into + 2
        if after < len(sig) and out[sig[after]].text == "(" and words[after + 1:after + 2] != ["SELECT"]:
            cols, j = [], after + 1
            while j < len(sig) and out[sig[j]].text != ")":
                if out[sig[j]].text != ",":
                    cols.append(bare_name(out[sig[j]]))
                j += 1
            stmt.replace_columns = cols
        elif after < len(sig) and words[after] == "VALUES":
            positional = sig[after]
    stmt.kind = "INSERT"
    if into is not None:
        _qualify_upsert(out, sig, words, out[sig[into + 1]].text)
    if ignore or replace:
        stmt.insert_or_replace = replace
        returning = next((i for i in sig if out[i].upper == "RETURNING"), None)
        marker = Token("conflict", " ON CONFLICT DO NOTHING " if ignore else " ON CONFLICT {target} DO UPDATE SET {set} ")
        if returning is not None:
            out.insert(returning, marker)
        else:
            out.append(marker)
    if positional is not None:
        # Positional VALUES name their columns, filled in by the connection: a hidden rowid column may sit before
        # columns added later, where SQLite's position counting would not see it. Inserted last: it shifts positions.
        out.insert(positional, Token("columns", " {columns} "))


_SQL_WORDS = {"CASE", "WHEN", "THEN", "ELSE", "END", "AND", "OR", "NOT", "NULL", "IS", "IN", "BETWEEN", "LIKE", "ILIKE",
              "GLOB", "TRUE", "FALSE", "DISTINCT", "FROM", "AS", "COLLATE", "ESCAPE", "EXISTS", "WHERE", "SET", "DEFAULT",
              "CURRENT_TIMESTAMP", "CURRENT_DATE", "CURRENT_TIME"}


_PG_ONLY_RESERVED = {"GRANT", "USER", "ARRAY", "BOTH", "LEADING", "TRAILING", "ONLY", "SYMMETRIC", "ASYMMETRIC",
                     "VARIADIC", "LATERAL", "PLACING", "ANALYSE", "CURRENT_USER", "SESSION_USER", "CURRENT_ROLE",
                     "LOCALTIME", "LOCALTIMESTAMP", "FETCH", "TABLESAMPLE", "SIMILAR", "VERBOSE", "FREEZE"}
_SEGMENT_END = {"AND", "OR", "THEN", "ORDER", "GROUP", "LIMIT", "OFFSET", "HAVING", "WINDOW", "UNION", "EXCEPT",
                "INTERSECT", "RETURNING", "ELSE", "END", "ON", "DO", "WHERE", "WHEN"}
_PREDICATE_WORDS = {"IS", "IN", "LIKE", "ILIKE", "BETWEEN", "EXISTS", "GLOB", "MATCH", "REGEXP", "TRUE", "FALSE", "NOT"}
_PREDICATE_OPS = {"=", "==", "<", ">", "<=", ">=", "<>", "!=", "#~#"}


def _typed_null_tests(out: list[Token]) -> None:
    """``? IS NULL``: Postgres cannot type a parameter that only meets a NULL test (SQLite needs no type), so it is
    tested as text, which any value is."""
    sig = _significant(out)
    words = [out[i].upper for i in sig]
    for n in reversed(range(len(sig))):
        if out[sig[n]].kind == "param" and (words[n + 1:n + 3] == ["IS", "NULL"] or words[n + 1:n + 4] == ["IS", "NOT", "NULL"]):
            out.insert(sig[n] + 1, Token("op", "::text"))


def _truthiness(out: list[Token]) -> None:
    """SQLite takes any value where a condition goes (``WHERE flag``, ``AND 1``, ``CASE WHEN json_valid(x)``) and
    Postgres only booleans: each operand there without a comparison of its own becomes ``hermes_truthy(...)``."""
    sig = _significant(out)
    words = [out[i].upper for i in sig]
    # Which ANDs belong to a BETWEEN, and which WHENs open a searched CASE (a simple CASE's are values).
    between_and, searched_when, pending, cases = set(), set(), {}, []
    depth = 0
    for n, word in enumerate(words):
        if word == "(":
            depth += 1
        elif word == ")":
            pending.pop(depth, None)
            depth -= 1
        elif word == "BETWEEN":
            pending[depth] = True
        elif word == "AND" and pending.pop(depth, False):
            between_and.add(n)
        elif word == "CASE":
            cases.append(words[n + 1:n + 2] == ["WHEN"])
        elif word == "END" and cases:
            cases.pop()
        elif word == "WHEN" and cases and cases[-1]:
            searched_when.add(n)
    joined = "JOIN" in words
    starts = []
    for n, word in enumerate(words):
        if word in ("WHERE", "HAVING", "OR") or (word == "AND" and n not in between_and) or n in searched_when:
            starts.append(n + 1)
        elif word == "ON" and joined and words[n + 1:n + 2] != ["CONFLICT"]:
            starts.append(n + 1)
        elif word == "NOT" and n and (words[n - 1] in ("WHERE", "HAVING", "AND", "OR", "NOT", "(") or n - 1 in searched_when):
            starts.append(n + 1)
    inserts: list[tuple[int, str]] = []
    while starts:
        start = starts.pop()
        end, depth, predicate = start, 0, False
        while end < len(sig):
            word = words[end]
            if word == "(":
                depth += 1
            elif word == ")":
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0 and (word in (",", ";") or (word in _SEGMENT_END and end not in between_and)):
                break
            elif depth == 0 and (word.split(" ")[0] in _PREDICATE_WORDS or word in _PREDICATE_OPS):
                predicate = True
            end += 1
        if start >= end or predicate or words[start] in ("NOT", "EXISTS", "NULL"):
            continue
        if words[start] == "(" and _closing(words, start) == end - 1:
            if words[start + 1:start + 2] != ["SELECT"]:
                starts.append(start + 1)  # a parenthesized condition: its first operand is one too
            continue
        inserts += [(sig[start], "hermes_truthy("), (sig[end - 1] + 1, ")")]
    for index, text in sorted(inserts, key=lambda item: item[0], reverse=True):
        out.insert(index, Token("op", text))


def _depth_at(words: list[str], index: int) -> int:
    return sum(1 if w == "(" else -1 if w == ")" else 0 for w in words[:index])


def _closing(words: list[str], opening: int) -> int:
    depth = 0
    for n in range(opening, len(words)):
        if words[n] == "(":
            depth += 1
        elif words[n] == ")":
            depth -= 1
            if depth == 0:
                return n
    return -1


def _qualify_upsert(out: list[Token], sig: list[int], words: list[str], table: str) -> None:
    """``ON CONFLICT ... DO UPDATE SET c = c + excluded.c``: a bare column there is the existing row's in SQLite and
    ambiguous in Postgres, so it is qualified with the target table (an assigned column stays bare)."""
    start = next((n + 3 for n in range(len(words) - 2) if words[n:n + 3] == ["DO", "UPDATE", "SET"]), None)
    if start is None:
        return
    depth, skip_depth, assigning = 0, None, True
    for n in range(start, len(sig)):
        token, upper = out[sig[n]], words[n]
        nxt = out[sig[n + 1]].text if n + 1 < len(sig) else ""
        prev = out[sig[n - 1]].text if n else ""
        if token.text == "(":
            depth += 1
            if skip_depth is None and words[n + 1:n + 2] == ["SELECT"]:
                skip_depth = depth  # a subquery's columns are its own
        elif token.text == ")":
            if skip_depth == depth:
                skip_depth = None
            depth -= 1
        elif depth == 0 and upper == "RETURNING":
            return
        elif depth == 0 and token.text == ",":
            assigning = True
        elif depth == 0 and upper == "WHERE":
            assigning = False
        elif token.kind in ("word", "qid") and skip_depth is None and upper not in _SQL_WORDS:
            if assigning and depth == 0 and nxt == "=":
                assigning = False  # the assigned column
            elif nxt not in (".", "(") and prev != ".":
                out[sig[n]] = Token(token.kind, f"{table}.{token.text}")


def _expressions(out: list[Token], is_ddl: bool) -> None:
    sig = _significant(out)
    for n, i in enumerate(sig):
        token = out[i]
        nxt = out[sig[n + 1]] if n + 1 < len(sig) else None
        prev = out[sig[n - 1]] if n else None
        if token.kind != "word":
            continue
        name = token.upper
        if name == "IS" and nxt is not None:
            after_not = out[sig[n + 2]] if n + 2 < len(sig) else None
            if nxt.upper == "NOT" and after_not is not None and after_not.upper not in ("NULL", "TRUE", "FALSE", "DISTINCT"):
                out[i], out[sig[n + 1]] = Token("word", "IS DISTINCT"), Token("word", "FROM")
            elif nxt.upper not in ("NULL", "NOT", "TRUE", "FALSE", "DISTINCT"):
                out[i] = Token("word", "IS NOT DISTINCT FROM")
        elif name == "LIKE" and not is_ddl:
            out[i] = Token("word", "ILIKE")
        elif name == "GLOB" and not is_ddl:
            out[i] = Token("op", "#~#")
        elif name == "LIMIT" and nxt is not None and nxt.text == "-" and n + 2 < len(sig) and out[sig[n + 2]].text == "1":
            out[sig[n + 1]], out[sig[n + 2]] = Token("ws", ""), Token("word", "ALL")
        elif nxt is not None and nxt.text == "(" and (prev is None or prev.text != "."):
            if name == "CAST" and not is_ddl:
                _cast(out, sig, n)
            elif name in ("LAST_INSERT_ROWID", "CHANGES") and n + 2 < len(sig) and out[sig[n + 2]].text == ")":
                # Per-connection counters the connection fills in (connection.Connection._run).
                out[i], out[sig[n + 1]], out[sig[n + 2]] = Token("word", "{%s}" % name.lower()), Token("ws", ""), Token("ws", "")
            elif name == "IFNULL":
                out[i] = Token("word", "coalesce")
            elif name in _TIME_FUNCTIONS and not (is_ddl and prev is not None and prev.upper == "DEFAULT"):
                out[i] = Token("word", _TIME_FUNCTIONS[name])
            elif name in ("MAX", "MIN") and _top_level_commas(out, sig, n + 1):
                out[i] = Token("word", "greatest" if name == "MAX" else "least")
            elif name == "GROUP_CONCAT":
                out[i] = Token("word", "hermes_group_concat")
            elif name in _HELPER_FUNCTIONS:
                out[i] = Token("word", "hermes_" + name.lower())


def _cast(out: list[Token], sig: list[int], n: int) -> None:
    """``CAST(x AS type)`` with SQLite's type names: BLOB becomes ``hermes_blob(x)`` (text's UTF-8 bytes, as SQLite
    gives), the numeric names their Postgres types."""
    depth, as_at = 0, None
    for m in range(n + 1, len(sig)):
        text = out[sig[m]].text
        if text == "(":
            depth += 1
        elif text == ")":
            depth -= 1
            if depth == 0:
                break
        elif depth == 1 and out[sig[m]].upper == "AS":
            as_at = m
    else:
        return
    if as_at is None or as_at + 1 >= m:
        return
    type_name = out[sig[as_at + 1]].upper
    if type_name == "BLOB" and as_at + 2 == m:
        out[sig[n]] = Token("word", "hermes_blob")
        out[sig[as_at]] = out[sig[as_at + 1]] = Token("ws", "")
    elif type_name in _TYPE_MAP and _TYPE_MAP[type_name]:
        out[sig[as_at + 1]] = Token("word", _TYPE_MAP[type_name])


def _top_level_commas(out: list[Token], sig: list[int], open_index: int) -> bool:
    depth = 0
    for i in sig[open_index:]:
        if out[i].text == "(":
            depth += 1
        elif out[i].text == ")":
            depth -= 1
            if depth == 0:
                return False
        elif out[i].text == "," and depth == 1:
            return True
    return False


def _order_by(out: list[Token]) -> None:
    """Give each ORDER BY term SQLite's NULL placement: NULLs sort as the smallest value."""
    sig = _significant(out)
    inserts: list[tuple[int, str]] = []
    n = 0
    while n < len(sig) - 1:
        if not (out[sig[n]].upper == "ORDER" and out[sig[n + 1]].upper == "BY"):
            n += 1
            continue
        depth, has_nulls, direction = 0, False, "ASC"
        j = n + 2
        while True:
            token = out[sig[j]] if j < len(sig) else None
            ends_clause = token is None or (depth == 0 and (token.text in (")", ";") or token.upper in _CLAUSE_END))
            if ends_clause or (depth == 0 and token.text == ","):
                if not has_nulls:
                    inserts.append((sig[j] if token is not None else len(out),
                                    " NULLS FIRST " if direction == "ASC" else " NULLS LAST "))
                if ends_clause:
                    break
                has_nulls, direction = False, "ASC"
            elif token.text == "(":
                depth += 1
            elif token.text == ")":
                depth -= 1
            elif depth == 0 and token.upper in ("ASC", "DESC"):
                direction = token.upper
            elif depth == 0 and token.upper == "NULLS":
                has_nulls = True
            j += 1
        n = j
    for position, text in sorted(inserts, reverse=True):
        out.insert(position, Token("nulls", text))
