"""Row-ledger transactions for the coordination and worker ledgers.

A ledger verb works on whole rows of small, bounded tables (pruned by count and age) under careful state rules. An
``atomic`` verb runs in one ``store.transaction()``: ``table(name)`` loads that table's rows once, row-locked, as a
mapping of key to row dict; the verb reads and changes the dicts; on success every inserted, changed and removed row
is written back, and on any exception nothing is. A verb called from inside another shares the outer transaction.
"""

from __future__ import annotations

import contextvars
import copy
import functools
from collections.abc import MutableMapping
from typing import Any, Callable, Iterator, TypeVar

from psycopg import Connection, sql
from psycopg.types.json import Jsonb

from ._sql import rows as _rows

# Each ledger table's key column and its jsonb columns.
TABLES: dict[str, tuple[str, frozenset[str]]] = {
    "async_delegations": ("delegation_id", frozenset({"task", "event", "result", "children"})),
    "delivery_obligations": ("obligation_id", frozenset()),
    "resumable_subagents": ("subagent_id", frozenset({"document"})),
}

_active: contextvars.ContextVar[dict[int, "LedgerTransaction"]] = contextvars.ContextVar("pg_ledger_transactions", default={})


class LedgerTable(MutableMapping[str, dict[str, Any]]):
    """One table's rows in the running transaction, keyed by its key column."""

    def __init__(self, conn: Connection, name: str, *, lock: bool = True) -> None:
        self.name = name
        self.key, self.json_columns = TABLES[name]
        # A read-only store's transaction may not lock rows; it never writes them back either.
        query = sql.SQL("SELECT * FROM {} FOR UPDATE" if lock else "SELECT * FROM {}").format(sql.Identifier(name))
        found = _rows(conn, query.as_string(conn))
        self._rows = {row[self.key]: row for row in found}
        self._loaded = copy.deepcopy(self._rows)

    def __getitem__(self, key: str) -> dict[str, Any]:
        return self._rows[key]

    def __setitem__(self, key: str, row: dict[str, Any]) -> None:
        self._rows[key] = row

    def __delitem__(self, key: str) -> None:
        del self._rows[key]

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._rows))

    def __len__(self) -> int:
        return len(self._rows)

    def _cell(self, column: str, value: Any) -> Any:
        return Jsonb(value) if column in self.json_columns and value is not None else value

    def flush(self, conn: Connection) -> None:
        table = sql.Identifier(self.name)
        for key in self._loaded.keys() - self._rows.keys():
            conn.execute(sql.SQL("DELETE FROM {} WHERE {} = %s").format(table, sql.Identifier(self.key)), (key,))
        for key, row in self._rows.items():
            before = self._loaded.get(key)
            if before is None:
                columns = list(row)
                conn.execute(sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    table, sql.SQL(", ").join(map(sql.Identifier, columns)), sql.SQL(", ").join(sql.Placeholder() * len(columns))),
                    [self._cell(column, row[column]) for column in columns])
            elif row != before:
                changed = [column for column in row if row[column] != before.get(column)]
                conn.execute(sql.SQL("UPDATE {} SET {} WHERE {} = %s").format(
                    table, sql.SQL(", ").join(sql.SQL("{} = %s").format(sql.Identifier(column)) for column in changed),
                    sql.Identifier(self.key)), [*(self._cell(column, row[column]) for column in changed), key])


class LedgerTransaction:
    def __init__(self, conn: Connection, *, read_only: bool = False) -> None:
        self.conn = conn
        self.read_only = read_only
        self._tables: dict[str, LedgerTable] = {}

    def table(self, name: str) -> LedgerTable:
        if name not in self._tables:
            self._tables[name] = LedgerTable(self.conn, name, lock=not self.read_only)
        return self._tables[name]

    def flush(self) -> None:
        for table in self._tables.values():
            table.flush(self.conn)


F = TypeVar("F", bound=Callable[..., Any])


def atomic(method: F) -> F:
    """Run a repository verb in one ledger transaction on its store, or in the caller's when one is running."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        running = _active.get()
        if id(self._store) in running:
            return method(self, *args, **kwargs)
        with self._store.transaction() as conn:
            transaction = LedgerTransaction(conn, read_only=bool(getattr(self._store, "read_only", False)))
            token = _active.set({**running, id(self._store): transaction})
            try:
                result = method(self, *args, **kwargs)
                transaction.flush()
                return result
            finally:
                _active.reset(token)

    return wrapper  # type: ignore[return-value]


def current(store: Any) -> LedgerTransaction:
    """The ledger transaction running on ``store``; only callable inside an ``atomic`` verb."""
    return _active.get()[id(store)]
