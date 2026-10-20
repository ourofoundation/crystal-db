"""A stand-in for Ouro's dataset endpoints, for testing code that uses
``crystaldb.ouro.OuroSharedDatabase`` without a server.

``FakeOuro`` keeps the five tables in one SQLite file and answers the
endpoints the store uses the way the API does: read-only SQL returns rows
with JSON columns parsed and flags as booleans, upserts replace rows by
primary key, updates set the given columns of rows matched by key, and
deletes remove rows by key. ``fake.db`` is a ``StructureDatabase`` on that file, for
setting up and inspecting what the server holds.

    fake = FakeOuro(tmp_path / "server.db")
    store = fake.store()
"""

from __future__ import annotations

import json
import json as _json
import re
import sqlite3
import threading
import uuid
import zlib
from pathlib import Path

from .database import StructureDatabase
from .ouro import BOOLEANS, JSON_COLUMNS, PRIMARY_KEYS, TABLES, OuroSharedDatabase


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code, self.text = body, status, json.dumps(body)

    def json(self):
        return self.body


class FakeOuro:
    """Five 'datasets' as tables of one SQLite database."""

    def __init__(self, path):
        path = Path(path)
        if path.is_dir():
            path = path / "server.db"
        self.path = path
        self.db = StructureDatabase(path)
        self.db.close()
        # The store reads several things at once, so the server's connection
        # is shared across threads under a lock.
        self.server = sqlite3.connect(path, check_same_thread=False)
        self.server.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self.db = StructureDatabase(path)
        self.ids = {table: str(uuid.uuid4()) for table in TABLES}
        self.names = {self.ids[table]: table for table in TABLES}
        self.queries = 0
        self.client = self

    def get(self, path, **_):
        dataset_id = path.rsplit("/", 1)[-1]
        return Response({"data": {"metadata": {"table_name": f"dataset_{self.names[dataset_id]}"}}})

    def request(self, method, path, json=None, **_):
        """DELETE /datasets/:id/rows: rows by primary key (a value, or a dict
        of the key columns)."""
        assert method == "DELETE" and path.endswith("/rows"), (method, path)
        with self.lock:
            table = self.names[path.split("/")[2]]
            key = PRIMARY_KEYS[table]
            deleted = 0
            for item in json["ids"]:
                values = [item] if len(key) == 1 and not isinstance(item, dict) else [item[c] for c in key]
                cursor = self.server.execute(
                    f"DELETE FROM {table} WHERE {' AND '.join(f'{c} = ?' for c in key)}", values
                )
                deleted += cursor.rowcount
            self.server.commit()
            return Response({"data": {"deleted": deleted}})

    def post(self, path, json=None, **_):
        with self.lock:
            return self._post(path, json)

    # The API refuses request bodies over 100 kB.
    MAX_BODY = 100_000

    def _post(self, path, json):
        if len(_json.dumps(json)) > self.MAX_BODY:
            return Response({"error": {"message": "Payload Too Large"}}, 413)
        if path.endswith("/query-custom"):
            self.queries += 1
            sql = re.sub(r'datasets\."dataset_(\w+)"', r"\1", json["query"])
            rows = [dict(row) for row in self.server.execute(sql)]
            return Response({"data": [self._as_api(row) for row in rows]})
        table = self.names[path.split("/")[2]]
        try:
            return self._write(table, json)
        except sqlite3.IntegrityError as exc:
            # As Postgres does: a row that clashes with another unique key
            # (the same source and source id) fails the whole batch.
            self.server.rollback()
            return Response({"error": {"message": f"duplicate key value violates unique constraint: {exc}"}}, 409)

    def _write(self, table, json):
        key = PRIMARY_KEYS[table]
        updated = 0
        for row in json["rows"]:
            columns = list(row)
            values = [self._as_sqlite(column, row[column]) for column in columns]
            if json["mode"] == "update":
                # Only the columns a row carries change; nothing is inserted.
                sets = [column for column in columns if column not in key]
                cursor = self.server.execute(
                    f"UPDATE {table} SET {', '.join(f'{column} = ?' for column in sets)} "
                    f"WHERE {' AND '.join(f'{column} = ?' for column in key)}",
                    [values[columns.index(c)] for c in sets] + [values[columns.index(c)] for c in key],
                )
                updated += cursor.rowcount
                continue
            # An upsert by primary key (INSERT OR REPLACE would instead
            # delete a row clashing on another unique key).
            sets = [column for column in columns if column not in key]
            self.server.execute(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))}) "
                f"ON CONFLICT ({', '.join(key)}) DO "
                + (f"UPDATE SET {', '.join(f'{column} = excluded.{column}' for column in sets)}" if sets else "NOTHING"),
                values,
            )
        self.server.commit()
        if json["mode"] == "update":
            return Response({"data": {"inserted": 0, "updated": updated}})
        return Response({"data": {"inserted": len(json["rows"])}})

    @staticmethod
    def _as_api(row):
        out = {}
        for column, value in row.items():
            if value is None:
                out[column] = None
            elif column == "cif":
                out[column] = zlib.decompress(value).decode() if isinstance(value, bytes) else value
            elif column in BOOLEANS:
                out[column] = bool(value)
            elif column in JSON_COLUMNS:
                out[column] = json.loads(value)
            else:
                out[column] = value
        return out

    @staticmethod
    def _as_sqlite(column, value):
        if value is None:
            return None
        if column == "cif":
            return zlib.compress(value.encode(), 6)
        if column in BOOLEANS:
            return int(value)
        if column in JSON_COLUMNS:
            return json.dumps(value)
        return value

    def store(self, non_commercial: bool = False) -> OuroSharedDatabase:
        """An OuroSharedDatabase on this fake server."""
        return OuroSharedDatabase(self, self.ids, non_commercial=non_commercial)
