"""The shared database kept as Ouro datasets, one per table.

Each table of the SQLite schema (structures, geometries, calculations, runs,
run_structures) is an Ouro dataset with the same columns and primary key.
Postgres takes concurrent writes, so there is no single writer: a working
copy is filled from the datasets with indexed queries, the caller works on
it as an ordinary ``StructureDatabase``, and on exit its changes are
upserted straight back.

    store = OuroSharedDatabase(ouro, {"structures": "<dataset id>", ...})
    with store.working_copy(["Fe-Co"]) as db:
        ...  # changes are upserted on exit
    with store.reader(formulas=["Fe3Co"]) as db:
        ...  # read only

``ouro`` is an ``ouro.Ouro`` client (ouro-py) acting as a user who can
read and write the datasets. Reads go through the datasets' read-only SQL
endpoint, paged by key; writes through upserts by primary key.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .database import StructureDatabase
from .surfaces import DEFAULT_SURFACE

logger = logging.getLogger(__name__)

TABLES = ("structures", "geometries", "calculations", "runs", "run_structures")
PRIMARY_KEYS = {
    "structures": ("id",),
    "geometries": ("structure_id", "kind"),
    "calculations": ("structure_id", "surface"),
    "runs": ("id",),
    "run_structures": ("run_id", "structure_id"),
}
BOOLEANS = {"is_valid", "is_on_hull", "is_dynamically_stable"}
JSON_COLUMNS = {"elements", "stoichiometry", "metadata", "hubbards", "parameters"}
# Rows per read page and per upsert request (upserts are also capped by size).
PAGE_ROWS = 5000
# Ids per lookup query (each id is a primary key lookup), and lookups in
# flight at once while filling a copy.
ID_CHUNK = 1000
FILL_REQUESTS = 6
UPSERT_ROWS = 2000
UPSERT_BYTES = 5_000_000


class OuroError(RuntimeError):
    """A request to Ouro failed."""


class OuroSharedDatabase:
    def __init__(self, ouro: Any, datasets: Dict[str, str]):
        """``datasets`` maps each table name to its dataset id."""
        missing = [table for table in TABLES if table not in datasets]
        if missing:
            raise ValueError(f"Dataset ids missing for: {', '.join(missing)}")
        self.ouro = ouro
        self.datasets = dict(datasets)
        self._tables: Optional[Dict[str, str]] = None
        self._columns: Dict[str, List[str]] = {}
        self._counts: Optional[Tuple[float, Dict[str, Dict[str, Any]]]] = None

    # -------------------- Requests --------------------

    def _request(self, method: str, path: str, **kwargs) -> Any:
        response = getattr(self.ouro.client, method)(path, **kwargs)
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400 or (isinstance(body, dict) and body.get("error")):
            error = body.get("error") if isinstance(body, dict) else None
            message = error.get("message") if isinstance(error, dict) else error
            raise OuroError(f"{method.upper()} {path}: {response.status_code} {message or response.text[:300]}")
        return body.get("data") if isinstance(body, dict) else body

    @property
    def tables(self) -> Dict[str, str]:
        """Postgres table name of each dataset (``datasets."dataset_<id>"``)."""
        if self._tables is None:
            self._tables = {
                table: self._request("get", f"/datasets/{dataset_id}")["metadata"]["table_name"]
                for table, dataset_id in self.datasets.items()
            }
        return self._tables

    def table(self, name: str) -> str:
        return f'datasets."{self.tables[name]}"'

    def query(self, sql: str) -> List[Dict[str, Any]]:
        """Rows of a read-only SQL query over the tables (``self.table(name)``)."""
        rows = self._request(
            "post", f"/datasets/{self.datasets['structures']}/query-custom", json={"query": sql}
        )
        return rows or []

    def _keyed_pages(self, table: str, select: str, where: str) -> Iterator[List[Dict[str, Any]]]:
        """Pages of a filtered table read, in primary key order.

        ``select`` and ``where`` refer to the table as ``t``. Each page
        starts after the last row's key, so every page costs the same.
        """
        alias = "t"
        key = PRIMARY_KEYS[table]
        key_cols = ", ".join(f"{alias}.{column}" for column in key)
        after = ""
        while True:
            rows = self.query(
                f"SELECT {select} FROM {self.table(table)} {alias} WHERE ({where}){after} "
                f"ORDER BY {key_cols} LIMIT {PAGE_ROWS}"
            )
            if not rows:
                return
            yield rows
            if len(rows) < PAGE_ROWS:
                return
            last = rows[-1]
            values = ", ".join(_literal(last[column]) for column in key)
            after = f" AND ({key_cols}) > ({values})"

    def upsert(self, table: str, rows: Sequence[Dict[str, Any]]) -> int:
        """Insert or update rows by primary key, in batches."""
        dataset_id = self.datasets[table]
        sent = 0
        for batch in _batches(rows):
            self._request(
                "post",
                f"/datasets/{dataset_id}/data",
                json={"mode": "upsert", "rows": batch, "finalize": False},
            )
            sent += len(batch)
        return sent

    # -------------------- Reading into a local copy --------------------

    def _fill(
        self,
        out_path: Path,
        *,
        systems: Iterable[str] = (),
        formulas: Iterable[str] = (),
        structure_ids: Iterable[str] = (),
        surface: Optional[str] = None,
    ) -> Tuple[StructureDatabase, Dict[str, str]]:
        """A local StructureDatabase holding the structures selected by
        chemical system (with subsystems), reduced formula or id, with their
        geometries, calculations and runs. Returns it and the ``updated_at``
        of each structure it holds, by id."""
        subsystems = sorted(
            {
                sub
                for system in systems
                for sub in StructureDatabase.get_all_subsystems(
                    StructureDatabase.normalize_chemsys(system).split("-")
                )
            }
        )
        conditions = []
        if subsystems:
            conditions.append(f"t.chemsys IN ({_literals(subsystems)})")
        formulas = sorted(set(formulas))
        if formulas:
            conditions.append(f"t.reduced_formula IN ({_literals(formulas)})")
        ids = sorted(set(structure_ids))
        for chunk in _chunks(ids, ID_CHUNK):
            conditions.append(f"t.id IN ({_literals(chunk)})")
        db = StructureDatabase(out_path, surface=surface or DEFAULT_SURFACE)
        if not conditions:
            return db, {}

        started = time.time()
        held: Dict[str, str] = {}
        for rows in self._keyed_pages("structures", "t.*", " OR ".join(conditions)):
            self._insert(db, "structures", rows)
            held.update((row["id"], row["updated_at"]) for row in rows)
        counts = {"structures": len(held)}
        # Children by chunks of ids (primary key lookups), several requests
        # at once; rows go into the local copy as each answer arrives.
        tasks = [
            (table, f"SELECT * FROM {self.table(table)} WHERE structure_id IN ({_literals(chunk)})")
            for table in ("geometries", "calculations", "run_structures")
            for chunk in _chunks(list(held), ID_CHUNK)
        ]
        with ThreadPoolExecutor(max_workers=FILL_REQUESTS) as pool:
            futures = {pool.submit(self.query, sql): table for table, sql in tasks}
            for future in as_completed(futures):
                table, rows = futures[future], future.result()
                counts[table] = counts.get(table, 0) + len(rows)
                if rows:
                    self._insert(db, table, rows)
        run_ids = [row[0] for row in db.conn.execute("SELECT DISTINCT run_id FROM run_structures")]
        for chunk in _chunks(run_ids, ID_CHUNK):
            rows = self.query(f"SELECT * FROM {self.table('runs')} WHERE id IN ({_literals(chunk)})")
            if rows:
                self._insert(db, "runs", rows)
        db._commit()
        logger.info("Filled %s in %.1f s", counts, time.time() - started)
        return db, held

    def _insert(self, db: StructureDatabase, table: str, rows: List[Dict[str, Any]]) -> None:
        if table not in self._columns:
            self._columns[table] = db._table_columns(table)
        columns = [column for column in self._columns[table] if column in rows[0]]
        db.conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' * len(columns))})",
            ([_to_sqlite(column, row.get(column)) for column in columns] for row in rows),
        )

    def open_reader(
        self,
        surface: Optional[str] = None,
        *,
        systems: Iterable[str] = (),
        formulas: Iterable[str] = (),
        structure_ids: Iterable[str] = (),
    ) -> StructureDatabase:
        """A read-only local copy of the selected structures; its files go
        away when it is closed."""
        tmp = tempfile.mkdtemp(prefix="crystaldb-read-")
        try:
            db, _ = self._fill(
                Path(tmp) / "read.db",
                systems=systems,
                formulas=formulas,
                structure_ids=structure_ids,
                surface=surface,
            )
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        close = db.close

        def close_and_remove() -> None:
            close()
            shutil.rmtree(tmp, ignore_errors=True)

        db.close = close_and_remove  # type: ignore[method-assign]
        return db

    @contextmanager
    def reader(
        self,
        surface: Optional[str] = None,
        *,
        systems: Iterable[str] = (),
        formulas: Iterable[str] = (),
        structure_ids: Iterable[str] = (),
    ) -> Iterator[StructureDatabase]:
        """A read-only local copy of the selected structures."""
        with tempfile.TemporaryDirectory(prefix="crystaldb-read-") as tmp:
            db, _ = self._fill(
                Path(tmp) / "read.db",
                systems=systems,
                formulas=formulas,
                structure_ids=structure_ids,
                surface=surface,
            )
            try:
                yield db
            finally:
                db.close()

    @contextmanager
    def working_copy(
        self,
        chemical_systems: Iterable[str] = (),
        ingest: Optional[Callable[[bytes], dict]] = None,
        surface: Optional[str] = None,
        *,
        structure_ids: Iterable[str] = (),
    ) -> Iterator[StructureDatabase]:
        """A writable local copy of some systems; changes are upserted on exit.

        The copy holds every structure in the systems and their subsystems,
        so hulls built from it match the shared database. Changes are sent
        even when the block raises, so work done before a failure is kept.
        ``ingest`` replaces the upsert with a callable taking the changes as
        a SQLite file's bytes (tests and dry runs).
        """
        systems = list(chemical_systems)
        with tempfile.TemporaryDirectory(prefix="crystaldb-work-") as tmp:
            since = datetime.now().isoformat()
            db, held = self._fill(
                Path(tmp) / "work.db", systems=systems, structure_ids=structure_ids, surface=surface
            )
            try:
                yield db
            finally:
                delta_path = Path(tmp) / "delta.db"
                changed = db.export_changes(delta_path, since)
                db.close()
                if changed:
                    # A run that finished must not fail because its results
                    # could not be saved for later runs.
                    try:
                        sending = time.time()
                        if ingest is not None:
                            result = ingest(delta_path.read_bytes())
                        else:
                            result = self.push(delta_path, filled=held)
                        logger.info(
                            "Sent %d changed structures in %.1f s: %s",
                            changed, time.time() - sending, result,
                        )
                    except Exception:
                        logger.exception("Could not send %d changed structures", changed)

    # -------------------- Writing changes back --------------------

    def push(self, delta_path: Path, filled: Optional[Dict[str, str]] = None) -> Dict[str, int]:
        """Upsert a SQLite file of changes (``StructureDatabase.export_changes``).

        A structure the shared tables already hold under another id (the same
        ``(source, source_id)``, or for generated structures the same
        ``structure_hash``) is merged into that row, as the old single writer
        did; an existing row's identity fields change only when the changed
        copy is newer, and it keeps its id, source and creation time.
        ``filled`` is the ``updated_at`` of each structure a working copy was
        filled with: those need no lookup, and the ones it did not change are
        not written back (a change file carries them only to map ids, and
        writing them would undo whatever changed them since).
        """
        delta = StructureDatabase(delta_path, readonly=True)
        try:
            for attempt in range(2):
                try:
                    return self._push(delta, dict(filled or {}))
                except OuroError as exc:
                    # Another run inserted the same new structure meanwhile:
                    # map onto it and try again.
                    if attempt == 0 and ("23505" in str(exc) or "duplicate key" in str(exc)):
                        logger.info("Concurrent insert, merging again: %s", exc)
                        filled = None
                        continue
                    raise
            raise AssertionError("unreachable")
        finally:
            delta.close()

    def _push(self, delta: StructureDatabase, filled: Dict[str, str]) -> Dict[str, int]:
        rows = {
            table: [dict(row) for row in delta.conn.execute(f"SELECT * FROM {table}")]
            for table in TABLES
        }
        structures = rows["structures"]
        existing = self._existing(structures, filled)
        target: Dict[str, str] = {}
        merged = new = 0
        out_structures = []
        for row in structures:
            match = existing.get(row["id"]) or existing.get(_identity(row))
            if match is None:
                target[row["id"]] = row["id"]
                out_structures.append(_to_ouro("structures", row))
                new += 1
                continue
            target[row["id"]] = match["id"]
            merged += row["id"] != match["id"]
            # Rows already held keep their own id, source and creation time,
            # and take changed identity fields only when this copy is newer.
            if match.get("updated_at") is not None and row["updated_at"] <= match["updated_at"]:
                continue
            updated = dict(row, id=match["id"])
            for column in ("source", "source_id", "created_at"):
                if column in match:
                    updated[column] = match[column]
            out_structures.append(_to_ouro("structures", updated))

        def remap(table: str, column: str) -> List[Dict[str, Any]]:
            return [
                _to_ouro(table, dict(row, **{column: target.get(row[column], row[column])}))
                for row in rows[table]
            ]

        counts = {
            "structures": self.upsert("structures", out_structures),
            "geometries": self.upsert("geometries", remap("geometries", "structure_id")),
            "calculations": self.upsert("calculations", remap("calculations", "structure_id")),
            "runs": self.upsert("runs", [_to_ouro("runs", row) for row in rows["runs"]]),
            "run_structures": self.upsert("run_structures", remap("run_structures", "structure_id")),
        }
        counts["new_structures"] = new
        counts["merged_into_existing"] = merged
        return counts

    def _existing(self, structures: List[Dict[str, Any]], filled: Dict[str, str]) -> Dict[Any, Dict[str, Any]]:
        """Stored rows matching the changed structures, keyed by id and by
        identity ((source, source_id), or structure_hash when source_id is
        null). Rows a working copy was filled with need no lookup."""
        found: Dict[Any, Dict[str, Any]] = {}
        unknown = [row for row in structures if row["id"] not in filled]
        for row in structures:
            if row["id"] in filled:
                found[row["id"]] = {"id": row["id"], "updated_at": filled[row["id"]]}
        if not unknown:
            return found
        stored: List[Dict[str, Any]] = []
        select = (
            f"SELECT id, source, source_id, structure_hash, created_at, updated_at "
            f"FROM {self.table('structures')} WHERE "
        )
        for chunk in _chunks([row["id"] for row in unknown], ID_CHUNK):
            stored += self.query(select + f"id IN ({_literals(chunk)})")
        keyed = [(row["source"], row["source_id"]) for row in unknown if row["source_id"] is not None]
        for chunk in _chunks(keyed, ID_CHUNK):
            pairs = ", ".join(f"({_literal(source)}, {_literal(source_id)})" for source, source_id in chunk)
            stored += self.query(select + f"(source, source_id) IN ({pairs})")
        hashes = [row["structure_hash"] for row in unknown if row["source_id"] is None and row["structure_hash"]]
        for chunk in _chunks(hashes, ID_CHUNK):
            stored += self.query(select + f"source_id IS NULL AND structure_hash IN ({_literals(chunk)})")
        for row in stored:
            found[row["id"]] = row
            found[_identity(row)] = row
        return found

    # -------------------- Summaries --------------------

    def source_counts(self, max_age: float = 3600.0) -> Dict[str, Dict[str, Any]]:
        """Valid structures per source, in the shape of
        ``StructureDatabase.source_counts`` (no import times). Counting reads
        the whole table, so a count is reused for ``max_age`` seconds."""
        cached = self._counts
        if cached is not None and time.time() - cached[0] < max_age:
            return cached[1]
        rows = self.query(
            f"SELECT source, count(*) AS n FROM {self.table('structures')} "
            f"WHERE is_valid GROUP BY source"
        )
        counts = {row["source"]: {"structures": row["n"], "imported_at": None} for row in rows}
        self._counts = (time.time(), counts)
        return counts


# -------------------- Value conversion --------------------


def _identity(row: Dict[str, Any]) -> Tuple:
    if row.get("source_id") is not None:
        return ("source", row["source"], row["source_id"])
    return ("hash", row.get("structure_hash"))


def _literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _literals(values: Iterable[Any]) -> str:
    return ", ".join(_literal(value) for value in values)


def _to_sqlite(column: str, value: Any) -> Any:
    """A dataset value as the SQLite schema stores it."""
    if value is None:
        return None
    if column == "cif":
        return zlib.compress(value.encode(), 6)
    if column in BOOLEANS:
        return int(bool(value))
    if column in JSON_COLUMNS:
        return json.dumps(value)
    return value


def _to_ouro(table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """A SQLite row as the dataset takes it."""
    out: Dict[str, Any] = {}
    for column, value in row.items():
        if value is None:
            out[column] = None
        elif column == "cif":
            out[column] = zlib.decompress(value).decode() if isinstance(value, (bytes, bytearray)) else value
        elif column in BOOLEANS:
            out[column] = bool(value)
        elif column in JSON_COLUMNS:
            try:
                out[column] = json.loads(value) if isinstance(value, str) else value
            except ValueError:
                out[column] = value
        else:
            out[column] = value
    return out


def _chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _batches(rows: Sequence[Dict[str, Any]]) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    size = 0
    for row in rows:
        row_size = len(json.dumps(row, default=str))
        if batch and (len(batch) >= UPSERT_ROWS or size + row_size > UPSERT_BYTES):
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += row_size
    if batch:
        yield batch
