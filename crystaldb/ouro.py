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
    with store.bulk_copy(sources=["mp"]) as db:
        ...  # whole-database jobs: imports, stored hull distances

``ouro`` is an ``ouro.Ouro`` client (ouro-py) acting as a user who can
read and write the datasets. Reads go through the datasets' read-only SQL
endpoint, paged by key; writes through upserts by primary key.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import tempfile
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

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
# Ids per lookup query (each id is a primary key lookup).
ID_CHUNK = 1000
# Most read requests one store has in flight, across every copy it fills at
# once (a route container fills several), so the API is never flooded.
MAX_REQUESTS = 12
UPSERT_ROWS = 2000
UPSERT_BYTES = 5_000_000
UPSERT_REQUESTS = 4
# A copy of a large part of the database (everything, or a whole source)
# reads each table in this many key ranges, several at once. Once it holds
# more structures than BULK_CHILDREN, its child rows come by key range too
# (whole tables) instead of by lists of ids.
KEY_RANGES = 16
RANGE_REQUESTS = 8
BULK_CHILDREN = 1_000_000
BULK_CACHE_KIB = 2 * 1024 * 1024
BULK_COMMIT_ROWS = 200_000
# Stored hull distances: a copy sends changes to these alone as patches, so
# recomputing hulls never rewrites a calculation's energy.
HULL_COLUMNS = ("e_above_hull", "is_on_hull", "hull_computed_at")


class OuroError(RuntimeError):
    """A request to Ouro failed."""


def stored_id(db: StructureDatabase, structure_id: str) -> str:
    """The id a structure added in a working copy is stored under once the
    copy's changes are sent: its own, unless a concurrent run stored the
    same structure first (``working_copy``)."""
    return getattr(db, "merged_ids", {}).get(structure_id, structure_id)


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
        self._uncommitted = 0
        self._slots = threading.BoundedSemaphore(MAX_REQUESTS)

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
        with self._slots:
            rows = self._request(
                "post", f"/datasets/{self.datasets['structures']}/query-custom", json={"query": sql}
            )
        return rows or []

    def _page(self, table: str, select: str, where: str) -> List[Dict[str, Any]]:
        """The first PAGE_ROWS rows of a filtered table read, in key order."""
        key_cols = ", ".join(f"t.{column}" for column in PRIMARY_KEYS[table])
        return self.query(
            f"SELECT {select} FROM {self.table(table)} t WHERE ({where}) "
            f"ORDER BY {key_cols} LIMIT {PAGE_ROWS}"
        )

    @staticmethod
    def _after(table: str, row: Dict[str, Any]) -> str:
        """Rows with a primary key past ``row``'s (the table as ``t``)."""
        key = PRIMARY_KEYS[table]
        key_cols = ", ".join(f"t.{column}" for column in key)
        values = ", ".join(_literal(row[column]) for column in key)
        return f"({key_cols}) > ({values})"

    def _keyed_pages(self, table: str, select: str, where: str) -> Iterator[List[Dict[str, Any]]]:
        """Pages of a filtered table read, in primary key order.

        ``select`` and ``where`` refer to the table as ``t``. Each page
        starts after the last row's key, so every page costs the same.
        """
        condition = where
        while True:
            rows = self._page(table, select, condition)
            if not rows:
                return
            yield rows
            if len(rows) < PAGE_ROWS:
                return
            condition = f"({where}) AND {self._after(table, rows[-1])}"

    def _pages(self, table: str, select: str, where: str) -> Iterator[List[Dict[str, Any]]]:
        """Pages of a filtered table read: one request when the rows fit in
        a page, else the rest by key range, several ranges at once (pages
        then come in no particular order)."""
        rows = self._page(table, select, where)
        if rows:
            yield rows
        if len(rows) < PAGE_ROWS:
            return
        yield from self._ranged_pages(table, select, f"({where}) AND {self._after(table, rows[-1])}")

    def _ranged_pages(self, table: str, select: str, where: str) -> Iterator[List[Dict[str, Any]]]:
        """Pages of a filtered table read in KEY_RANGES ranges of its first
        key column (a uuid), several ranges at once; pages come in no
        particular order."""
        column = f"t.{PRIMARY_KEYS[table][0]}"
        bounds = [None, *(f"{digit:x}0000000-0000-0000-0000-000000000000" for digit in range(1, KEY_RANGES)), None]
        pages: "queue.Queue[Any]" = queue.Queue(maxsize=RANGE_REQUESTS * 2)
        done = object()

        def read(low: Optional[str], high: Optional[str]) -> None:
            limits = [f"{column} >= {_literal(low)}"] if low else []
            limits += [f"{column} < {_literal(high)}"] if high else []
            ranged = " AND ".join([f"({where})", *limits])
            try:
                for rows in self._keyed_pages(table, select, ranged):
                    pages.put(rows)
            except BaseException as exc:
                pages.put(exc)
            finally:
                pages.put(done)

        with ThreadPoolExecutor(max_workers=RANGE_REQUESTS) as pool:
            for low, high in zip(bounds, bounds[1:]):
                pool.submit(read, low, high)
            remaining = len(bounds) - 1
            try:
                while remaining:
                    item = pages.get()
                    if item is done:
                        remaining -= 1
                    elif isinstance(item, BaseException):
                        raise item
                    else:
                        yield item
            finally:
                # Let readers still running finish without blocking.
                while remaining:
                    if pages.get() is done:
                        remaining -= 1

    def upsert(self, table: str, rows: Sequence[Dict[str, Any]]) -> int:
        """Insert or replace whole rows by primary key, in batches."""
        return self._write(table, rows, "upsert")

    def update(self, table: str, rows: Sequence[Dict[str, Any]]) -> int:
        """Set the columns each row carries on the stored row with its
        primary key; keys that match nothing are skipped. Returns the number
        of rows updated."""
        return self._write(table, rows, "update")

    def _write(self, table: str, rows: Sequence[Dict[str, Any]], mode: str) -> int:
        dataset_id = self.datasets[table]
        batches = list(_batches(rows))

        def send(batch: List[Dict[str, Any]]) -> int:
            data = self._request(
                "post",
                f"/datasets/{dataset_id}/data",
                json={"mode": mode, "rows": batch, "finalize": False},
            )
            if mode == "update" and isinstance(data, dict):
                return int(data.get("updated", 0))
            return len(batch)

        if len(batches) <= 1:
            return sum(send(batch) for batch in batches)
        with ThreadPoolExecutor(max_workers=UPSERT_REQUESTS) as pool:
            return sum(pool.map(send, batches))

    # -------------------- Reading into a local copy --------------------

    def _fill(
        self,
        out_path: Path,
        *,
        systems: Iterable[str] = (),
        formulas: Iterable[str] = (),
        structure_ids: Iterable[str] = (),
        sources: Iterable[str] = (),
        everything: bool = False,
        surface: Optional[str] = None,
        surfaces: Optional[Iterable[str]] = None,
        geometries: Union[bool, Iterable[str]] = True,
        runs: bool = True,
    ) -> Tuple[StructureDatabase, Dict[str, str]]:
        """A local StructureDatabase holding the structures selected by
        chemical system (with subsystems), reduced formula, id or source (or
        every structure), with their geometries, calculations and runs.
        ``surfaces`` limits the calculations to those surfaces.
        ``geometries`` is True for every geometry, False for none (hulls need
        none), or reduced formulas whose structures' geometries to read
        (matching a structure needs only its formula's; ``load_geometries``
        fetches others later). ``runs=False`` leaves runs and their links out.
        Returns it and the ``updated_at`` of each structure it holds, by id."""
        subsystems = sorted(
            {
                sub
                for system in systems
                for sub in StructureDatabase.get_all_subsystems(
                    StructureDatabase.normalize_chemsys(system).split("-")
                )
            }
        )
        # (column, values) of the structures selected; any one may match.
        selectors: List[Tuple[str, Sequence[Any]]] = []
        if subsystems:
            selectors.append(("chemsys", subsystems))
        formulas = sorted(set(formulas))
        if formulas:
            selectors.append(("reduced_formula", formulas))
        ids = sorted(set(structure_ids))
        for chunk in _chunks(ids, ID_CHUNK):
            selectors.append(("id", chunk))
        sources = sorted(set(sources))
        if sources:
            selectors.append(("source", sources))

        # Only a copy of whole sources (or everything) reads by key range.
        bulk = everything or (bool(sources) and len(selectors) == 1)
        db = StructureDatabase(out_path, surface=surface or DEFAULT_SURFACE)
        if not selectors and not everything:
            _snapshot_hulls(db)
            return db, {}

        started = time.time()
        held: Dict[str, str] = {}
        where = "TRUE" if everything else " OR ".join(
            f"t.{column} IN ({_literals(values)})" for column, values in selectors
        )
        children = {"calculations": "TRUE"}
        if runs:
            children["run_structures"] = "TRUE"
        if geometries:
            children["geometries"] = "TRUE"
        geometry_formulas = None if geometries is True or not geometries else set(geometries)
        if surfaces is not None:
            children["calculations"] = f"t.surface IN ({_literals(sorted(set(surfaces))) or 'NULL'})"
        # Millions of rows arrive in random key order: load them without the
        # secondary indexes, in a big cache, committing as they come, and
        # build the indexes once at the end.
        indexes = _drop_indexes(db) if bulk else []
        if bulk:
            db.conn.execute(f"PRAGMA cache_size=-{BULK_CACHE_KIB}")
        counts: Dict[str, int] = {}
        lookups: Dict[Any, str] = {}
        pool = ThreadPoolExecutor(max_workers=MAX_REQUESTS)

        def look_up_children(structures: Iterable[Dict[str, Any]]) -> None:
            """Queue the child rows of some structures: primary key lookups
            by chunks of ids (a child table's own key ranges are slow to
            filter on the server). Geometries only for ``geometry_formulas``
            when given."""
            structures = list(structures)
            for table, filter in children.items():
                ids = [
                    row["id"]
                    for row in structures
                    if table != "geometries" or geometry_formulas is None or row["reduced_formula"] in geometry_formulas
                ]
                for chunk in _chunks(ids, ID_CHUNK):
                    sql = f"SELECT * FROM {self.table(table)} t WHERE t.structure_id IN ({_literals(chunk)}) AND {filter}"
                    lookups[pool.submit(self.query, sql)] = table

        try:
            for rows in (self._ranged_pages if bulk else self._pages)("structures", "t.*", where):
                self._insert(db, "structures", rows, commit=bulk)
                held.update((row["id"], row["updated_at"]) for row in rows)
                if not bulk:
                    # Children are looked up while further pages arrive.
                    look_up_children(rows)
            counts["structures"] = len(held)
            if bulk and len(held) > BULK_CHILDREN:
                # Most of the database: read the child tables whole by key
                # range (filtering by source on the server costs a join per
                # page), then drop the rows of structures the copy does not hold.
                for table, filter in children.items():
                    for rows in self._ranged_pages(table, "t.*", filter):
                        self._insert(db, table, rows, commit=True)
                    if not everything:
                        db.conn.execute(f"DELETE FROM {table} WHERE structure_id NOT IN (SELECT id FROM structures)")
                    counts[table] = db.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            elif bulk:
                look_up_children(
                    {"id": row[0], "reduced_formula": row[1]}
                    for row in db.conn.execute("SELECT id, reduced_formula FROM structures")
                )
            # Rows go into the copy as each answer arrives.
            for future in as_completed(lookups):
                table, rows = lookups[future], future.result()
                counts[table] = counts.get(table, 0) + len(rows)
                if rows:
                    self._insert(db, table, rows, commit=bulk)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        self._fill_runs(db, runs)
        db._commit()
        if indexes:
            indexing = time.time()
            for sql in indexes:
                db.conn.execute(sql)
            db.conn.execute("PRAGMA cache_size=-131072")
            db._commit()
            logger.info("Built %d indexes in %.1f s", len(indexes), time.time() - indexing)
        _snapshot_hulls(db)
        db._commit()
        logger.info("Filled %s in %.1f s", counts, time.time() - started)
        return db, held

    def _fill_runs(self, db: StructureDatabase, runs: bool) -> None:
        """The runs a copy's run links point to (``runs=False``: none)."""
        if not runs:
            return
        run_ids = [row[0] for row in db.conn.execute("SELECT DISTINCT run_id FROM run_structures")]
        for chunk in _chunks(run_ids, ID_CHUNK):
            rows = self.query(f"SELECT * FROM {self.table('runs')} WHERE id IN ({_literals(chunk)})")
            if rows:
                self._insert(db, "runs", rows)

    def load_geometries(self, db: StructureDatabase, structure_ids: Iterable[str]) -> int:
        """Bring the stored geometries of some structures into a copy filled
        without them (``geometries=`` a list of formulas). They keep their
        stored ``updated_at``, so they never count as the copy's changes.
        Returns the number of geometries added."""
        have = {row[0] for row in db.conn.execute("SELECT DISTINCT structure_id FROM geometries")}
        ids = sorted(set(structure_ids) - have)
        added = 0
        with ThreadPoolExecutor(max_workers=MAX_REQUESTS) as pool:
            for rows in pool.map(
                lambda chunk: self.query(
                    f"SELECT * FROM {self.table('geometries')} WHERE structure_id IN ({_literals(chunk)})"
                ),
                list(_chunks(ids, ID_CHUNK)),
            ):
                if rows:
                    self._insert(db, "geometries", rows)
                    added += len(rows)
        db._commit()
        return added

    def _insert(
        self, db: StructureDatabase, table: str, rows: List[Dict[str, Any]], commit: bool = False
    ) -> None:
        """Rows into the copy; ``commit`` commits every BULK_COMMIT_ROWS rows,
        so a big fill never holds one huge transaction."""
        if table not in self._columns:
            self._columns[table] = db._table_columns(table)
        columns = [column for column in self._columns[table] if column in rows[0]]
        db.conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' * len(columns))})",
            ([_to_sqlite(column, row.get(column)) for column in columns] for row in rows),
        )
        if commit:
            self._uncommitted += len(rows)
            if self._uncommitted >= BULK_COMMIT_ROWS:
                db._commit()
                self._uncommitted = 0

    def open_reader(
        self,
        surface: Optional[str] = None,
        *,
        systems: Iterable[str] = (),
        formulas: Iterable[str] = (),
        structure_ids: Iterable[str] = (),
        geometries: Union[bool, Iterable[str]] = True,
        runs: bool = True,
    ) -> StructureDatabase:
        """A read-only local copy of the selected structures; its files go
        away when it is closed. ``geometries`` and ``runs`` leave out what
        the caller does not read (see ``_fill``)."""
        tmp = tempfile.mkdtemp(prefix="crystaldb-read-")
        try:
            db, _ = self._fill(
                Path(tmp) / "read.db",
                systems=systems,
                formulas=formulas,
                structure_ids=structure_ids,
                surface=surface,
                geometries=geometries,
                runs=runs,
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
        geometries: Union[bool, Iterable[str]] = True,
        runs: bool = True,
    ) -> Iterator[StructureDatabase]:
        """A read-only local copy of the selected structures (``geometries``
        and ``runs`` as in ``_fill``)."""
        with tempfile.TemporaryDirectory(prefix="crystaldb-read-") as tmp:
            db, _ = self._fill(
                Path(tmp) / "read.db",
                systems=systems,
                formulas=formulas,
                structure_ids=structure_ids,
                surface=surface,
                geometries=geometries,
                runs=runs,
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
        geometries: Union[bool, Iterable[str]] = True,
        runs: bool = True,
    ) -> Iterator[StructureDatabase]:
        """A writable local copy of some systems; changes are upserted on exit.

        The copy holds every structure in the systems and their subsystems,
        so hulls built from it match the shared database. Changes are sent
        even when the block raises, so work done before a failure is kept.
        ``ingest`` replaces the upsert with a callable taking the changes as
        a SQLite file's bytes (tests and dry runs). ``geometries`` and
        ``runs`` leave out what the caller does not read (see ``_fill``).

        After the block, ``db.merged_ids`` maps the id of each structure the
        copy added that the shared tables already held under another id (a
        concurrent run stored it first) to that id; ``stored_id`` reads it.
        """
        systems = list(chemical_systems)
        with tempfile.TemporaryDirectory(prefix="crystaldb-work-") as tmp:
            since = datetime.now().isoformat()
            db, held = self._fill(
                Path(tmp) / "work.db",
                systems=systems,
                structure_ids=structure_ids,
                surface=surface,
                geometries=geometries,
                runs=runs,
            )
            db.merged_ids = {}
            try:
                yield db
            finally:
                # A run that finished must not fail because its results
                # could not be saved for later runs.
                try:
                    self._send_changes(db, held, since, ingest)
                except Exception:
                    logger.exception("Could not send the copy's changes")

    @contextmanager
    def bulk_copy(
        self,
        *,
        sources: Iterable[str] = (),
        surfaces: Optional[Iterable[str]] = None,
        geometries: bool = True,
        surface: Optional[str] = None,
    ) -> Iterator[StructureDatabase]:
        """A writable local copy of every structure from ``sources`` (every
        structure when none are named), for whole-database jobs: imports and
        stored hull distances. ``surfaces`` and ``geometries`` leave out what
        the job does not read (see ``_fill``). Changes are sent when the
        block succeeds, and a failure to send them raises.
        """
        sources = list(sources)
        with tempfile.TemporaryDirectory(prefix="crystaldb-bulk-") as tmp:
            since = datetime.now().isoformat()
            db, held = self._fill(
                Path(tmp) / "work.db",
                sources=sources,
                everything=not sources,
                surfaces=surfaces,
                geometries=geometries,
                surface=surface,
            )
            try:
                yield db
            except BaseException:
                db.close()
                raise
            self._send_changes(db, held, since)

    def _send_changes(
        self,
        db: StructureDatabase,
        held: Dict[str, str],
        since: str,
        ingest: Optional[Callable[[bytes], dict]] = None,
    ) -> Dict[str, Any]:
        """Send what changed in a filled copy since ``since``, and close it.

        New and changed rows go through ``push``; stored hull distances that
        changed on calculations the copy did not otherwise change go as
        patches of the hull columns alone. ``ingest`` takes the changes as a
        SQLite file's bytes instead (tests and dry runs; patches are counted,
        not sent).
        """
        delta_path = db.db_path.parent / "delta.db"
        try:
            changed = db.export_changes(delta_path, since)
            patches = _hull_patches(db, since)
        finally:
            db.close()
        if not changed and not patches:
            return {}
        sending = time.time()
        result: Dict[str, Any] = {}
        if changed:
            result = ingest(delta_path.read_bytes()) if ingest is not None else self.push(delta_path, filled=held)
            db.merged_ids = result.pop("merged_ids", {})
        if patches:
            result["hull_patches"] = len(patches) if ingest is not None else self.update("calculations", patches)
        logger.info(
            "Sent %d changed structures and %d hull distances in %.1f s: %s",
            changed, len(patches), time.time() - sending, result,
        )
        return result

    # -------------------- Writing changes back --------------------

    def push(self, delta_path: Path, filled: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
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
        Returns the rows written per table, and ``merged_ids``: each merged
        structure's id in the file mapped to the id it is stored under.
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

    def _push(self, delta: StructureDatabase, filled: Dict[str, str]) -> Dict[str, Any]:
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
        counts["merged_ids"] = {old: stored for old, stored in target.items() if old != stored}
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


# -------------------- Hull distances --------------------


def _drop_indexes(db: StructureDatabase) -> List[str]:
    """Drop a copy's secondary indexes; returns the statements that build
    them again."""
    rows = db.conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL"
    ).fetchall()
    for name, _ in rows:
        db.conn.execute(f'DROP INDEX "{name}"')
    db._commit()
    return [sql for _, sql in rows]


def _snapshot_hulls(db: StructureDatabase) -> None:
    """Keep the stored hull distances a copy was filled with, to tell later
    which ones a job changed."""
    columns = ", ".join(HULL_COLUMNS)
    db.conn.execute("DROP TABLE IF EXISTS _filled_hulls")
    db.conn.execute(
        f"CREATE TABLE _filled_hulls AS SELECT structure_id, surface, {columns} FROM calculations"
    )
    db._commit()


def _hull_patches(db: StructureDatabase, since: str) -> List[Dict[str, Any]]:
    """Hull distances changed on calculations not otherwise changed since
    ``since`` (those go whole, with the copy's other changes)."""
    changed = " OR ".join(f"c.{column} IS NOT f.{column}" for column in HULL_COLUMNS)
    rows = db.conn.execute(
        f"""
        SELECT c.structure_id, c.surface, {', '.join(f'c.{column}' for column in HULL_COLUMNS)}
        FROM calculations c
        JOIN _filled_hulls f ON f.structure_id = c.structure_id AND f.surface = c.surface
        WHERE c.updated_at < ? AND ({changed})
        """,
        (since,),
    )
    return [_to_ouro("calculations", dict(row)) for row in rows]


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
