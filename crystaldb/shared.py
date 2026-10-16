"""One structure database shared by several Modal apps on one Volume.

A Modal Volume keeps the last commit of a file, so two containers writing
the SQLite file would silently drop each other's rows. Only one deployed
writer (a Modal class with ``max_containers=1`` and a ``drain`` method that
calls ``SharedDatabase.drain``) writes it. Everyone else:

- reads a snapshot: ``reader()`` reloads the Volume (at most every
  ``reload_interval`` seconds, and only when nothing in the container has
  the file open) and opens it read-only;
- or, to write, works on a local copy: ``working_copy(systems)`` extracts
  the systems to local disk, yields a writable database, then exports what
  changed as a shard file into the Volume's inbox (``submit``) and wakes the
  writer.

The writer's ``drain`` merges every shard waiting in the inbox on one local
copy of the database. A merge costs a copy of the whole file each way, so
shards that queue up while the writer is busy are merged together, not one
copy per shard.

Usage, in any app that mounts the Volume::

    shared = SharedDatabase(volume, "/data/structures.db", writer=("my-app", "Writer"))
    with shared.reader() as db:
        ...
    with shared.working_copy(["Fe-Co"]) as db:
        ...  # changes reach the writer on exit

Set ``CRYSTALDB_WRITER_APP`` to another deployed app name to send writes
there instead, or to ``none`` to keep a deployment's writes out (a dev
deployment, say).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
import uuid
import weakref
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple, Union

from .database import StructureDatabase
from .surfaces import DEFAULT_SURFACE

logger = logging.getLogger(__name__)

WRITER_APP_ENV = "CRYSTALDB_WRITER_APP"

INBOX = "inbox"
# Shards that failed to merge, kept for a look instead of blocking the inbox.
FAILED = "failed"


class SharedDatabase:
    def __init__(
        self,
        volume: Any,
        path: Union[str, Path],
        writer: Optional[Tuple[str, str]] = None,
        reload_interval: float = 30.0,
    ):
        """``writer`` is the deployed (app, class) that merges shards; without
        one, working copies' changes are not sent anywhere."""
        self.volume = volume
        self.path = Path(path)
        self.writer = writer
        self.reload_interval = reload_interval
        self._lock = threading.Lock()
        # Weak references, so a reader a request forgot to close stops
        # blocking reloads once it is garbage.
        self._readers: "weakref.WeakSet[StructureDatabase]" = weakref.WeakSet()
        self._last_reload = 0.0

    def _maybe_reload(self) -> None:
        if time.time() - self._last_reload < self.reload_interval:
            return
        if any(reader._conn is not None for reader in list(self._readers)):
            return
        try:
            self.volume.reload()
            self._last_reload = time.time()
        except Exception as exc:
            # Files open elsewhere in the container block a reload; the
            # current view is then used until the next request.
            logger.info("Volume reload skipped: %s", exc)

    def open_reader(self, surface: Optional[str] = None) -> StructureDatabase:
        """A read-only snapshot. Close it when done."""
        with self._lock:
            self._maybe_reload()
            db = StructureDatabase(
                self.path,
                surface=surface or DEFAULT_SURFACE,
                readonly=True,
                snapshot=True,
            )
            db.conn  # open now, so a concurrent reload sees it
            self._readers.add(db)
        return db

    @contextmanager
    def reader(self, surface: Optional[str] = None) -> Iterator[StructureDatabase]:
        db = self.open_reader(surface)
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def working_copy(
        self,
        chemical_systems: Iterable[str],
        ingest: Optional[Callable[[bytes], dict]] = None,
        surface: Optional[str] = None,
    ) -> Iterator[StructureDatabase]:
        """A writable local copy of some systems; changes go to the writer on exit.

        The copy holds every structure in the systems and their subsystems,
        so hulls built from it match the shared database. Changes are sent
        even when the block raises, so work done before a failure is kept.
        ``ingest`` replaces ``submit`` with a callable taking the shard's
        bytes (tests pass a no-op).
        """
        surface = surface or DEFAULT_SURFACE
        systems = list(chemical_systems)
        with tempfile.TemporaryDirectory(prefix="crystaldb-work-") as tmp:
            work_path = Path(tmp) / "work.db"
            extracting = time.time()
            with self.reader(surface) as reader:
                count = reader.extract_systems(work_path, systems)
            logger.info(
                "Working copy of %s: %d structures in %.1f s",
                ", ".join(systems), count, time.time() - extracting,
            )

            since = datetime.now().isoformat()
            db = StructureDatabase(work_path, surface=surface)
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
                            result = self.submit(delta_path)
                        logger.info(
                            "Sent %d changed structures to the writer in %.1f s: %s",
                            changed, time.time() - sending, result,
                        )
                    except Exception:
                        logger.exception("Could not send %d changed structures to the writer", changed)

    @property
    def inbox(self) -> Path:
        return self.path.parent / INBOX

    def writer_app(self) -> Optional[str]:
        """The app whose writer owns the file, or None when writes are off."""
        import os

        app_name = os.getenv(WRITER_APP_ENV) or (self.writer[0] if self.writer else None)
        return None if app_name in (None, "none") else app_name

    def wake_writer(self) -> Optional[str]:
        """Ask the deployed writer to drain the inbox; returns the call id.

        Not awaited: the writer takes one call at a time and may be busy with
        a long maintenance task. Drains that find the inbox empty return at
        once.
        """
        import modal

        app_name = self.writer_app()
        if app_name is None or self.writer is None:
            return None
        return modal.Cls.from_name(app_name, self.writer[1])().drain.spawn().object_id

    def submit(self, shard_path: Union[str, Path]) -> dict:
        """Put a shard in the inbox on the Volume and wake the writer.

        The file is written under a temporary name and renamed, so neither a
        background commit nor the writer ever sees it half-written.
        """
        if self.writer_app() is None:
            logger.info("%s=none: shard not sent", WRITER_APP_ENV)
            return {"queued": None}
        self.inbox.mkdir(parents=True, exist_ok=True)
        name = f"{datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:12]}.db"
        partial = self.inbox / f"{name}.part"
        shutil.copyfile(shard_path, partial)
        partial.replace(self.inbox / name)
        self.volume.commit()
        return {"queued": name, "drain_call": self.wake_writer()}

    @contextmanager
    def exclusive_local_copy(self) -> Iterator[Path]:
        """The whole file on local disk, written back and committed on success.

        Every write by the writer goes through this: SQLite is much faster on
        local disk, the Volume only ever sees a complete file (swapped in by
        rename), and since the writer is the only writer nothing changes the
        file in the meantime.
        """
        with tempfile.TemporaryDirectory(prefix="crystaldb-maintenance-") as tmp:
            local = Path(tmp) / self.path.name
            self.volume.reload()
            shutil.copyfile(self.path, local)
            yield local
            db = StructureDatabase(local)
            db.record_source_counts()
            db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.close()
            staged = self.path.with_suffix(".staged")
            shutil.copyfile(local, staged)
            staged.replace(self.path)
            # The writer checkpoints after every merge, so these are empty;
            # removed so they can never be replayed against the new file.
            for suffix in ("-wal", "-shm"):
                Path(f"{self.path}{suffix}").unlink(missing_ok=True)
            self.volume.commit()

    def merge_shard(self, shard: bytes) -> dict:
        """Put one shard (as bytes) in the inbox and drain it here: a
        working copy's ``ingest`` when this process is the writer (tests,
        local databases)."""
        self.inbox.mkdir(parents=True, exist_ok=True)
        name = f"{datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:12]}.db"
        (self.inbox / f"{name}.part").write_bytes(shard)
        (self.inbox / f"{name}.part").replace(self.inbox / name)
        # Committed first, so the drain's Volume reload keeps it.
        self.volume.commit()
        return self.merge_directory(self.inbox)

    def drain(self) -> dict:
        """Merge every shard in the inbox. Only the writer calls this."""
        return self.merge_directory(self.inbox)

    def merge_directory(self, folder: Union[str, Path], analyze: bool = False) -> dict:
        """Merge every ``*.db`` shard in ``folder`` on one local copy, then remove them.

        The merge runs on a local copy that replaces the file in one rename,
        because a Volume can commit in the background at any moment and must
        never capture the SQLite file half-written. A shard that fails to
        merge moves to ``failed/`` beside it rather than blocking the rest.

        Stored hull distances are not refreshed: routes compute hulls fresh,
        and the writer's ``hulls`` maintenance task refreshes them in bulk.
        """
        started = time.time()
        folder = Path(folder)
        self.volume.reload()
        if not any(folder.glob("*.db")):
            return {"shards": 0, "seconds": round(time.time() - started, 1)}

        new = merged = 0
        done: list = []
        failed: list = []
        with self.exclusive_local_copy() as local:
            # Listed after the copy's reload, so every committed shard shows.
            shards = sorted(folder.glob("*.db"))
            db = StructureDatabase(local)
            try:
                for i, shard in enumerate(shards, 1):
                    try:
                        n, m = db.merge_from(shard, recompute_hulls=False)
                    except Exception:
                        logger.exception("Shard %s failed to merge", shard.name)
                        failed.append(shard)
                        continue
                    new, merged = new + n, merged + m
                    done.append(shard)
                    if i % 10 == 0:
                        logger.info("merged %d/%d shards", i, len(shards))
                if analyze:
                    db.analyze()
            finally:
                db.close()
        # After the new file is in place: a shard is only removed once its
        # rows are in, and merging one again changes nothing.
        for shard in done:
            shard.unlink(missing_ok=True)
        if failed:
            (folder / FAILED).mkdir(exist_ok=True)
            for shard in failed:
                shard.replace(folder / FAILED / shard.name)
        self.volume.commit()
        return {
            "shards": len(done),
            "failed": [s.name for s in failed],
            "new_structures": new,
            "merged_into_existing": merged,
            "seconds": round(time.time() - started, 1),
        }
