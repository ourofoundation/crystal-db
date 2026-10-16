#!/usr/bin/env python3
"""
Import reference structures and their PBE energies into a crystaldb database.

Supports importing from:
  - Materials Project (provider='mp'): GGA/GGA+U entries, MP's own MP2020
    corrections
  - Alexandria local .json.bz2 chunks (provider='alexandria'): raw PBE
    energies, MP2020 applied here

Each structure is stored as published (the ``source`` geometry) with a
``pbe`` calculation; see ``crystaldb/sources.py``. Re-running an import refreshes
existing rows in place, so it doubles as an update: identity fields and
metadata are rewritten, records retired earlier come back if they are
importable again, and a record whose structure changed loses its MLIP
energies and geometries (they were for the old structure). With
``--retire-missing`` a full import also retires records the source no longer
publishes. Structures then need hull distances and MLIP energies for MLIP
hulls, e.g. on the ones within 100 meV/atom of the PBE hull:

    python scripts/published_hulls.py --database structures.db
    # then single-points near the hull on each MLIP surface

Usage:
    # Materials Project (MP_API_KEY in the environment or --api-key)
    python import.py --database structures.db --provider mp

    # Alexandria from downloaded chunk files
    python import.py --database structures.db --provider alexandria \\
        --alexandria-local-dir alexandria_pbe_2025_07_02

    # Filter to materials containing these elements
    python import.py --database structures.db --provider mp --elements Fe Co

    # Dry run: count what would be imported
    python import.py --database structures.db --provider alexandria \\
        --alexandria-local-dir ... --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
import warnings
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from crystaldb import StructureDatabase
from crystaldb.database import compress_cif
from crystaldb.sources import (
    NON_COMMERCIAL_LICENSES,
    SourceRecord,
    alexandria,
    alexandria_chunk,
    alexandria_chunks,
    chunk_records,
    get_source,
    materials_project,
)
from crystaldb.surfaces import PBE, SOURCE_GEOMETRY

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _space_group_symbol(number: int) -> Optional[str]:
    # Building a SpaceGroup costs ~30 ms, most of an import's time uncached.
    from pymatgen.symmetry.groups import SpaceGroup

    try:
        return SpaceGroup.from_int_number(int(number)).symbol
    except Exception:
        return None


_NOW = time.strftime("%Y-%m-%dT%H:%M:%S")


def prepare(record: SourceRecord, source: str) -> Optional[Dict[str, Any]]:
    """A record as ready-to-insert values, computed off the main process."""
    from crystaldb.corrections import mp2020_correction

    if record.license in NON_COMMERCIAL_LICENSES:
        return {"source_id": record.source_id, "excluded_license": record.license}

    structure = record.structure
    composition = structure.composition
    try:
        cif = structure.to(fmt="cif")  # unsymmetrized: the cell the energy is for
    except Exception:
        return None

    sg_number, sg_symbol = record.space_group_number, record.space_group_symbol
    if sg_number is None:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
            sg_number = analyzer.get_space_group_number()
            sg_symbol = analyzer.get_space_group_symbol()
        except Exception:
            pass
    if sg_number is not None and sg_symbol is None:
        sg_symbol = _space_group_symbol(sg_number)

    correction = record.correction_per_atom
    run_type = (record.parameters or {}).get("run_type")
    hubbards = (record.parameters or {}).get("hubbards")
    if record.energy_per_atom is not None and correction is None:
        result = mp2020_correction(structure, record.energy_per_atom, record.parameters)
        correction, run_type, hubbards = result.per_atom, result.run_type, result.hubbards

    stoichiometry = {
        el.symbol: int(round(n)) for el, n in composition.element_composition.items()
    }
    elements = sorted(stoichiometry)
    formula = composition.reduced_formula
    return {
        "source_id": record.source_id,
        "formula": formula,
        "reduced_formula": formula,
        "elements": json.dumps(elements),
        "chemsys": "-".join(elements),
        "stoichiometry": json.dumps(stoichiometry),
        "num_atoms": len(structure),
        "space_group_number": sg_number,
        "space_group_symbol": sg_symbol,
        "structure_hash": StructureDatabase.compute_structure_hash(
            formula, 0.0, sg_number, source, record.source_id
        ),
        "metadata": json.dumps(
            {**record.metadata, "license": record.license or get_source(source).license}
        ),
        "cif": compress_cif(cif),
        "energy_per_atom": record.energy_per_atom,
        "correction_per_atom": correction,
        "run_type": run_type,
        "hubbards": json.dumps(hubbards or {}),
    }


def prepare_batch(records: List[SourceRecord], source: str) -> List[Dict[str, Any]]:
    return [p for p in (prepare(r, source) for r in records) if p is not None]


def prepare_alexandria_chunk(path: str, elements: Optional[List[str]]) -> List[Dict[str, Any]]:
    """One Alexandria chunk, read, parsed and prepared in a worker."""
    return prepare_batch(list(alexandria_chunk(Path(path), elements)), "alexandria")


# A refreshed record whose PBE energy moved by more than this (eV/atom), or
# whose cell size changed, is a different structure from the stored one.
CHANGED_ENERGY_EV = 1e-4


@dataclass
class Existing:
    id: str
    num_atoms: int
    pbe_energy: Optional[float]


def load_existing(db: StructureDatabase, source: str) -> Dict[str, Existing]:
    return {
        row[0]: Existing(row[1], row[2], row[3])
        for row in db.conn.execute(
            """
            SELECT s.source_id, s.id, s.num_atoms, c.energy_per_atom
            FROM structures s
            LEFT JOIN calculations c ON c.structure_id = s.id AND c.surface = ?
            WHERE s.source = ? AND s.source_id IS NOT NULL
            """,
            (PBE.id, source),
        )
    }


def _changed(old: Existing, row: Dict[str, Any]) -> bool:
    if old.num_atoms != row["num_atoms"]:
        return True
    if old.pbe_energy is None or row["energy_per_atom"] is None:
        return old.pbe_energy != row["energy_per_atom"]
    return abs(old.pbe_energy - row["energy_per_atom"]) > CHANGED_ENERGY_EV


def write(
    db: StructureDatabase,
    source: str,
    existing: Dict[str, Existing],
    rows,
    seen: Optional[set] = None,
) -> Dict[str, int]:
    """Insert prepared rows in bulk: new structures, refreshed ones, retired ones."""
    conn = db.conn
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    retired, new, refreshed, changed = [], [], [], []
    for row in rows:
        if seen is not None:
            seen.add(row["source_id"])
        old = existing.get(row["source_id"])
        if "excluded_license" in row:
            if old is not None:
                retired.append((f"license {row['excluded_license']}", now, old.id))
            continue
        if old is None:
            sid = str(uuid.uuid4())
            existing[row["source_id"]] = Existing(sid, row["num_atoms"], row["energy_per_atom"])
            new.append((sid, row))
        else:
            refreshed.append((old.id, row))
            if _changed(old, row):
                changed.append((old.id,))
            existing[row["source_id"]] = Existing(old.id, row["num_atoms"], row["energy_per_atom"])

    conn.executemany(
        "UPDATE structures SET is_valid = 0, error_message = ?, updated_at = ? WHERE id = ?",
        retired,
    )
    conn.executemany(
        """
        INSERT INTO structures (
            id, source, source_id, formula, reduced_formula, elements, chemsys,
            stoichiometry, num_atoms, space_group_number, space_group_symbol,
            structure_hash, is_valid, error_message, metadata, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, ?, ?, ?)
        """,
        (
            (
                sid, source, r["source_id"], r["formula"], r["reduced_formula"],
                r["elements"], r["chemsys"], r["stoichiometry"], r["num_atoms"],
                r["space_group_number"], r["space_group_symbol"], r["structure_hash"],
                r["metadata"], now, now,
            )
            for sid, r in new
        ),
    )
    # Everything about the record comes from the source; metadata keys set
    # elsewhere (a migration, a backfill) are kept unless the source sets them.
    conn.executemany(
        """
        UPDATE structures SET formula = ?, reduced_formula = ?, elements = ?,
            chemsys = ?, stoichiometry = ?, num_atoms = ?,
            space_group_number = COALESCE(?, space_group_number),
            space_group_symbol = COALESCE(?, space_group_symbol),
            structure_hash = ?, is_valid = 1, error_message = NULL,
            metadata = json_patch(COALESCE(metadata, '{}'), ?), updated_at = ?
        WHERE id = ?
        """,
        (
            (
                r["formula"], r["reduced_formula"], r["elements"], r["chemsys"],
                r["stoichiometry"], r["num_atoms"], r["space_group_number"],
                r["space_group_symbol"], r["structure_hash"], r["metadata"], now, sid,
            )
            for sid, r in refreshed
        ),
    )
    # Energies and relaxed geometries on other surfaces were for the old
    # structure; the single-point job fills them again.
    conn.executemany(
        "DELETE FROM calculations WHERE structure_id = ? AND surface != 'pbe'", changed
    )
    conn.executemany(
        "DELETE FROM geometries WHERE structure_id = ? AND kind != 'source'", changed
    )
    conn.executemany(
        "INSERT OR REPLACE INTO geometries (structure_id, kind, cif, updated_at) VALUES (?, ?, ?, ?)",
        ((sid, SOURCE_GEOMETRY, r["cif"], now) for sid, r in new + refreshed),
    )
    conn.executemany(
        """
        INSERT INTO calculations (
            structure_id, surface, energy_per_atom, total_energy, correction_per_atom,
            run_type, hubbards, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (structure_id, surface) DO UPDATE SET
            energy_per_atom = excluded.energy_per_atom,
            total_energy = excluded.total_energy,
            correction_per_atom = excluded.correction_per_atom,
            run_type = excluded.run_type,
            hubbards = excluded.hubbards,
            e_above_hull = NULL, is_on_hull = NULL, hull_computed_at = NULL,
            updated_at = excluded.updated_at
        """,
        (
            (
                sid, PBE.id, r["energy_per_atom"], r["energy_per_atom"] * r["num_atoms"],
                r["correction_per_atom"], r["run_type"], r["hubbards"], now, now,
            )
            for sid, r in new + refreshed
            if r["energy_per_atom"] is not None
        ),
    )
    conn.commit()
    return {
        "new": len(new),
        "updated": len(refreshed),
        "changed": len(changed),
        "excluded": len(retired),
    }


def retire_missing(db: StructureDatabase, source: str, existing: Dict[str, Existing], seen: set) -> int:
    """Retire this source's records that the import did not see."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    gone = [
        ("no longer published by the source", now, old.id)
        for source_id, old in existing.items()
        if source_id not in seen
    ]
    db.conn.executemany(
        "UPDATE structures SET is_valid = 0, error_message = ?, updated_at = ? "
        "WHERE id = ? AND is_valid = 1",
        gone,
    )
    db.conn.commit()
    return len(gone)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database", required=True)
    parser.add_argument("--provider", choices=["mp", "alexandria"], default="mp")
    parser.add_argument("--api-key", type=str, default=None, help="Materials Project API key")
    parser.add_argument("--alexandria-local-dir", type=str, default=None)
    parser.add_argument("--alexandria-local-pattern", type=str, default="alexandria_*.json.bz2")
    parser.add_argument("--elements", type=str, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retire-missing",
        action="store_true",
        help="Retire records the source no longer publishes (full imports only)",
    )
    args = parser.parse_args()

    source = args.provider
    if get_source(source).license in NON_COMMERCIAL_LICENSES:
        parser.error(f"{source} is licensed {get_source(source).license}; Ouro's routes are paid")
    partial = bool(args.elements) or (
        source == "alexandria" and args.alexandria_local_pattern != "alexandria_*.json.bz2"
    )
    if args.retire_missing and partial:
        parser.error("--retire-missing needs a full import (no --elements or file pattern)")
    if source == "mp":
        api_key = args.api_key or os.getenv("MP_API_KEY")
        if not api_key:
            parser.error("Set MP_API_KEY or pass --api-key")
        records = materials_project(api_key, elements=args.elements)
    else:
        if not args.alexandria_local_dir:
            parser.error("provider=alexandria requires --alexandria-local-dir")
        records = alexandria(
            Path(args.alexandria_local_dir),
            pattern=args.alexandria_local_pattern,
            elements=args.elements,
        )

    db = StructureDatabase(args.database)
    existing = load_existing(db, source)
    seen: set = set()
    logger.info(f"{source} import into {args.database}")
    logger.info(f"  {len(existing):,} {source} structures already in the database")

    if args.dry_run:
        seen = new = 0
        for record in records:
            seen += 1
            new += record.source_id not in existing
        logger.info(f"  {seen:,} records, {new:,} new, {seen - new:,} would be refreshed")
        db.close()
        return

    totals = {"new": 0, "updated": 0, "changed": 0, "excluded": 0}
    started = time.time()
    with Pool(args.workers) as pool:
        pending = deque()

        def drain_one():
            counts = write(db, source, existing, pending.popleft().get(), seen)
            for key in totals:
                totals[key] += counts[key]
            done = totals["new"] + totals["updated"]
            logger.info(
                f"  {done:,} written ({totals['new']:,} new) "
                f"at {done / (time.time() - started):,.0f}/s"
            )

        if source == "alexandria":
            # Whole chunk files per worker: reading and parsing them is most
            # of the work, so it must not happen in the main process.
            for path in alexandria_chunks(Path(args.alexandria_local_dir), args.alexandria_local_pattern):
                pending.append(
                    pool.apply_async(prepare_alexandria_chunk, (str(path), args.elements))
                )
                if len(pending) >= args.workers:
                    drain_one()
        else:
            for batch in chunk_records(records, args.batch):
                pending.append(pool.apply_async(prepare_batch, (batch, source)))
                if len(pending) >= args.workers * 2:
                    drain_one()
        while pending:
            drain_one()

    retired = retire_missing(db, source, existing, seen) if args.retire_missing else 0
    if not partial:
        # When the reference set was last brought up to date (the novelty
        # route reports it), so partial imports leave it alone.
        db.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (f"imported:{source}", time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        db.conn.commit()
    db.analyze()
    db.close()
    logger.info("")
    logger.info(
        f"Done: {totals['new']:,} new, {totals['updated']:,} refreshed "
        f"({totals['changed']:,} with a changed structure), "
        f"{totals['excluded']:,} left out for their license, {retired:,} retired as gone "
        f"in {time.time() - started:.0f}s"
    )
    logger.info(
        f"Next: python scripts/published_hulls.py --database {args.database}, "
        "then the singlepoint_near_hull job"
    )


if __name__ == "__main__":
    main()
