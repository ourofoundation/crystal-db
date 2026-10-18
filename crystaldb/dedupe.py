"""Merging stored generated structures that are copies of one crystal.

Generators store through ``crystaldb.intake.add_unique``, so a run adds each
crystal once; this cleans up what earlier runs stored. For each reduced
formula with generated structures (``plan``):

- a generated structure equal to a published or submitted one is merged
  into it (published first, closest first);
- copies among generated structures are merged into the lowest in energy.

"Equal" is ``crystaldb.matching``'s rule, the one ``add_if_new`` uses.
Published and submitted structures are never merged away.

``apply`` sends the merges to the shared datasets: each copy's run links
move to the structure it is merged into, an energy (with its geometry) on a
surface that structure has none for moves too, and then the copy's rows are
deleted.

    with store.reader(formulas=generated_formulas(store)) as db:
        merges = plan(db)
        apply(store, db, merges)  # or report summarize(merges) (a dry run)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence

from .matching import find_matches
from .sources import is_generated, is_published

if TYPE_CHECKING:
    from .database import StructureDatabase
    from .ouro import OuroSharedDatabase

logger = logging.getLogger(__name__)

# Stored hull distances refer to the copy's own row; a moved energy drops them.
HULL_FIELDS = ("e_above_hull", "is_on_hull", "hull_computed_at")


@dataclass
class Merge:
    duplicate: str  # the generated structure merged away
    source: str  # its source (ggen, matra-genoa, ...)
    keeper: str  # the structure it is merged into
    formula: str
    # What it is merged into: "published", "submitted" or "generated".
    into: str
    rms_distance: float


def generated_formulas(store: "OuroSharedDatabase") -> List[str]:
    """Reduced formulas with at least one generated structure."""
    from .sources import SOURCES

    published_or_user = [s for s in SOURCES if not is_generated(s)]
    rows = store.query(
        f"SELECT DISTINCT reduced_formula FROM {store.table('structures')} "
        f"WHERE is_valid AND source NOT IN ({', '.join(repr(s) for s in published_or_user)})"
    )
    return sorted(row["reduced_formula"] for row in rows)


def plan(
    db: "StructureDatabase",
    formulas: Optional[Iterable[str]] = None,
    progress: Callable[[int, int], None] = lambda done, total: None,
) -> List[Merge]:
    """The merges that leave one structure per distinct crystal, for every
    formula in ``db`` with generated structures (or ``formulas``). ``db``
    must hold every structure of those formulas, with geometries."""
    if formulas is None:
        rows = db.conn.execute("SELECT DISTINCT reduced_formula, source FROM structures WHERE is_valid = 1")
        formulas = sorted({row[0] for row in rows if is_generated(row[1])})
    formulas = list(formulas)
    energies = dict(
        db.conn.execute(
            "SELECT structure_id, energy_per_atom FROM calculations "
            "WHERE surface = ? AND energy_per_atom IS NOT NULL",
            (db.surface,),
        ).fetchall()
    )
    merges: List[Merge] = []
    for done, formula in enumerate(formulas, 1):
        candidates = db.get_match_candidates(formula)
        fixed = [c for c in candidates if not is_generated(c.source)]
        generated = sorted(
            (c for c in candidates if is_generated(c.source)),
            key=lambda c: (c.structure_id not in energies, energies.get(c.structure_id, 0.0), c.structure_id),
        )
        kept = []
        for candidate in generated:
            cell = candidate.primitive()
            if cell is None:
                kept.append(candidate)
                continue
            matches = find_matches(cell, fixed) or find_matches(cell, kept)
            if not matches:
                kept.append(candidate)
                continue
            match = matches[0]
            into = (
                "published" if is_published(match.source)
                else "generated" if is_generated(match.source)
                else "submitted"
            )
            merges.append(
                Merge(candidate.structure_id, candidate.source, match.structure_id, formula, into, match.rms_distance)
            )
        progress(done, len(formulas))
    return merges


def _plan_formulas(args) -> List[Merge]:
    from .database import StructureDatabase

    path, surface, formulas = args
    db = StructureDatabase(path, surface=surface, readonly=True)
    try:
        return plan(db, formulas)
    finally:
        db.close()


def plan_parallel(db: "StructureDatabase", workers: int, log: Callable[[str], None] = logger.info) -> List[Merge]:
    """``plan`` with the formulas split across ``workers`` processes, each
    reading ``db``'s file. Formulas are dealt out busiest first, so the
    largest ones do not end up in one process."""
    from concurrent.futures import ProcessPoolExecutor

    rows = db.conn.execute("SELECT reduced_formula, source FROM structures WHERE is_valid = 1").fetchall()
    sizes = Counter(row[0] for row in rows)
    generated = {row[0] for row in rows if is_generated(row[1])}
    order = sorted(generated, key=lambda f: -sizes[f])
    groups = [order[i::workers] for i in range(workers)]
    db._commit()
    merges: List[Merge] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, found in enumerate(pool.map(_plan_formulas, [(str(db.db_path), db.surface, g) for g in groups if g]), 1):
            merges += found
            log(f"Planned {done}/{workers} formula groups: {len(merges)} merges so far")
    return merges


def summarize(merges: Sequence[Merge]) -> Dict[str, Any]:
    return {
        "merges": len(merges),
        "by_source": dict(Counter(m.source for m in merges)),
        "into": dict(Counter(m.into for m in merges)),
        "formulas": len({m.formula for m in merges}),
    }


def apply(store: "OuroSharedDatabase", db: "StructureDatabase", merges: Sequence[Merge]) -> Dict[str, int]:
    """Send ``merges`` (from ``plan`` on ``db``, a copy of the shared
    datasets) to the shared datasets. Rows that keep information are written
    first, so a failure part-way loses nothing; then the copies' rows are
    deleted."""
    from .ouro import _to_ouro

    keeper_of = {m.duplicate: m.keeper for m in merges}
    if not keeper_of:
        return {}

    def rows(table: str, ids: Sequence[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            out += [dict(r) for r in db.conn.execute(f"SELECT * FROM {table} WHERE structure_id IN ({marks})", chunk)]
        return out

    duplicates = list(keeper_of)
    keepers = sorted(set(keeper_of.values()))
    held_calcs = {(r["structure_id"], r["surface"]) for r in rows("calculations", keepers) if r["energy_per_atom"] is not None}
    held_geoms = {(r["structure_id"], r["kind"]) for r in rows("geometries", keepers)}
    held_links = {(r["run_id"], r["structure_id"]) for r in rows("run_structures", keepers)}

    dup_calcs, dup_geoms, dup_links = rows("calculations", duplicates), rows("geometries", duplicates), rows("run_structures", duplicates)
    geoms_by = {(r["structure_id"], r["kind"]): r for r in dup_geoms}

    # Energies on surfaces the keeper has none for, with the geometry they were computed on.
    moved_calcs, moved_geoms = [], []
    for calc in sorted(dup_calcs, key=lambda r: (r["energy_per_atom"] is None, r["energy_per_atom"] or 0.0)):
        keeper = keeper_of[calc["structure_id"]]
        if calc["energy_per_atom"] is None or (keeper, calc["surface"]) in held_calcs:
            continue
        held_calcs.add((keeper, calc["surface"]))
        moved_calcs.append(_to_ouro("calculations", {**calc, "structure_id": keeper, **{f: None for f in HULL_FIELDS}}))
        geometry = geoms_by.get((calc["structure_id"], calc["surface"]))
        if geometry is not None and (keeper, geometry["kind"]) not in held_geoms:
            held_geoms.add((keeper, geometry["kind"]))
            moved_geoms.append(_to_ouro("geometries", {**geometry, "structure_id": keeper}))
    moved_links = []
    for link in dup_links:
        key = (link["run_id"], keeper_of[link["structure_id"]])
        if key not in held_links:
            held_links.add(key)
            moved_links.append(_to_ouro("run_structures", {**link, "structure_id": key[1]}))

    counts = {
        "geometries_moved": store.upsert("geometries", moved_geoms),
        "calculations_moved": store.upsert("calculations", moved_calcs),
        "run_links_moved": store.upsert("run_structures", moved_links),
    }
    counts["run_links_deleted"] = store.delete("run_structures", dup_links)
    counts["calculations_deleted"] = store.delete("calculations", dup_calcs)
    counts["geometries_deleted"] = store.delete("geometries", dup_geoms)
    counts["structures_deleted"] = store.delete("structures", [{"id": sid} for sid in duplicates])
    logger.info("Applied %d merges: %s", len(merges), counts)
    return counts
