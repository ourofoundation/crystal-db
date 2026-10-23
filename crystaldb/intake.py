"""Adding a structure only if the database does not have it yet.

Every structure is matched against all stored structures of its reduced
formula, published or made on Ouro (``crystaldb.matching``). A match is
returned instead of a second copy; the database only grows by new crystals,
and by copies of non-commercial data, which paid routes cannot see.
A structure resubmitted under the same ``(source, source_id)``, such as a
user's file that was replaced, updates its own row.

Generators store a run's relaxed structures with ``add_unique``: copies of
one crystal within the run collapse to the lowest in energy first, then
each goes through ``add_if_new``.

The energy given is the structure's raw energy on the database's surface,
on the given geometry (relaxed on that surface).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from pymatgen.core import Structure

from .matching import Match, NoveltyResult
from .validation import space_group

if TYPE_CHECKING:
    from .matching import NoveltyIndex
    from .database import StructureDatabase

ADDED = "added"
UPDATED = "updated"
EXISTS = "exists"

# Energies closer than this (eV/atom) count as the same score: repeated GPU
# runs of one structure differ in the last float32 digits.
SAME_ENERGY_EV = 1e-4
# Metadata that names the run, not the structure.
RUN_METADATA = ("action_id",)
# Relaxed structures within this energy (eV/atom) of each other are compared
# as possible copies of one crystal.
DUPLICATE_WINDOW_EV = 0.01


@dataclass
class Intake:
    status: str  # ADDED, UPDATED or EXISTS
    structure_id: str
    # The stored structure this one equals, when EXISTS.
    match: Optional[Match]
    # How the structure compares with published data (known, new polymorph, ...).
    novelty: NoveltyResult
    # False when the stored row already held this geometry, energy and
    # metadata (UPDATED with nothing written).
    changed: bool = True


def _unchanged(
    db: "StructureDatabase",
    structure_id: str,
    cif: str,
    energy_per_atom: float,
    metadata: Optional[Dict[str, Any]],
) -> bool:
    """Whether a stored row already holds this geometry, energy and metadata."""
    stored = db.get_structure(structure_id)
    if stored is None or not stored.is_valid or stored.energy_per_atom is None:
        return False
    if abs(stored.energy_per_atom - energy_per_atom) > SAME_ENERGY_EV:
        return False
    if db.get_geometry(structure_id, db.surface) != cif:
        return False

    def identity(meta):
        return {k: v for k, v in (meta or {}).items() if k not in RUN_METADATA}

    return identity(stored.generation_metadata) == identity(metadata)


def add_if_new(
    db: "StructureDatabase",
    structure: Structure,
    energy_per_atom: float,
    *,
    source: str,
    source_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    index: Optional["NoveltyIndex"] = None,
) -> Intake:
    """Store ``structure`` on ``db.surface`` unless an equal one is stored.

    When the equal structure has no energy on this surface yet, it takes this
    one's geometry and energy, so the work is not lost. ``run_id`` links the
    stored (or matched) structure to a generator run. ``index`` is the
    NoveltyIndex to match against (its candidates are cached per formula);
    by default a fresh one over ``db``.
    """
    novelty = (index or db.novelty_index()).check(structure)
    composition = structure.composition
    stoichiometry = {el.symbol: int(n) for el, n in composition.element_composition.items()}
    sg_number, sg_symbol = space_group(structure)
    cif = structure.to(fmt="cif")
    total = energy_per_atom * len(structure)
    changed = True

    own = None
    if source_id is not None:
        own = db.conn.execute(
            "SELECT id FROM structures WHERE source = ? AND source_id = ?",
            (source, source_id),
        ).fetchone()
    if own is not None and _unchanged(db, own["id"], cif, energy_per_atom, metadata):
        # The same file scored the same way again: nothing to write, so no
        # shard reaches the writer.
        sid, status, match, changed = own["id"], UPDATED, None, False
    elif own is not None:
        sid = own["id"]
        elements = sorted(stoichiometry)
        db.conn.execute(
            """
            UPDATE structures SET formula = ?, reduced_formula = ?, elements = ?,
                chemsys = ?, stoichiometry = ?, num_atoms = ?, space_group_number = ?,
                space_group_symbol = ?, is_valid = 1, error_message = NULL
            WHERE id = ?
            """,
            (
                composition.reduced_formula, composition.reduced_formula,
                json.dumps(elements), "-".join(elements), json.dumps(stoichiometry),
                len(structure), sg_number, sg_symbol, sid,
            ),
        )
        db._update_structure(
            sid,
            energy_per_atom=energy_per_atom,
            total_energy=total,
            cif_content=cif,
            generation_metadata=metadata or {},
        )
        status, match = UPDATED, None
    else:
        # A copy of non-commercial data (GNoME, ...) is stored anyway: paid
        # routes never see that data, so for them this is a new crystal. The
        # match still shows in ``novelty``.
        candidates = [m for m in (*novelty.matches, *novelty.ouro_matches) if not m.non_commercial]
        match = min(candidates, key=lambda m: m.rms_distance) if candidates else None
        if match is not None:
            sid = match.structure_id
            known = db.get_structure(sid)
            if known is not None and known.energy_per_atom is None:
                db.put_geometry(sid, db.surface, cif, defer_commit=True)
                db.set_calculation(
                    sid,
                    energy_per_atom,
                    total_energy=energy_per_atom * known.num_atoms,
                    structure=structure,
                    metadata={"geometry": db.surface, "from": source},
                    defer_commit=True,
                )
            status = EXISTS
        else:
            sid = db.add_structure(
                formula=composition.reduced_formula,
                stoichiometry=stoichiometry,
                energy_per_atom=energy_per_atom,
                total_energy=total,
                num_atoms=len(structure),
                space_group_number=sg_number,
                space_group_symbol=sg_symbol,
                structure=structure,
                cif_content=cif,
                generation_metadata=metadata,
                source=source,
                source_id=source_id,
                check_existing=False,
                defer_commit=True,
            )
            status = ADDED
    if run_id is not None:
        db.link_structure_to_run(run_id, sid)
    db._commit()
    return Intake(status=status, structure_id=sid, match=match, novelty=novelty, changed=changed)


@dataclass
class Generated:
    """A relaxed structure a generator wants stored."""

    structure: Structure
    # Raw energy per atom on the database's surface, on this geometry.
    energy_per_atom: float
    metadata: Optional[Dict[str, Any]] = None
    # An unconverged relaxation's energy is not final.
    converged: bool = True


def collapse(items: Sequence[Generated]) -> List[Tuple[Generated, int]]:
    """One representative per distinct crystal among structures of one
    reduced formula, with how many copies it stands for. Representatives
    are the lowest in energy of their copies, converged ones first."""
    from .matching import primitive, rms_distance

    reps: List[List[Any]] = []  # [item, primitive cell, copies]
    for item in sorted(items, key=lambda g: g.energy_per_atom):
        cell = primitive(item.structure)
        for rep in reps:
            if (
                item.energy_per_atom - rep[0].energy_per_atom <= DUPLICATE_WINDOW_EV
                and rms_distance(rep[1], cell) is not None
            ):
                rep[2] += 1
                break
        else:
            reps.append([item, cell, 1])
    reps.sort(key=lambda rep: (not rep[0].converged, rep[0].energy_per_atom))
    return [(item, copies) for item, _, copies in reps]


def add_unique(
    db: "StructureDatabase",
    items: Sequence[Generated],
    *,
    source: str,
    run_id: Optional[str] = None,
    keep: Optional[int] = None,
) -> List[Tuple[Generated, int, Intake]]:
    """Store a generator's relaxed structures, each distinct crystal once.

    They are grouped by reduced formula and collapsed (``collapse``); each
    representative goes through ``add_if_new``, so one equal to a stored
    structure (published, or from any earlier run) is linked to ``run_id``
    instead of stored again. ``keep`` limits each formula to its lowest
    converged representatives. Returns (representative, copies, intake).

    The representatives are matched against the database as it was when the
    call started, through one shared index: ``collapse`` already told them
    apart from each other, so matching each against the ones stored just
    before it would only repeat that work.
    """
    index = db.novelty_index()
    groups: Dict[str, List[Generated]] = {}
    for item in items:
        groups.setdefault(item.structure.composition.reduced_formula, []).append(item)
    out: List[Tuple[Generated, int, Intake]] = []
    for group in groups.values():
        reps = collapse(group)
        if keep is not None:
            reps = [rep for rep in reps if rep[0].converged][:keep]
        for item, copies in reps:
            intake = add_if_new(
                db,
                item.structure,
                item.energy_per_atom,
                source=source,
                metadata={**(item.metadata or {}), "copies": copies},
                run_id=run_id,
                index=index,
            )
            out.append((item, copies, intake))
    return out

