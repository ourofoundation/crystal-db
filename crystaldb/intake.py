"""Adding a structure only if the database does not have it yet.

Every structure is matched against all stored structures of its reduced
formula, published or made on Ouro (``crystaldb.matching``). A match is
returned instead of a second copy; the database only grows by new crystals.
A structure resubmitted under the same ``(source, source_id)``, such as a
user's file that was replaced, updates its own row.

The energy given is the structure's raw energy on the database's surface,
on the given geometry (relaxed on that surface).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from pymatgen.core import Structure

from .matching import Match, NoveltyResult
from .validation import space_group

if TYPE_CHECKING:
    from .database import StructureDatabase

ADDED = "added"
UPDATED = "updated"
EXISTS = "exists"


@dataclass
class Intake:
    status: str  # ADDED, UPDATED or EXISTS
    structure_id: str
    # The stored structure this one equals, when EXISTS.
    match: Optional[Match]
    # How the structure compares with published data (known, new polymorph, ...).
    novelty: NoveltyResult


def add_if_new(
    db: "StructureDatabase",
    structure: Structure,
    energy_per_atom: float,
    *,
    source: str,
    source_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Intake:
    """Store ``structure`` on ``db.surface`` unless an equal one is stored.

    When the equal structure has no energy on this surface yet, it takes this
    one's geometry and energy, so the work is not lost. ``run_id`` links the
    stored (or matched) structure to a generator run.
    """
    novelty = db.novelty_index().check(structure)
    composition = structure.composition
    stoichiometry = {el.symbol: int(n) for el, n in composition.element_composition.items()}
    sg_number, sg_symbol = space_group(structure)
    cif = structure.to(fmt="cif")
    total = energy_per_atom * len(structure)

    own = None
    if source_id is not None:
        own = db.conn.execute(
            "SELECT id FROM structures WHERE source = ? AND source_id = ?",
            (source, source_id),
        ).fetchone()
    if own is not None:
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
        candidates = [*novelty.matches, *novelty.ouro_matches]
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
    return Intake(status=status, structure_id=sid, match=match, novelty=novelty)
