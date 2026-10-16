"""Reference energies an MLIP hull needs before it can be trusted.

A hull on an MLIP surface is only as complete as its reference phases: a
published phase near the PBE hull with no energy on the surface would leave
a hole the hull silently closes over. Every route that builds such a hull
fills those first, here, the same way:

- which phases: valid structures within ``NEAR_HULL_EV`` of the PBE hull of
  the system (computed fresh, so new imports count) with no energy on the
  database's surface;
- how: a single-point at the published geometry. Relaxing first changes ORB
  energies of these phases by a median 0.2 meV/atom (p99 3.9), far below
  the model's own ~8 meV/atom scatter against PBE;
- stored as the surface's energy with ``{"geometry": "source", "relaxed":
  False}``, the convention the bulk ``singlepoint_near_hull`` job uses too.

Callers bring the energies (an ASE calculator, a batched torch-sim model via
``crystaldb.relaxation.singlepoint``), so this module needs no GPU stack.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

from pymatgen.core import Structure

from .surfaces import PBE, SOURCE_GEOMETRY

if TYPE_CHECKING:
    from .database import StructureDatabase

logger = logging.getLogger(__name__)

# PBE phases this close to the PBE hull (eV/atom) must have an energy on an
# MLIP surface before that surface's hull is used.
NEAR_HULL_EV = 0.1
SINGLEPOINT_METADATA = {"geometry": SOURCE_GEOMETRY, "relaxed": False}

# Total energies (eV) for structures, in order; None where scoring failed.
Energies = Callable[[Sequence[Structure]], List[Optional[float]]]


def _chunks(items: Sequence[str], size: int = 500):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def missing_near_hull(
    db: "StructureDatabase", chemical_system: str, near_hull_ev: float = NEAR_HULL_EV
) -> List[str]:
    """Ids near the PBE hull of a system with no energy on ``db.surface``, closest first."""
    from .database import StructureDatabase

    pbe = StructureDatabase(db.db_path, surface=PBE.id, readonly=True)
    try:
        _, pbe_hull, _ = pbe.compute_hull(chemical_system, update_database=False)
    finally:
        pbe.close()
    near = [sid for e, sid in sorted((e, sid) for sid, e in pbe_hull.items()) if e <= near_hull_ev]
    have = set()
    for chunk in _chunks(near):
        have.update(
            row[0]
            for row in db.conn.execute(
                f"""
                SELECT structure_id FROM calculations
                WHERE surface = ? AND energy_per_atom IS NOT NULL
                  AND structure_id IN ({','.join('?' * len(chunk))})
                """,
                [db.surface, *chunk],
            )
        )
    return [sid for sid in near if sid not in have]


def reference_structure(db: "StructureDatabase", structure_id: str) -> Optional[Structure]:
    """A structure's published geometry, or None if it does not read back as itself.

    pymatgen occasionally reads a stored CIF with merged or extra sites,
    changing the formula; an energy for that would be saved under the
    reference's id, so such reads are refused. Cells larger than the stored
    one (some CIFs read back as conventional cells) are reduced.
    """
    row = db.conn.execute(
        "SELECT reduced_formula, num_atoms FROM structures WHERE id = ?", (structure_id,)
    ).fetchone()
    cif = db.get_geometry(structure_id, SOURCE_GEOMETRY)
    if row is None or cif is None:
        return None
    try:
        structure = Structure.from_str(cif, fmt="cif")
    except Exception as exc:
        logger.debug("Unreadable geometry for %s: %s", structure_id, exc)
        return None
    if structure.composition.reduced_formula != row["reduced_formula"]:
        return None
    if len(structure) > row["num_atoms"]:
        primitive = structure.get_primitive_structure()
        if len(primitive) < len(structure):
            structure = primitive
    return structure


def store_singlepoint(
    db: "StructureDatabase",
    structure_id: str,
    structure: Structure,
    total_energy: float,
    surface: Optional[str] = None,
) -> None:
    """Record a single-point at the published geometry as the surface's energy."""
    energy_per_atom = total_energy / len(structure)
    num_atoms = db.conn.execute(
        "SELECT num_atoms FROM structures WHERE id = ?", (structure_id,)
    ).fetchone()[0]
    db.set_calculation(
        structure_id,
        energy_per_atom,
        surface=surface,
        # Per the stored cell, which may differ from the one scored.
        total_energy=energy_per_atom * num_atoms,
        structure=structure,
        metadata=SINGLEPOINT_METADATA,
        defer_commit=True,
    )


def fill_near_hull(
    db: "StructureDatabase",
    chemical_system: str,
    energies: Energies,
    limit: int = 2000,
    near_hull_ev: float = NEAR_HULL_EV,
) -> Tuple[int, int]:
    """Score the system's missing near-hull references on ``db.surface``.

    At most ``limit`` per call (the closest to the hull first); the rest
    wait for a later run or the bulk job. Returns (scored, still_missing).
    """
    missing = missing_near_hull(db, chemical_system, near_hull_ev)
    if not missing:
        return 0, 0
    todo = []
    for sid in missing[:limit]:
        structure = reference_structure(db, sid)
        if structure is not None:
            todo.append((sid, structure))
    scored = 0
    if todo:
        for (sid, structure), energy in zip(todo, energies([s for _, s in todo])):
            if energy is None:
                logger.warning("Reference single-point failed for %s", sid)
                continue
            store_singlepoint(db, sid, structure, energy)
            scored += 1
        db._commit()
    return scored, len(missing) - scored


def ase_energies(calculator) -> Energies:
    """``Energies`` from an ASE calculator, one structure at a time."""
    from pymatgen.io.ase import AseAtomsAdaptor

    def energies(structures: Sequence[Structure]) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for structure in structures:
            try:
                atoms = AseAtomsAdaptor.get_atoms(structure)
                atoms.calc = calculator
                out.append(float(atoms.get_potential_energy()))
            except Exception as exc:
                logger.debug("Single-point failed: %s", exc)
                out.append(None)
        return out

    return energies
