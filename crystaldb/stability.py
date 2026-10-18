"""Energy above the hull for a submitted structure.

``assess`` is the whole job a hull route does: correct the structure's
energy with MP2020, store it in the shared database if it is new
(``crystaldb.intake``), fill missing reference energies near the PBE hull
(``crystaldb.references``), and place it on the hull of its chemical system
built from every stored structure there.

The hull can be built on a different surface than the one the structure was
scored on: on ``pbe`` the references are the published DFT energies and the
structure keeps its MLIP energy, so model error does not cancel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedEntry

from .corrections import Correction, mp2020_correction
from .intake import ADDED, UPDATED, Intake, add_if_new
from .references import Energies, fill_near_hull
from .sources import USER_SOURCE, is_generated, is_published
from .surfaces import get_surface
from .validation import space_group

if TYPE_CHECKING:
    from .database import StructureDatabase
    from .ouro import OuroSharedDatabase

INPUT_ENTRY_ID = "input_structure"
# Most missing reference single-points done per call; the rest wait for the
# bulk job (or the next call).
FILL_LIMIT = 2000


class HullError(ValueError):
    """The structure cannot be placed on a hull (no references, MP2020 rejects it)."""


@dataclass
class HullResult:
    e_above_hull: float  # eV/atom, >= 0
    formation_energy_per_atom: float
    decomposition: Dict[str, float]  # reduced formula -> fraction
    # No stored structure has this composition.
    new_composition: bool
    same_composition: List[str]  # entry ids of stored structures at this composition
    # Lowest stored structure at this composition (entry id, formation energy).
    lowest_at_composition: Tuple[str, float]
    input_is_lowest: bool
    phase_diagram: PhaseDiagram
    input_entry: ComputedEntry
    references: List[ComputedEntry]
    references_by_source: Dict[str, int]
    # Entry ids of user submissions among the references ("ouro-<source_id>").
    user_contributions: List[str]
    # References left out because MP2020 rejects them.
    incompatible: int


@dataclass
class Assessment:
    hull: HullResult
    intake: Intake
    correction: Correction
    references_scored: int = 0
    references_missing: int = 0
    surface: str = ""
    hull_surface: str = ""


def entry_id(row) -> str:
    """How a stored structure is labeled in a phase diagram.

    Published structures by their own id (``mp-149`` links to Materials
    Project); user submissions as ``ouro-<file id>``; anything else as
    ``<source>-<structure id>``.
    """
    if row["source"] == USER_SOURCE:
        return f"ouro-{row['source_id']}"
    if is_published(row["source"]) and row["source_id"]:
        return row["source_id"]
    return f"{row['source'] or 'ggen'}-{row['id']}"


def reference_entries(
    db: "StructureDatabase",
    chemical_system: str,
    *,
    include_user: bool = True,
    include_generated_p1: bool = False,
    exclude_ids: Sequence[str] = (),
) -> Tuple[List[ComputedEntry], Dict[str, int], List[str], int]:
    """Hull entries for every structure in a system on ``db.surface``.

    Returns (entries, count by source, user contribution entry ids, number
    left out because MP2020 rejects them).
    """
    excluded = set(exclude_ids)

    def keep(r) -> bool:
        if r["id"] in excluded:
            return False
        if not include_user and r["source"] == USER_SOURCE:
            return False
        # Generated P1 structures are mostly symmetry-broken leftovers;
        # published P1 phases stay.
        if not include_generated_p1 and r["space_group_symbol"] == "P1" and is_generated(r["source"]):
            return False
        return True

    rows = [r for r in db.hull_rows(chemical_system) if keep(r)]
    incompatible = sum(1 for r in rows if r["correction_per_atom"] is None)
    entries: List[ComputedEntry] = []
    by_source: Dict[str, int] = {}
    contributions: List[str] = []
    for r in rows:
        if r["correction_per_atom"] is None:
            continue
        label = entry_id(r)
        if r["source"] == USER_SOURCE:
            contributions.append(label)
        entries.append(db.hull_entry(r, entry_id=label))
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    return entries, by_source, contributions, incompatible


def place_on_hull(
    structure: Structure,
    corrected_energy_per_atom: float,
    references: List[ComputedEntry],
) -> Tuple[PhaseDiagram, ComputedEntry, float, float, Dict[str, float]]:
    """Hull of the references (lowest per formula) plus the structure.

    Returns (diagram, input entry, e above hull, formation energy,
    decomposition).
    """
    best: Dict[str, ComputedEntry] = {}
    for entry in references:
        key = entry.composition.reduced_formula
        if key not in best or entry.energy_per_atom < best[key].energy_per_atom:
            best[key] = entry
    entry = ComputedEntry(
        structure.composition,
        corrected_energy_per_atom * len(structure),
        entry_id=INPUT_ENTRY_ID,
        data={"space_group": space_group(structure)[1]},
    )
    try:
        diagram = PhaseDiagram(list(best.values()) + [entry])
        decomposition, e_above = diagram.get_decomp_and_e_above_hull(entry)
        formation = diagram.get_form_energy_per_atom(entry)
    except Exception as exc:
        have = {e.composition.elements[0].symbol for e in references if e.is_element}
        missing = sorted({el.symbol for el in structure.composition.elements} - have)
        detail = f"No phase diagram: {exc}"
        if missing:
            detail += f" (no reference for {', '.join(missing)})"
        raise HullError(detail) from None
    fractions = {e.composition.reduced_formula: float(a) for e, a in (decomposition or {}).items()}
    return diagram, entry, max(float(e_above or 0.0), 0.0), float(formation or 0.0), fractions


def assess(
    shared: "OuroSharedDatabase",
    structure: Structure,
    energy_per_atom: float,
    energies: Energies,
    *,
    surface: str,
    hull_surface: Optional[str] = None,
    source: str = USER_SOURCE,
    source_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    include_user: bool = True,
    include_generated_p1: bool = False,
    fill_limit: int = FILL_LIMIT,
    log: Callable[[str], None] = lambda message: None,
) -> Assessment:
    """Store a scored structure if new and place it on its system's hull.

    ``energy_per_atom`` is the structure's raw energy on ``surface`` (an
    MLIP surface); ``energies`` scores missing references on that surface.
    ``hull_surface`` (default ``surface``) is the surface the hull is built
    on.
    """
    from .database import StructureDatabase
    from .ouro import stored_id

    hull_surface = get_surface(hull_surface or surface).id
    correction = mp2020_correction(structure, energy_per_atom)
    if correction.per_atom is None:
        raise HullError(
            f"MP2020 corrections reject {structure.composition.reduced_formula}, so it "
            "cannot share a hull with the reference data"
        )
    chemsys = "-".join(sorted({el.symbol for el in structure.composition.elements}))
    scored = missing = 0
    formula = structure.composition.reduced_formula
    # Matching needs the geometries of this formula only, and the hull none;
    # references about to be scored get theirs when needed.
    with shared.working_copy([chemsys], surface=surface, geometries=[formula], runs=False) as work:
        intake = add_if_new(
            work, structure, energy_per_atom, source=source, source_id=source_id, metadata=metadata
        )
        if hull_surface == work.surface:
            scored, missing = fill_near_hull(
                work,
                chemsys,
                energies,
                limit=fill_limit,
                prepare=lambda ids: shared.load_geometries(work, ids),
            )
            if scored:
                log(f"Scored {scored} reference phase(s) near the PBE hull with no {get_surface(surface).label} energy yet")
        hull_db = (
            work
            if hull_surface == work.surface
            else StructureDatabase(work.db_path, surface=hull_surface, readonly=True)
        )
        # The structure itself is the input entry; an equal stored structure
        # (EXISTS) stays among the references.
        own = (intake.structure_id,) if intake.status in (ADDED, UPDATED) else ()
        try:
            references, by_source, contributions, incompatible = reference_entries(
                hull_db,
                chemsys,
                include_user=include_user,
                include_generated_p1=include_generated_p1,
                exclude_ids=own,
            )
        finally:
            if hull_db is not work:
                hull_db.close()

    intake.structure_id = stored_id(work, intake.structure_id)
    corrected = energy_per_atom + correction.per_atom
    diagram, entry, e_above, formation, decomposition = place_on_hull(
        structure, corrected, references
    )
    same = [e for e in references if e.composition.reduced_formula == formula]
    if same:
        lowest = min(same, key=diagram.get_form_energy_per_atom)
        lowest_formation = diagram.get_form_energy_per_atom(lowest)
        lowest_at = (lowest.entry_id, float(lowest_formation))
        input_is_lowest = formation <= lowest_formation + 1e-6
    else:
        lowest_at, input_is_lowest = (INPUT_ENTRY_ID, formation), True
    hull = HullResult(
        e_above_hull=e_above,
        formation_energy_per_atom=formation,
        decomposition=decomposition,
        new_composition=not same,
        same_composition=[e.entry_id for e in same],
        lowest_at_composition=lowest_at,
        input_is_lowest=input_is_lowest,
        phase_diagram=diagram,
        input_entry=entry,
        references=references,
        references_by_source=by_source,
        user_contributions=contributions,
        incompatible=incompatible,
    )
    return Assessment(
        hull=hull,
        intake=intake,
        correction=correction,
        references_scored=scored,
        references_missing=missing,
        surface=get_surface(surface).id,
        hull_surface=hull_surface,
    )
