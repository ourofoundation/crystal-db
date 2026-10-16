"""Hull of a submitted structure, adding only new structures (EMT stands in for Orb)."""

from __future__ import annotations

from datetime import datetime

import pytest

ase = pytest.importorskip("ase")

from ase.calculators.emt import EMT  # noqa: E402
from pymatgen.core import Lattice, Structure  # noqa: E402
from pymatgen.io.ase import AseAtomsAdaptor  # noqa: E402

from crystaldb import StructureDatabase  # noqa: E402
from crystaldb.intake import ADDED, EXISTS, UPDATED, add_if_new  # noqa: E402
from crystaldb.references import ase_energies  # noqa: E402
from crystaldb.shared import SharedDatabase  # noqa: E402
from crystaldb.stability import HullError, assess  # noqa: E402
from crystaldb.surfaces import ORB_V3, PBE, SOURCE_GEOMETRY  # noqa: E402
from crystaldb.validation import InvalidStructure, read_cif  # noqa: E402


class Volume:
    def reload(self):
        pass

    def commit(self):
        pass


class LocalShared(SharedDatabase):
    """Working copies merge straight back, standing in for the remote writer."""

    def working_copy(self, chemical_systems, ingest=None, surface=None):
        return super().working_copy(
            chemical_systems, ingest=lambda shard: self.merge_shard(shard), surface=surface
        )


def fcc(element, a):
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), [element], [[0, 0, 0]])


def l12(majority, minority, a):
    return Structure(
        Lattice.cubic(a),
        [minority, majority, majority, majority],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )


def b2(a, b, lattice):
    return Structure(Lattice.cubic(lattice), [a, b], [[0, 0, 0], [0.5, 0.5, 0.5]])


def l10(a, b):
    return Structure(
        Lattice.tetragonal(2.55, 3.6), [a, b], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).get_sorted_structure()


def emt_energy(structure) -> float:
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = EMT()
    return float(atoms.get_potential_energy()) / len(structure)


def add_reference(db, source, source_id, structure, pbe_energy):
    comp = structure.composition
    sid = db.add_structure(
        formula=comp.reduced_formula,
        stoichiometry={el.symbol: int(n) for el, n in comp.items()},
        energy_per_atom=None,
        total_energy=None,
        num_atoms=len(structure),
        structure=structure,
        source=source,
        source_id=source_id,
        geometry_kind=SOURCE_GEOMETRY,
    )
    db.set_calculation(sid, pbe_energy, surface=PBE.id, structure=structure)
    return sid


def add_stored(db, structure, energy, source, source_id=None, space_group_symbol=None):
    comp = structure.composition
    return db.add_structure(
        formula=comp.reduced_formula,
        stoichiometry={el.symbol: int(n) for el, n in comp.items()},
        energy_per_atom=energy,
        total_energy=energy * len(structure),
        num_atoms=len(structure),
        structure=structure,
        source=source,
        source_id=source_id,
        space_group_symbol=space_group_symbol,
    )


@pytest.fixture
def shared(tmp_path):
    path = tmp_path / "shared.db"
    db = StructureDatabase(path)
    add_reference(db, "mp", "mp-30", fcc("Cu", 3.6), -3.7)
    add_reference(db, "mp", "mp-23", fcc("Ni", 3.52), -5.5)
    # Near the PBE hull (below the Cu-Ni tie line) and far above it.
    add_reference(db, "alexandria", "agm1", l12("Ni", "Cu", 3.54), -5.1)
    add_reference(db, "alexandria", "agm2", l12("Cu", "Ni", 3.58), -3.6)
    # An earlier submission and a GGen structure.
    add_stored(db, b2("Cu", "Ni", 2.85), 0.2, "ouro", "file-earlier")
    add_stored(db, l12("Cu", "Ni", 3.6), 0.4, "ggen")
    db.close()
    return LocalShared(Volume(), path)


def run(shared, structure, surface=ORB_V3.id, **kwargs):
    return assess(
        shared,
        structure,
        emt_energy(structure),
        ase_energies(EMT()),
        surface=ORB_V3.id,
        hull_surface=surface,
        source_id=kwargs.pop("source_id", "file-input"),
        **kwargs,
    )


def test_fills_near_hull_references_once(shared):
    result = run(shared, l10("Cu", "Ni"))
    # Cu, Ni and the near-hull CuNi3 were scored; Cu3Ni, 0.5 eV above, was not.
    assert (result.references_scored, result.references_missing) == (3, 0)
    assert result.intake.status == ADDED
    assert result.hull.references_by_source == {"mp": 2, "alexandria": 1, "ouro": 1, "ggen": 1}
    assert result.hull.user_contributions == ["ouro-file-earlier"]

    db = StructureDatabase(shared.path, surface=ORB_V3.id)
    assert db.get_structure_by_source("ouro", "file-input").energy_per_atom is not None
    scored = db.get_structure_by_source("alexandria", "agm1")
    assert scored.energy_per_atom is not None
    assert db.get_geometry(scored.id, ORB_V3.id) is None
    assert db.get_structure_by_source("alexandria", "agm2").energy_per_atom is None
    db.close()

    # Resubmitting the same file updates its own row, which is never a
    # reference against itself.
    again = run(shared, l10("Cu", "Ni"))
    assert again.intake.status == UPDATED
    assert again.references_scored == 0
    assert again.hull.e_above_hull == pytest.approx(result.hull.e_above_hull, abs=1e-6)
    assert "ouro-file-input" not in again.hull.user_contributions


def test_a_known_structure_is_not_stored_again(shared):
    # The earlier submission, relaxed a little differently.
    result = run(shared, b2("Cu", "Ni", 2.88))
    assert result.intake.status == EXISTS
    db = StructureDatabase(shared.path)
    assert result.intake.match.structure_id == db.get_structure_by_source("ouro", "file-earlier").id
    assert db.get_structure_by_source("ouro", "file-input") is None
    db.close()
    # The equal stored structure stays on the hull beside the input.
    assert "ouro-file-earlier" in result.hull.same_composition


def test_user_materials_can_be_left_out(shared):
    result = run(shared, l10("Cu", "Ni"), include_user=False)
    assert "ouro" not in result.hull.references_by_source
    assert result.hull.references_by_source["ggen"] == 1


def test_pbe_surface_uses_published_energies(shared):
    result = run(shared, l10("Cu", "Ni"), surface=PBE.id)
    assert result.hull_surface == PBE.id
    assert result.references_scored == 0
    assert result.hull.references_by_source == {"mp": 2, "alexandria": 2}


def test_generated_p1_left_out_by_default(shared):
    db = StructureDatabase(shared.path)
    add_stored(db, b2("Cu", "Ni", 3.0), 0.3, "matra-genoa", space_group_symbol="P1")
    db.close()
    assert "matra-genoa" not in run(shared, l10("Cu", "Ni")).hull.references_by_source
    included = run(shared, l10("Cu", "Ni"), include_generated_p1=True, source_id="file-2")
    assert included.hull.references_by_source["matra-genoa"] == 1


def test_missing_elemental_reference_is_a_hull_error(shared):
    with pytest.raises(HullError, match="no reference for Al"):
        run(shared, b2("Ni", "Al", 2.88))


def test_add_if_new_links_runs_and_fills_missing_energies(tmp_path):
    db = StructureDatabase(tmp_path / "work.db")
    published = add_reference(db, "mp", "mp-2213", b2("Fe", "Ni", 2.86), -7.0)
    run_id = db.create_run("Fe-Ni")
    found = add_if_new(db, b2("Fe", "Ni", 2.88), -7.1, source="mattergen", run_id=run_id)
    assert (found.status, found.structure_id) == (EXISTS, published)
    assert found.novelty.status == "known"
    # The published structure had no energy on this surface; it takes this one.
    assert db.get_structure(published).energy_per_atom == pytest.approx(-7.1)
    new = add_if_new(db, l10("Fe", "Ni"), -6.9, source="mattergen", run_id=run_id)
    assert new.status == ADDED and new.novelty.status == "new polymorph"
    assert db.get_structure(new.structure_id).source == "mattergen"
    linked = {r[0] for r in db.conn.execute("SELECT structure_id FROM run_structures")}
    assert linked == {published, new.structure_id}
    db.close()


def test_read_cif_refuses_overlapping_atoms():
    ok = read_cif(b2("Cu", "Ni", 2.88).to(fmt="cif"))
    assert ok.composition.reduced_formula == "CuNi"
    bad = Structure(Lattice.cubic(3.0), ["Cu", "Ni"], [[0, 0, 0], [0.05, 0, 0]])
    with pytest.raises(InvalidStructure, match="overlapping"):
        read_cif(bad.to(fmt="cif"))
    with pytest.raises(InvalidStructure, match="empty"):
        read_cif(b"  ")


def test_resubmitting_the_same_file_writes_nothing(tmp_path):
    db = StructureDatabase(tmp_path / "work.db")
    structure = b2("Fe", "Ni", 2.88)
    meta = {"file_id": "f1", "name": "FeNi", "action_id": "run-1"}
    first = add_if_new(db, structure, -7.1, source="ouro", source_id="f1", metadata=meta)
    assert first.status == ADDED
    since = datetime.now().isoformat()

    # Same geometry and score, from another run: no change to send.
    again = add_if_new(
        db, structure, -7.1 + 1e-6, source="ouro", source_id="f1", metadata={**meta, "action_id": "run-2"}
    )
    assert (again.status, again.structure_id) == (UPDATED, first.structure_id)
    assert db.export_changes(tmp_path / "delta.db", since) == 0

    # A new score is stored.
    add_if_new(db, structure, -7.2, source="ouro", source_id="f1", metadata=meta)
    assert db.get_structure(first.structure_id).energy_per_atom == pytest.approx(-7.2)
    assert db.export_changes(tmp_path / "delta.db", since) == 1
    db.close()
