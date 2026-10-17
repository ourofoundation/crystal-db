"""Keeping the shared database consistent: sources, merges, imports."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from crystaldb.database import StructureDatabase
from crystaldb.matching import KNOWN, NEW_COMPOSITION, REFERENCE
from crystaldb.references import SINGLEPOINT_METADATA, fill_near_hull, missing_near_hull
from crystaldb.testing import FakeOuro
from crystaldb.sources import is_generated, is_published
from crystaldb.surfaces import ORB_V3, PBE, SOURCE_GEOMETRY


def bcc(element: str, a: float = 2.87) -> Structure:
    return Structure(Lattice.cubic(a), [element, element], [[0, 0, 0], [0.5, 0.5, 0.5]])


def b2(a: str, b: str, lattice: float = 2.9) -> Structure:
    return Structure(Lattice.cubic(lattice), [a, b], [[0, 0, 0], [0.5, 0.5, 0.5]])


def add(db, structure, energy, source="ggen", **kwargs):
    comp = structure.composition
    return db.add_structure(
        formula=comp.formula.replace(" ", ""),
        stoichiometry={el.symbol: int(n) for el, n in comp.items()},
        energy_per_atom=energy,
        total_energy=None if energy is None else energy * len(structure),
        num_atoms=len(structure),
        structure=structure,
        source=source,
        **kwargs,
    )


# -------------------- Sources --------------------


def test_unregistered_sources_are_generated_never_prior_art(tmp_path):
    assert is_published("mp") and is_published("alexandria")
    assert not is_published("ouro") and not is_generated("ouro")
    assert is_generated("ggen") and is_generated("mattergen")
    assert not is_published("mattergen")

    db = StructureDatabase(tmp_path / "ggen.db")
    add(db, bcc("Fe"), -8.0, source="mp", source_id="mp-13")
    add(db, bcc("Ni", 2.8), -5.0, source="mp", source_id="mp-23")
    # A new model's structure, then a second model finding the same crystal.
    first = add(db, b2("Fe", "Ni"), -7.0, source="mattergen", space_group_symbol="Pm-3m")
    second = add(db, b2("Fe", "Ni"), -7.0001, source="chemeleon", space_group_symbol="P1")
    index = db.novelty_index()
    assert index.classify(db.get_structure_by_source("mp", "mp-13")) == REFERENCE
    assert index.classify(db.get_structure(second)) == NEW_COMPOSITION
    assert {m.structure_id for m in index.check(b2("Fe", "Ni")).ouro_matches} == {first, second}
    # Generated P1 from an unregistered model is left out like GGen's.
    _, e_hull, counts = db.compute_hull("Fe-Ni", update_database=False, include_generated_p1=False)
    assert second not in e_hull and counts["excluded_generated_p1"] == 1
    db.close()


# -------------------- Merges never undo newer writes --------------------


def test_a_working_copy_does_not_revert_a_newer_import(tmp_path):
    fake = FakeOuro(tmp_path / "main.db")
    fe = add(fake.db, bcc("Fe"), -8.0, source="mp", source_id="mp-13", geometry_kind=SOURCE_GEOMETRY)
    store = fake.store()

    with store.working_copy(["Fe"]) as work:
        # While a worker holds its copy, an import refreshes the published
        # structure and retires it...
        importer = StructureDatabase(fake.path, surface=PBE.id)
        importer.put_geometry(fe, SOURCE_GEOMETRY, bcc("Fe", 2.9).to(fmt="cif"))
        importer.conn.execute(
            "UPDATE structures SET is_valid = 0, updated_at = ? WHERE id = ?",
            (datetime.now().isoformat(), fe),
        )
        importer.conn.commit()
        importer.close()

        # ...then the worker adds an ORB energy for the reference.
        work.set_calculation(fe, -8.3, surface=ORB_V3.id, metadata=SINGLEPOINT_METADATA)

    main = StructureDatabase(fake.path)
    row = main.conn.execute("SELECT is_valid FROM structures WHERE id = ?", (fe,)).fetchone()
    assert row["is_valid"] == 0
    geometry = Structure.from_str(main.get_geometry(fe, SOURCE_GEOMETRY), fmt="cif")
    assert geometry.lattice.a == pytest.approx(2.9)
    main.close()
    main = StructureDatabase(fake.path, surface=ORB_V3.id)
    assert main.get_structure(fe).energy_per_atom == pytest.approx(-8.3)
    main.close()


def test_failed_send_does_not_fail_the_run(tmp_path):
    store = FakeOuro(tmp_path / "main.db").store()

    def broken(shard):
        raise RuntimeError("Ouro unreachable")

    with store.working_copy(["Fe"], ingest=broken) as work:
        add(work, bcc("Fe"), -8.0)


# -------------------- Near-hull references --------------------


def test_fill_near_hull_scores_missing_references_once(tmp_path):
    db = StructureDatabase(tmp_path / "ggen.db", surface=PBE.id)
    fe = add(db, bcc("Fe"), -8.0, source="mp", source_id="mp-13", geometry_kind=SOURCE_GEOMETRY)
    ni = add(db, bcc("Ni", 2.8), -5.0, source="mp", source_id="mp-23", geometry_kind=SOURCE_GEOMETRY)
    near = add(db, b2("Fe", "Ni"), -6.55, source="alexandria", source_id="agm1", geometry_kind=SOURCE_GEOMETRY)
    add(db, b2("Fe", "Ni", 3.0), -6.0, source="alexandria", source_id="agm2", geometry_kind=SOURCE_GEOMETRY)
    db.close()

    orb = StructureDatabase(tmp_path / "ggen.db", surface=ORB_V3.id)
    # agm2 is 0.5 eV/atom above the PBE hull, so it is not needed.
    assert set(missing_near_hull(orb, "Fe-Ni")) == {fe, ni, near}
    calls = []

    def energies(structures):
        calls.append(len(structures))
        return [-1.0 * len(s) for s in structures]

    assert fill_near_hull(orb, "Fe-Ni", energies) == (3, 0)
    stored = orb.get_structure(near)
    assert stored.energy_per_atom == pytest.approx(-1.0)
    meta = orb.conn.execute(
        "SELECT metadata FROM calculations WHERE structure_id = ? AND surface = ?",
        (near, ORB_V3.id),
    ).fetchone()[0]
    assert json.loads(meta) == SINGLEPOINT_METADATA
    assert fill_near_hull(orb, "Fe-Ni", energies) == (0, 0)
    assert calls == [3]
    orb.close()


# -------------------- Re-imports --------------------


def _import_module():
    path = Path(__file__).parent.parent / "scripts" / "import.py"
    spec = importlib.util.spec_from_file_location("ggen_import", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reimport_refreshes_rows_and_drops_stale_mlip_data(tmp_path):
    imp = _import_module()
    from crystaldb.sources import SourceRecord

    db = StructureDatabase(tmp_path / "ggen.db", surface=PBE.id)

    def rows(records):
        return imp.prepare_batch(records, "alexandria")

    def record(source_id, structure, energy, **meta):
        return SourceRecord(
            "alexandria", source_id, structure, energy, metadata=meta, space_group_number=229
        )

    existing = imp.load_existing(db, "alexandria")
    imp.write(db, "alexandria", existing, rows([
        record("a1", bcc("Fe"), -8.0, alexandria_e_above_hull=0.0),
        record("a2", bcc("Ni", 2.8), -5.0),
    ]))
    a1 = db.get_structure_by_source("alexandria", "a1").id
    a2 = db.get_structure_by_source("alexandria", "a2").id
    db.set_calculation(a1, -8.4, surface=ORB_V3.id)
    db.set_calculation(a2, -5.4, surface=ORB_V3.id)
    db.conn.execute("UPDATE structures SET is_valid = 0, error_message = 'license BY-NC' WHERE id = ?", (a2,))
    db.conn.commit()

    # Next release: a1 is a different (larger) cell, a2 is unchanged and
    # importable again, a3 is gone.
    existing = imp.load_existing(db, "alexandria")
    seen = set()
    supercell = bcc("Fe").copy()
    supercell.make_supercell([2, 1, 1])
    counts = imp.write(db, "alexandria", existing, rows([
        record("a1", supercell, -8.1, alexandria_e_above_hull=0.02),
        record("a2", bcc("Ni", 2.8), -5.0),
    ]), seen)
    assert counts == {"new": 0, "updated": 2, "changed": 1, "excluded": 0}

    s1 = db.get_structure(a1)
    assert s1.num_atoms == 4
    assert s1.generation_metadata["alexandria_e_above_hull"] == 0.02
    assert StructureDatabase(tmp_path / "ggen.db", surface=ORB_V3.id, readonly=True).get_structure(a1).energy_per_atom is None
    s2 = db.get_structure(a2)
    assert s2.is_valid and s2.error_message is None
    orb = StructureDatabase(tmp_path / "ggen.db", surface=ORB_V3.id, readonly=True)
    assert orb.get_structure(a2).energy_per_atom == pytest.approx(-5.4)
    orb.close()

    imp.write(db, "alexandria", existing, rows([record("a3", bcc("Co", 2.8), -7.0)]), set())
    existing = imp.load_existing(db, "alexandria")
    assert imp.retire_missing(db, "alexandria", existing, seen) == 1
    assert not db.get_structure_by_source("alexandria", "a3").is_valid
    db.close()
