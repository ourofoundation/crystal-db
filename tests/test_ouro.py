"""OuroSharedDatabase against a fake Ouro that runs the same SQL on SQLite.

The fake answers the two endpoints the store uses, the way the API does:
read-only SQL returns rows with JSON columns parsed and flags as booleans,
and upserts replace rows by primary key.
"""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from crystaldb.database import StructureDatabase
from crystaldb.ouro import PRIMARY_KEYS, OuroSharedDatabase
from crystaldb.testing import FakeOuro

FECO = Structure(Lattice.cubic(2.9), ["Fe", "Co"], [[0, 0, 0], [0.5] * 3]).composition.reduced_formula


def bcc(element, a=2.87):
    return Structure(Lattice.cubic(a), [element, element], [[0, 0, 0], [0.5, 0.5, 0.5]])


def b2(a_el, b_el, a=2.9):
    return Structure(Lattice.cubic(a), [a_el, b_el], [[0, 0, 0], [0.5, 0.5, 0.5]])


def add(db, structure, energy, **kwargs):
    comp = structure.composition
    return db.add_structure(
        formula=comp.formula.replace(" ", ""),
        stoichiometry={el.symbol: int(n) for el, n in comp.items()},
        energy_per_atom=energy,
        total_energy=None if energy is None else energy * len(structure),
        num_atoms=len(structure),
        structure=structure,
        **kwargs,
    )


@pytest.fixture
def fake(tmp_path):
    fake = FakeOuro(tmp_path)
    add(fake.db, bcc("Fe"), -8.3, source="mp", source_id="mp-13")
    add(fake.db, bcc("Co", 2.82), -7.1, source="mp", source_id="mp-54")
    add(fake.db, b2("Fe", "Co"), -7.9, source="mp", source_id="mp-2090")
    add(fake.db, bcc("Ni", 2.8), -5.5, source="mp", source_id="mp-23")
    run = fake.db.create_run("Co-Fe")
    add(fake.db, b2("Fe", "Co", 2.95), -7.85, run_id=run)
    return fake


def test_working_copy_holds_the_system_and_its_subsystems(fake):
    store = OuroSharedDatabase(fake, fake.ids)
    with store.working_copy(["Fe-Co"]) as db:
        formulas = sorted(s.reduced_formula for s in db.get_structures_for_subsystem("Co-Fe"))
        assert formulas == sorted(["Co", "Fe", FECO, FECO])
        assert db.get_runs_for_system("Co-Fe")
        # Geometries read back as structures, energies on the surface.
        stored = db.get_structure_by_source("mp", "mp-2090")
        assert stored.get_structure().composition.reduced_formula == FECO
        assert stored.energy_per_atom == pytest.approx(-7.9)
        assert db.compute_hull("Co-Fe", update_database=False)[0] is not None
    # Nothing changed, nothing sent.
    assert fake.server.execute("SELECT count(*) FROM structures").fetchone()[0] == 5


def test_changes_go_back(fake):
    store = OuroSharedDatabase(fake, fake.ids)
    with store.working_copy(["Co-Fe"]) as db:
        new_id = add(db, b2("Fe", "Co", 3.0), -7.5, source="matra-genoa")
        mp = db.get_structure_by_source("mp", "mp-54")
        db.set_calculation(mp.id, -7.2, total_energy=-14.4, structure=bcc("Co", 2.82))
    server = fake.db
    assert server.get_structure(new_id).source == "matra-genoa"
    assert server.get_structure_by_source("mp", "mp-54").energy_per_atom == pytest.approx(-7.2)


def test_a_stored_structure_under_a_new_id_merges_by_source_id(fake, tmp_path):
    store = OuroSharedDatabase(fake, fake.ids)
    stored = fake.db.get_structure_by_source("mp", "mp-13")
    delta = StructureDatabase(tmp_path / "delta.db")
    dup = add(delta, bcc("Fe"), -8.31, source="mp", source_id="mp-13")
    delta.close()
    result = store.push(tmp_path / "delta.db")
    assert result["merged_into_existing"] == 1 and result["new_structures"] == 0
    assert fake.db.get_structure(dup) is None
    merged = fake.db.get_structure(stored.id)
    # Its id and creation time stay; the newer energy lands on it.
    assert merged.created_at == stored.created_at
    assert merged.energy_per_atom == pytest.approx(-8.31)


def test_generated_duplicates_merge_by_structure_hash(fake):
    store = OuroSharedDatabase(fake, fake.ids)
    # Two runs make the same generated structure; the second lands on the first.
    with store.working_copy(["Ni"]) as a:
        first = add(a, bcc("Ni", 2.81), -5.4, source="ggen")
    with store.working_copy(["Ni"]) as b:
        b.conn.execute("DELETE FROM structures WHERE id = ?", (first,))
        b._commit()
        second = add(b, bcc("Ni", 2.81), -5.4, source="ggen", check_existing=False)
    assert first != second
    rows = fake.server.execute("SELECT id FROM structures WHERE source = 'ggen' AND chemsys = 'Ni'").fetchall()
    assert [r[0] for r in rows] == [first]


def test_an_older_copy_does_not_overwrite_a_newer_row(fake, tmp_path):
    store = OuroSharedDatabase(fake, fake.ids)
    stored = fake.db.get_structure_by_source("mp", "mp-23")
    fake.server.execute("UPDATE structures SET updated_at = '2999-01-01T00:00:00' WHERE id = ?", (stored.id,))
    fake.server.commit()
    delta = StructureDatabase(tmp_path / "delta.db")
    add(delta, bcc("Ni", 2.8), -5.5, source="mp", source_id="mp-23", check_existing=False,
        generation_metadata={"note": "stale"})
    delta.close()
    store.push(tmp_path / "delta.db")
    assert fake.db.get_structure(stored.id).generation_metadata.get("note") != "stale"


def test_reader_by_formula_and_novelty(fake):
    store = OuroSharedDatabase(fake, fake.ids)
    with store.reader(formulas=[FECO]) as db:
        result = db.novelty_index().check(b2("Fe", "Co", 2.92))
        assert result.status == "known"
        assert [m.source_id for m in result.matches] == ["mp-2090"]
        assert [m.structure_id for m in result.ouro_matches]  # the generated copy
        assert db.get_structure_by_source("mp", "mp-13") is None  # other formulas stay out


def test_pages_cover_every_row(fake, monkeypatch):
    import crystaldb.ouro as module

    monkeypatch.setattr(module, "PAGE_ROWS", 2)
    monkeypatch.setattr(module, "ID_CHUNK", 1)
    store = OuroSharedDatabase(fake, fake.ids)
    with store.reader(systems=["Co-Fe-Ni"]) as db:
        assert db.conn.execute("SELECT count(*) FROM structures").fetchone()[0] == 5
        assert db.conn.execute("SELECT count(*) FROM geometries").fetchone()[0] == 5
        assert db.conn.execute("SELECT count(*) FROM calculations").fetchone()[0] == 5


def test_source_counts(fake):
    counts = OuroSharedDatabase(fake, fake.ids).source_counts()
    assert counts["mp"]["structures"] == 4 and counts["ggen"]["structures"] == 1


def test_primary_keys_match_the_sqlite_schema(tmp_path):
    db = StructureDatabase(tmp_path / "x.db")
    for table, key in PRIMARY_KEYS.items():
        pk = [row[1] for row in sorted(db.conn.execute(f"PRAGMA table_info({table})"), key=lambda r: r[5]) if row[5]]
        assert tuple(pk) == key, table


def test_open_reader_removes_its_files_on_close(fake):
    import os

    db = OuroSharedDatabase(fake, fake.ids).open_reader(systems=["Fe"])
    folder = db.db_path.parent
    assert db.get_structure_by_source("mp", "mp-13") is not None
    db.close()
    assert not os.path.exists(folder)


@pytest.mark.parametrize("bulk_children", [0, 10**6])
def test_bulk_copy_of_a_source(fake, monkeypatch, bulk_children):
    import crystaldb.ouro as module

    # Child rows by key range (0) or by lists of ids (a million).
    monkeypatch.setattr(module, "BULK_CHILDREN", bulk_children)
    monkeypatch.setattr(module, "PAGE_ROWS", 2)
    store = OuroSharedDatabase(fake, fake.ids)
    with store.bulk_copy(sources=["mp"]) as db:
        count = lambda table: db.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        assert count("structures") == 4 and count("geometries") == 4 and count("calculations") == 4
        assert {row[0] for row in db.conn.execute("SELECT source FROM structures")} == {"mp"}
    with store.bulk_copy(surfaces=[], geometries=False) as db:
        assert db.conn.execute("SELECT count(*) FROM structures").fetchone()[0] == 5
        assert db.conn.execute("SELECT count(*) FROM geometries").fetchone()[0] == 0
        assert db.conn.execute("SELECT count(*) FROM calculations").fetchone()[0] == 0


def test_hull_distances_go_back_as_patches(fake, monkeypatch):
    import crystaldb.ouro as module

    monkeypatch.setattr(module, "BULK_CHILDREN", 0)
    store = OuroSharedDatabase(fake, fake.ids)
    feco = fake.db.get_structure_by_source("mp", "mp-2090")
    with store.bulk_copy(geometries=False) as db:
        db.conn.execute("UPDATE calculations SET e_above_hull = 0.123, is_on_hull = 0, hull_computed_at = 'x'")
        db._commit()
        # Meanwhile a route writes a new energy for one of them.
        fake.server.execute("UPDATE calculations SET energy_per_atom = -7.95 WHERE structure_id = ?", (feco.id,))
        fake.server.commit()
    rows = fake.server.execute("SELECT structure_id, energy_per_atom, e_above_hull, is_on_hull FROM calculations").fetchall()
    assert len(rows) == 5 and all(row["e_above_hull"] == pytest.approx(0.123) and row["is_on_hull"] == 0 for row in rows)
    # The patch carried the hull columns alone: the newer energy stays.
    assert fake.db.get_structure(feco.id).energy_per_atom == pytest.approx(-7.95)


def test_a_working_copy_sends_the_hulls_it_computes(fake):
    store = OuroSharedDatabase(fake, fake.ids)
    with store.working_copy(["Co-Fe"]) as db:
        db.compute_hull("Co-Fe")
    stored = fake.server.execute(
        "SELECT count(*) FROM calculations WHERE hull_computed_at IS NOT NULL"
    ).fetchone()[0]
    assert stored == 4  # Fe, Co and both FeCo; Ni is not in the system


def test_a_failed_bulk_job_sends_nothing(fake):
    store = OuroSharedDatabase(fake, fake.ids)
    with pytest.raises(RuntimeError):
        with store.bulk_copy(sources=["mp"]) as db:
            add(db, bcc("Ni", 2.9), -5.6, source="mp", source_id="mp-new")
            raise RuntimeError("import failed")
    assert fake.db.get_structure_by_source("mp", "mp-new") is None
