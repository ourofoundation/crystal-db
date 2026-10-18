"""OuroSharedDatabase against a fake Ouro that runs the same SQL on SQLite.

The fake answers the two endpoints the store uses, the way the API does:
read-only SQL returns rows with JSON columns parsed and flags as booleans,
and upserts replace rows by primary key.
"""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from crystaldb.database import StructureDatabase
from crystaldb.ouro import PRIMARY_KEYS, OuroSharedDatabase, stored_id
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


def test_a_lean_copy_reads_only_what_it_needs(fake):
    store = OuroSharedDatabase(fake, fake.ids)
    count = lambda db, table: db.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    with store.working_copy(["Co-Fe"], geometries=[FECO], runs=False) as db:
        assert count(db, "structures") == 4 and count(db, "calculations") == 4
        # Geometries of the formula being matched only, and no runs.
        assert count(db, "geometries") == 2
        assert count(db, "runs") == 0 and count(db, "run_structures") == 0
        fe = db.get_structure_by_source("mp", "mp-13")
        assert db.get_geometry(fe.id) is None
        assert store.load_geometries(db, [fe.id]) == 1
        assert db.get_geometry(fe.id) is not None
        before = fake.server.execute("SELECT updated_at FROM geometries WHERE structure_id = ?", (fe.id,)).fetchone()[0]
    # Loaded geometries are not changes: nothing went back.
    after = fake.server.execute("SELECT updated_at FROM geometries WHERE structure_id = ?", (fe.id,)).fetchone()[0]
    assert after == before
    assert fake.server.execute("SELECT count(*) FROM run_structures").fetchone()[0] == 1


def test_concurrent_adds_of_one_file_report_the_stored_id(fake):
    store = OuroSharedDatabase(fake, fake.ids)
    # Both copies are filled before either sends: each adds the file as new.
    with store.working_copy(["Ni"]) as first:
        with store.working_copy(["Ni"]) as second:
            winner = add(second, bcc("Ni", 2.83), -5.45, source="ouro", source_id="file-1")
        loser = add(first, bcc("Ni", 2.83), -5.45, source="ouro", source_id="file-1")
    rows = fake.server.execute("SELECT id FROM structures WHERE source = 'ouro'").fetchall()
    assert [row[0] for row in rows] == [winner]
    assert loser != winner
    assert stored_id(first, loser) == winner and stored_id(second, winner) == winner
    # The loser's energy landed on the stored row, under its id.
    assert fake.server.execute(
        "SELECT count(*) FROM calculations WHERE structure_id = ?", (loser,)
    ).fetchone()[0] == 0


def test_one_row_pages_read_every_row(fake, monkeypatch):
    import crystaldb.ouro as module

    # A page of one row: the structures after the first come by key range,
    # and their child rows are looked up page by page.
    monkeypatch.setattr(module, "PAGE_ROWS", 1)
    store = OuroSharedDatabase(fake, fake.ids)
    with store.reader(systems=["Co-Fe"]) as db:
        for table, expected in (("structures", 4), ("geometries", 4), ("calculations", 4), ("run_structures", 1), ("runs", 1)):
            assert db.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == expected, table


def test_an_insert_that_clashes_merges_and_retries(fake, monkeypatch):
    store = OuroSharedDatabase(fake, fake.ids)
    lookup = store._existing
    calls = []

    def stale_first_lookup(structures, filled):
        # The first lookup ran before the other run's insert landed.
        calls.append(1)
        return {} if len(calls) == 1 else lookup(structures, filled)

    with store.working_copy(["Ni"]) as first:
        with store.working_copy(["Ni"]) as second:
            winner = add(second, bcc("Ni", 2.83), -5.45, source="ouro", source_id="file-1")
        monkeypatch.setattr(store, "_existing", stale_first_lookup)
        loser = add(first, bcc("Ni", 2.83), -5.45, source="ouro", source_id="file-1")
    assert len(calls) == 2  # the upsert clashed, and the merge ran again
    assert stored_id(first, loser) == winner
    assert fake.server.execute("SELECT count(*) FROM structures WHERE source = 'ouro'").fetchone()[0] == 1


def test_a_run_of_known_structures_still_sends_its_links(fake):
    store = OuroSharedDatabase(fake, fake.ids)
    feco = fake.db.get_structure_by_source("mp", "mp-2090")
    with store.working_copy(["Co-Fe"]) as db:
        run_id = db.create_run("Co-Fe")
        # Nothing new is stored: the run only finds a published structure.
        db.link_structure_to_run(run_id, feco.id)
        db.complete_run(run_id, num_candidates=1, num_successful=0, num_failed=0)
    assert fake.server.execute("SELECT count(*) FROM runs WHERE id = ?", (run_id,)).fetchone()[0] == 1
    links = fake.server.execute("SELECT structure_id FROM run_structures WHERE run_id = ?", (run_id,)).fetchall()
    assert [row[0] for row in links] == [feco.id]
