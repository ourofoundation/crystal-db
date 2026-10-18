"""Merging stored generated copies of one crystal (crystaldb.dedupe)."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from crystaldb.dedupe import apply, generated_formulas, plan, summarize
from crystaldb.testing import FakeOuro


def b2(a, b, lattice):
    return Structure(Lattice.cubic(lattice), [a, b], [[0, 0, 0], [0.5, 0.5, 0.5]])


def l10(a, b, c=3.6):
    return Structure(Lattice.tetragonal(2.55, c), [a, b], [[0, 0, 0], [0.5, 0.5, 0.5]])


def double_layers(a, b):
    """FeNi stacked AABB along c: another ordering, four atoms per primitive cell."""
    return Structure(
        Lattice.tetragonal(2.55, 7.2), [a, a, b, b],
        [[0, 0, 0], [0.5, 0.5, 0.25], [0, 0, 0.5], [0.5, 0.5, 0.75]],
    )


def add(db, structure, energy, **kwargs):
    comp = structure.composition
    return db.add_structure(
        formula=comp.reduced_formula,
        stoichiometry={el.symbol: int(n) for el, n in comp.items()},
        energy_per_atom=energy,
        total_energy=None if energy is None else energy * len(structure),
        num_atoms=len(structure),
        structure=structure,
        check_existing=False,
        **kwargs,
    )


@pytest.fixture
def fake(tmp_path):
    fake = FakeOuro(tmp_path)
    db = fake.db
    ids = {}
    # Published B2 FeNi without an energy on the database's surface.
    ids["mp"] = add(db, b2("Fe", "Ni", 2.86), None, source="mp", source_id="mp-2213")
    run_a, run_b = db.create_run("Fe-Ni"), db.create_run("Fe-Ni")
    # ggen stored the B2 twice, and an L1_0 three times (one in a doubled cell).
    ids["b2_high"] = add(db, b2("Fe", "Ni", 2.88), -7.0, source="ggen", run_id=run_a)
    ids["b2_low"] = add(db, b2("Fe", "Ni", 2.87), -7.1, source="ggen", run_id=run_b)
    ids["l10_low"] = add(db, l10("Fe", "Ni"), -6.9, source="ggen", run_id=run_a)
    ids["l10_high"] = add(db, l10("Fe", "Ni"), -6.89, source="matra-genoa", run_id=run_b)
    ids["l10_big"] = add(db, l10("Fe", "Ni") * (1, 1, 2), -6.88, source="ggen", run_id=run_b)
    # A different crystal stays.
    ids["other"] = add(db, double_layers("Fe", "Ni"), -6.5, source="ggen", run_id=run_a)
    fake.runs = (run_a, run_b)
    fake.structure_ids = ids
    return fake


def test_plan_merges_copies_into_published_then_lowest(fake):
    store = fake.store()
    assert generated_formulas(store) == ["FeNi"]
    ids = fake.structure_ids
    with store.reader(formulas=generated_formulas(store)) as db:
        merges = {m.duplicate: (m.keeper, m.into) for m in plan(db)}
    assert merges == {
        ids["b2_low"]: (ids["mp"], "published"),
        ids["b2_high"]: (ids["mp"], "published"),
        ids["l10_high"]: (ids["l10_low"], "generated"),
        ids["l10_big"]: (ids["l10_low"], "generated"),
    }


def test_apply_moves_links_and_missing_energies_then_deletes(fake):
    store = fake.store()
    ids, (run_a, run_b) = fake.structure_ids, fake.runs
    with store.reader(formulas=["FeNi"]) as db:
        merges = plan(db)
        assert summarize(merges)["into"] == {"published": 2, "generated": 2}
        counts = apply(store, db, merges)
    assert counts["structures_deleted"] == 4
    server = fake.server
    left = {row[0] for row in server.execute("SELECT id FROM structures")}
    assert left == {ids["mp"], ids["l10_low"], ids["other"]}
    # No orphans anywhere.
    for table in ("geometries", "calculations", "run_structures"):
        assert server.execute(f"SELECT count(*) FROM {table} WHERE structure_id NOT IN (SELECT id FROM structures)").fetchone()[0] == 0
    # Both runs now link the published B2 and the kept L1_0 they found.
    links = {(r[0], r[1]) for r in server.execute("SELECT run_id, structure_id FROM run_structures")}
    assert {(run_a, ids["mp"]), (run_b, ids["mp"]), (run_a, ids["l10_low"]), (run_b, ids["l10_low"])} <= links
    # The published B2 had no energy here: it takes the lowest copy's.
    assert fake.db.get_structure(ids["mp"]).energy_per_atom == pytest.approx(-7.1)
    assert fake.db.get_structure(ids["l10_low"]).energy_per_atom == pytest.approx(-6.9)
    # Planning again finds nothing left to merge.
    with store.reader(formulas=["FeNi"]) as db:
        assert plan(db) == []


def test_plan_parallel_matches_plan(fake):
    from crystaldb.dedupe import plan_parallel

    store = fake.store()
    with store.reader(formulas=["FeNi"]) as db:
        serial = sorted((m.duplicate, m.keeper) for m in plan(db))
        parallel = sorted((m.duplicate, m.keeper) for m in plan_parallel(db, workers=2, log=lambda message: None))
    assert parallel == serial
