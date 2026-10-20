"""Non-commercial data (GNoME, MP's "BY-NC" records): stored, read only by
stores opened for free routes, and never shaping the stored hull distances
of other structures."""

from __future__ import annotations

import csv
import zipfile

import pytest
from pymatgen.core import Lattice, Structure

from crystaldb.intake import ADDED, EXISTS, add_if_new
from crystaldb.sources import GNOME_PSEUDOPOTENTIAL_SHIFT, gnome, non_commercial_sql
from crystaldb.testing import FakeOuro


def bcc(element, a=2.87):
    return Structure(Lattice.cubic(a), [element, element], [[0, 0, 0], [0.5, 0.5, 0.5]])


def b2(a_el, b_el, a=2.9):
    return Structure(Lattice.cubic(a), [a_el, b_el], [[0, 0, 0], [0.5, 0.5, 0.5]])


def l12(a_el, b_el, a=3.6):
    """A3B in the Cu3Au structure."""
    return Structure(
        Lattice.cubic(a), [b_el, a_el, a_el, a_el],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )


def add(db, structure, energy, **kwargs):
    comp = structure.composition
    return db.add_structure(
        formula=comp.formula.replace(" ", ""),
        stoichiometry={el.symbol: int(n) for el, n in comp.items()},
        energy_per_atom=energy,
        total_energy=energy * len(structure),
        num_atoms=len(structure),
        structure=structure,
        **kwargs,
    )


@pytest.fixture
def fake(tmp_path):
    fake = FakeOuro(tmp_path)
    add(fake.db, bcc("Fe"), -8.3, source="mp", source_id="mp-13", generation_metadata={"license": "BY-C"})
    add(fake.db, bcc("Co", 2.82), -7.1, source="mp", source_id="mp-54")
    add(fake.db, b2("Fe", "Co"), -7.9, source="mp", source_id="mp-2090")
    # Below the commercial hull, so they would move it.
    add(fake.db, l12("Co", "Fe"), -7.8, source="gnome", source_id="0000aaaa11",
        generation_metadata={"license": "CC-BY-NC-4.0"})
    add(fake.db, l12("Fe", "Co"), -8.3, source="mp", source_id="mp-3200001",
        generation_metadata={"license": "BY-NC"})
    return fake


def sources(db):
    return sorted(row[0] for row in db.conn.execute("SELECT source_id FROM structures"))


COMMERCIAL = ["mp-13", "mp-2090", "mp-54"]
EVERYTHING = sorted(COMMERCIAL + ["0000aaaa11", "mp-3200001"])


def test_the_condition_matches_by_source_and_by_record_license(fake):
    rows = fake.server.execute(f"SELECT source_id FROM structures s WHERE {non_commercial_sql('s')}").fetchall()
    assert sorted(row[0] for row in rows) == ["0000aaaa11", "mp-3200001"]
    rows = fake.server.execute(f"SELECT count(*) FROM structures s WHERE NOT {non_commercial_sql('s')}").fetchone()
    assert rows[0] == 3  # never NULL, so NOT keeps rows without a license


@pytest.mark.parametrize("bulk_children", [0, 10**6])
def test_copies_leave_non_commercial_data_out_unless_asked(fake, monkeypatch, bulk_children):
    import crystaldb.ouro as module

    monkeypatch.setattr(module, "BULK_CHILDREN", bulk_children)
    with fake.store().working_copy(["Co-Fe"]) as db:
        assert sources(db) == COMMERCIAL
    with fake.store().reader(formulas=["Fe3Co", "FeCo3"]) as db:
        assert sources(db) == []
    with fake.store().bulk_copy() as db:
        assert sources(db) == COMMERCIAL
        # Child rows of the structures left out are dropped too.
        assert db.conn.execute("SELECT count(*) FROM calculations").fetchone()[0] == 3
    with fake.store(non_commercial=True).working_copy(["Co-Fe"]) as db:
        assert sources(db) == EVERYTHING
    assert set(fake.store().source_counts()) == {"mp"}
    assert fake.store().source_counts()["mp"]["structures"] == 3
    assert set(fake.store(non_commercial=True).source_counts()) == {"mp", "gnome"}


def test_hulls_built_with_non_commercial_data_stay_in_the_copy(fake):
    store = fake.store(non_commercial=True)
    with store.working_copy(["Co-Fe"]) as db:
        _, distances, _ = db.compute_hull("Co-Fe")
        feco = db.get_structure_by_source("mp", "mp-2090").id
        # The copy's own hull counts GNoME (it is above it here).
        assert distances[feco] > 0
        new = add(db, b2("Fe", "Co", 3.1), -7.0, source="ouro", source_id="user_file")
        db.compute_hull("Co-Fe")
    hulls = {
        row["source_id"]: row["hull_computed_at"]
        for row in fake.server.execute(
            "SELECT s.source_id, c.hull_computed_at FROM calculations c JOIN structures s ON s.id = c.structure_id"
        )
    }
    # Only the non-commercial rows took the hull distances.
    assert {k for k, v in hulls.items() if v is not None} == {"0000aaaa11", "mp-3200001"}
    assert fake.db.get_structure(new) is not None and hulls["user_file"] is None
    # A copy without them still stores what it computes.
    with fake.store().working_copy(["Co-Fe"]) as db:
        db.compute_hull("Co-Fe")
    stored = fake.server.execute(
        "SELECT s.source_id, c.e_above_hull FROM calculations c JOIN structures s ON s.id = c.structure_id "
        "WHERE s.source_id = 'mp-2090'"
    ).fetchone()
    assert stored["e_above_hull"] == pytest.approx(0.0)


def test_a_copy_of_non_commercial_data_is_still_stored(fake):
    with fake.store(non_commercial=True).working_copy(["Co-Fe"]) as db:
        result = add_if_new(db, l12("Co", "Fe"), -7.8, source="ouro", source_id="user_cofe3")
        assert result.status == ADDED
        # Novelty still reports the published copy.
        assert [m.source_id for m in result.novelty.matches] == ["0000aaaa11"]
        assert result.novelty.matches[0].non_commercial
        known = add_if_new(db, b2("Fe", "Co"), -7.9, source="ouro", source_id="user_feco")
        assert known.status == EXISTS and known.match.source_id == "mp-2090"
        assert not known.match.non_commercial
    with fake.store().reader(formulas=["FeCo3"]) as db:
        assert sources(db) == ["user_cofe3"]


def test_gnome_reader_shifts_energies_onto_mp_pseudopotentials(tmp_path):
    mg = Structure(Lattice.cubic(3.0), ["Mg", "Fe"], [[0, 0, 0], [0.5] * 3])
    fe = bcc("Fe")
    header = [
        "", "Composition", "MaterialId", "Elements", "NSites", "Space Group", "Space Group Number",
        "Uncorrected Energy", "Corrected Energy", "Decomposition Energy Per Atom All",
        "Decomposition Energy Per Atom MP", "Bandgap", "Dimensionality Cheon",
    ]
    with open(tmp_path / "stable_materials_summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow([0, "Fe1Mg1", "aaaa000001", "['Mg', 'Fe']", 2, "Pm-3m", 221, -10.0, -10.0, 0.0, -0.01, 0.0, "3D"])
        writer.writerow([1, "Fe2", "aaaa000002", "['Fe']", 2, "Im-3m", 229, -16.6, -16.6, 0.0, 0.0, "inf", "nan"])
        writer.writerow([2, "Fe2", "aaaa000003", "['Fe']", 2, "Im-3m", 229, -16.0, -16.0, 0.0, 0.0, "", "3D"])
    with zipfile.ZipFile(tmp_path / "by_id.zip", "w") as archive:
        archive.writestr("by_id/aaaa000001.CIF", mg.to(fmt="cif"))
        archive.writestr("by_id/aaaa000002.CIF", fe.to(fmt="cif"))
        # aaaa000003 has no CIF and is skipped.
    records = {r.source_id: r for r in gnome(tmp_path)}
    assert set(records) == {"aaaa000001", "aaaa000002"}
    assert records["aaaa000001"].energy_per_atom == pytest.approx((-10.0 - GNOME_PSEUDOPOTENTIAL_SHIFT["Mg"]) / 2)
    assert records["aaaa000002"].energy_per_atom == pytest.approx(-8.3)
    assert records["aaaa000001"].metadata["gnome_e_above_hull"] == 0.0
    assert records["aaaa000001"].space_group_number == 221
    # Non-finite values are left out: JSON cannot carry them.
    assert records["aaaa000002"].metadata["gnome_band_gap"] is None
    assert records["aaaa000002"].metadata["gnome_dimensionality"] is None
    assert [r.source_id for r in gnome(tmp_path, elements=["Mg"])] == ["aaaa000001"]
