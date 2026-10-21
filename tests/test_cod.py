"""The COD dataset: which entries keep a CIF, its rows, and novelty
matches against it."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from crystaldb.cod import CodDataset, cif_path, dataset_row, is_organic, parse_cif, wants_cif
from crystaldb.matching import KNOWN, NEW_POLYMORPH
from crystaldb.testing import FakeOuro


def rocksalt(a=5.64):
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def meta(cod_id=1000041, formula="- Cl Na -", **kwargs):
    return {
        "id": cod_id,
        "formula": formula,
        "cellformula": "- Cl4 Na4 -",
        "flags": "has coordinates",
        "onhold": None,
        "status": None,
        "method": None,
        **kwargs,
    }


def test_which_entries_keep_a_cif():
    assert wants_cif(meta())
    # Organic and metal-organic (carbon with hydrogen) entries keep only metadata.
    assert is_organic("- C6 H12 O6 -") and not is_organic("- C Ca O3 -")
    assert not wants_cif(meta(formula="- C2 Cu H4 O4 -"))
    assert not wants_cif(meta(flags="has Fobs"))
    assert not wants_cif(meta(status="retracted"))
    assert not wants_cif(meta(onhold="2027-01-01"))
    assert cif_path(1000041) == "1/00/00/1000041.cif"


def test_rows_describe_the_cif_as_read():
    cif = rocksalt().to(fmt="cif", symprec=0.1)
    parsed = parse_cif(cif)
    assert (parsed.reduced_formula, parsed.num_atoms, parsed.ordered) == ("NaCl", 8, True)
    assert parsed.primitive_atoms == 2
    row = dataset_row(meta(celltemp=293.0, Robs=None, method="theoretical"), cif, parsed)
    assert row["reduced_formula"] == "NaCl" and row["chemsys"] == "Cl-Na"
    assert row["num_atoms"] == 8 and row["temperature_k"] == 293.0
    assert row["experimental"] is False
    assert row["url"] == "https://www.crystallography.net/cod/1000041.html"
    assert row["cif"] == cif
    # Without a readable CIF: formula and cell size from COD's formulas, no CIF.
    unread = dataset_row(meta(), cif, None)
    assert unread["cif"] is None and unread["num_atoms"] == 8 and unread["reduced_formula"] == "NaCl"
    assert parse_cif("not a cif") is None


SCHEMA = """
CREATE TABLE cod (
    id INTEGER PRIMARY KEY, reduced_formula TEXT, space_group_number INTEGER,
    num_atoms INTEGER, primitive_atoms INTEGER, experimental INTEGER, cif TEXT
);
"""


@pytest.fixture
def cod(tmp_path):
    fake = FakeOuro(tmp_path / "server.db")
    dataset_id = fake.add_dataset("cod", SCHEMA)
    rows = [
        (1000041, "NaCl", 225, 8, 2, 1, rocksalt().to(fmt="cif")),
        # A theoretical entry, one whose CIF was not kept, and one whose
        # primitive cell size rules it out without parsing it.
        (1000042, "NaCl", 225, 8, 2, 0, rocksalt(5.7).to(fmt="cif")),
        (1000043, "NaCl", 221, 2, 2, 1, None),
        (1000044, "NaCl", 225, 8, 4, 1, rocksalt(5.6).to(fmt="cif")),
    ]
    fake.server.executemany("INSERT INTO cod VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    fake.server.commit()
    return fake, CodDataset(fake, dataset_id)


def test_novelty_matches_cod_entries(cod):
    fake, dataset = cod
    assert dataset.count() == 3
    candidates = dataset.candidates("NaCl")
    assert sorted(c.source_id for c in candidates) == ["1000041", "1000042", "1000044"]

    index = fake.db.novelty_index(extra=[dataset.candidates])
    result = index.check(rocksalt(5.5))
    assert result.status == KNOWN
    assert {m.source_id for m in result.matches} == {"1000041", "1000042"}
    assert result.experimental
    assert result.same_formula_counts == {"cod": 3}
    parsed = {c.source_id for c in index.candidates("NaCl") if c._primitive is not None}
    assert parsed == {"1000041", "1000042"}
    assert dataset.entry(1000043)["space_group_number"] == 221

    # A different crystal of the same formula: CsCl-type.
    cscl = Structure(Lattice.cubic(3.4), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert fake.db.novelty_index(extra=[dataset.candidates]).check(cscl).status == NEW_POLYMORPH


def test_large_cifs_lose_their_diffraction_data(tmp_path, monkeypatch):
    import crystaldb.cod as cod

    cif = rocksalt().to(fmt="cif", symprec=0.1)
    bulky = cif + "_shelx_hkl_file\n;\n" + "   1   0   0  12.0  0.5\n" * 2000 + ";\n"
    bulky += "loop_\n_refln_index_h\n_refln_index_k\n_refln_index_l\n" + "1 0 0\n" * 2000
    path = tmp_path / cif_path(1000041)
    path.parent.mkdir(parents=True)
    path.write_text(bulky)
    monkeypatch.setattr(cod, "MAX_CIF_BYTES", 20_000)
    slim = cod.read_cif(tmp_path, 1000041)
    assert "_shelx_hkl_file" not in slim and "_refln_index_h" not in slim
    assert parse_cif(slim) == parse_cif(cif)
