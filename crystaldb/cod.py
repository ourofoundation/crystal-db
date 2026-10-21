"""The Crystallography Open Database (COD) as one Ouro dataset.

COD (https://www.crystallography.net/cod/, CC0) holds experimentally
determined crystal structures: inorganic, mineral, metal-organic and
organic. It publishes no energies, so it is not a crystal DB source: it is a
dataset of its own, one row per COD entry, public like COD itself. Rows carry
COD's metadata (formula, cell, space group, temperature, R factor, the paper)
for every released entry, and the CIF as COD publishes it for the inorganic
ones (no carbon-hydrogen compounds; the largest without their embedded
diffraction data). Novelty checks match against those CIFs as published
prior art next to the crystal DB's sources (``CodDataset.candidates``), and
the matches say they were observed experimentally.

Building the rows needs two things from COD, both public:

- its metadata, from COD's read-only MySQL server (``fetch_metadata``);
- the CIFs of the inorganic entries, mirrored with rsync
  (``rsync_cifs``, about 80k files, 3 GB).

``scripts/cod.py`` does both, parses the CIFs (``parse_cif``) and creates
or refreshes the dataset by upserting on the COD id.
"""

from __future__ import annotations

import io
import logging
import re
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .matching import Candidate
from .sources import get_source

logger = logging.getLogger(__name__)

SOURCE = "cod"
MYSQL_HOST = "sql.crystallography.net"
MYSQL_USER = "cod_reader"
MYSQL_DATABASE = "cod"
RSYNC_URL = "rsync://www.crystallography.net/cif/"
# CIFs larger than this are left out (a handful of entries carry whole
# reflection lists); a dataset row stays a few kB.
MAX_CIF_BYTES = 500_000

# COD's columns read, renamed for the dataset where the name changes.
METADATA_COLUMNS = {
    "file": "id",
    "formula": "formula",
    "cellformula": "cellformula",
    "sg": "space_group",
    "sgNumber": "space_group_number",
    "a": "a",
    "b": "b",
    "c": "c",
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "vol": "cell_volume",
    "Z": "z",
    "mineral": "mineral",
    "commonname": "common_name",
    "chemname": "chemical_name",
    "celltemp": "celltemp",
    "diffrtemp": "diffrtemp",
    "cellpressure": "cellpressure",
    "diffrpressure": "diffrpressure",
    "method": "method",
    "radType": "radiation",
    "Robs": "robs",
    "Rall": "rall",
    "authors": "authors",
    "title": "title",
    "journal": "journal",
    "year": "year",
    "volume": "journal_volume",
    "firstpage": "first_page",
    "doi": "doi",
    "flags": "flags",
    "status": "status",
    "duplicateof": "duplicate_of",
    "onhold": "onhold",
    "svnrevision": "revision",
    "date": "date",
}

# The dataset's columns, in order. ``id`` is COD's id and the primary key.
COLUMNS = (
    "id",
    "formula",
    "reduced_formula",
    "chemsys",
    "nelements",
    "num_atoms",
    "primitive_atoms",
    "space_group",
    "space_group_number",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "cell_volume",
    "z",
    "mineral",
    "common_name",
    "chemical_name",
    "temperature_k",
    "pressure_kpa",
    "method",
    "radiation",
    "r_factor",
    "experimental",
    "organic",
    "disordered",
    "has_coordinates",
    "duplicate_of",
    "status",
    "authors",
    "title",
    "journal",
    "year",
    "journal_volume",
    "first_page",
    "doi",
    "revision",
    "deposited",
    "url",
    "cif",
)
INTEGER_COLUMNS = ("id", "nelements", "num_atoms", "primitive_atoms", "space_group_number", "duplicate_of", "year", "revision")
FLOAT_COLUMNS = ("a", "b", "c", "alpha", "beta", "gamma", "cell_volume", "z", "temperature_k", "pressure_kpa", "r_factor")
BOOLEAN_COLUMNS = ("experimental", "organic", "disordered", "has_coordinates")
# Indexed in the dataset: novelty reads by reduced formula, people by system.
INDEXES = (["reduced_formula"], ["chemsys"])

_ELEMENT = re.compile(r"([A-Z][a-z]?)([0-9.]*)")


# -------------------- Reading COD --------------------


def fetch_metadata(connection=None) -> List[Dict[str, Any]]:
    """Every entry's metadata row from COD's SQL server, under the dataset's
    column names (``METADATA_COLUMNS``)."""
    if connection is None:
        import pymysql

        connection = pymysql.connect(
            host=MYSQL_HOST, user=MYSQL_USER, database=MYSQL_DATABASE, connect_timeout=30
        )
    select = ", ".join(f"`{column}`" for column in METADATA_COLUMNS)
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT {select} FROM data ORDER BY file")
        rows = [dict(zip(METADATA_COLUMNS.values(), row)) for row in cursor.fetchall()]
    logger.info(f"COD: {len(rows):,} entries")
    return rows


def released(meta: Dict[str, Any]) -> bool:
    """COD shows the entry: not on hold before publication."""
    return meta.get("onhold") is None


def formula_counts(formula: Optional[str]) -> Dict[str, float]:
    """``"- C6 H12 O6 -"`` (COD's Hill-ordered formula) as element counts."""
    counts: Dict[str, float] = {}
    for element, number in _ELEMENT.findall((formula or "").strip(" -")):
        try:
            counts[element] = counts.get(element, 0.0) + (float(number) if number else 1.0)
        except ValueError:
            continue
    return counts


def is_organic(formula: Optional[str]) -> bool:
    """Carbon with hydrogen: organic and metal-organic compounds."""
    counts = formula_counts(formula)
    return "C" in counts and "H" in counts


def flags(meta: Dict[str, Any]) -> set:
    return {flag.strip() for flag in (meta.get("flags") or "").split(",") if flag.strip()}


def wants_cif(meta: Dict[str, Any]) -> bool:
    """Whether the dataset keeps this entry's CIF: released, inorganic, with
    atom coordinates, and not retracted."""
    return (
        released(meta)
        and not is_organic(meta.get("formula"))
        and "has coordinates" in flags(meta)
        and meta.get("status") != "retracted"
    )


def cif_path(cod_id: int) -> str:
    """Where COD keeps an entry's CIF: ``1/00/00/1000041.cif``."""
    text = str(cod_id)
    return f"{text[0]}/{text[1:3]}/{text[3:5]}/{text}.cif"


def rsync_cifs(cod_ids: Iterable[int], directory: Path) -> None:
    """Mirror these entries' CIFs from COD into ``directory`` (only what
    changed since the last run is sent)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    listing = directory / "files.txt"
    listing.write_text("".join(f"{cif_path(cod_id)}\n" for cod_id in cod_ids))
    started = time.time()
    subprocess.run(
        ["rsync", "-a", "--files-from", str(listing), RSYNC_URL, str(directory)],
        check=True,
    )
    logger.info(f"rsync of {sum(1 for _ in listing.open()):,} CIFs took {time.time() - started:.0f}s")


# -------------------- Dataset rows --------------------


@dataclass
class ParsedCif:
    reduced_formula: str
    num_atoms: int
    ordered: bool
    # Sites in its primitive cell as novelty matching reduces it: only
    # structures with as many can match it, so candidates with another
    # count are skipped without parsing their CIFs.
    primitive_atoms: Optional[int] = None


def parse_cif(text: str) -> Optional[ParsedCif]:
    """The structure a COD CIF describes, as matching will read it, or None
    if pymatgen cannot read it."""
    from pymatgen.io.cif import CifParser

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structures = CifParser(io.StringIO(text)).parse_structures(primitive=False)
    except Exception:
        return None
    if not structures:
        return None
    from .matching import primitive

    structure = structures[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        primitive_atoms = len(primitive(structure))
    return ParsedCif(
        reduced_formula=structure.composition.reduced_formula,
        num_atoms=len(structure),
        ordered=structure.is_ordered,
        primitive_atoms=primitive_atoms,
    )


def _reduced_formula(counts: Dict[str, float]) -> Optional[str]:
    from pymatgen.core import Composition

    try:
        return Composition(counts).reduced_formula if counts else None
    except Exception:
        return None


def _first(*values):
    return next((value for value in values if value is not None), None)


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text != "?" else None


def dataset_row(
    meta: Dict[str, Any], cif: Optional[str] = None, parsed: Optional[ParsedCif] = None
) -> Dict[str, Any]:
    """One dataset row: COD's metadata, and the CIF when it is kept.

    The reduced formula and atom count come from the parsed CIF when there
    is one (what novelty matching compares), else from COD's formulas.
    """
    counts = formula_counts(meta.get("formula"))
    elements = sorted(counts)
    cell_counts = formula_counts(meta.get("cellformula"))
    entry_flags = flags(meta)
    cod_id = int(meta["id"])
    method = _text(meta.get("method"))
    date = meta.get("date")
    return {
        "id": cod_id,
        "formula": _text((meta.get("formula") or "").strip(" -")),
        "reduced_formula": parsed.reduced_formula if parsed else _reduced_formula(counts),
        "chemsys": "-".join(elements) or None,
        "nelements": len(elements) or None,
        "num_atoms": parsed.num_atoms if parsed else (round(sum(cell_counts.values())) or None),
        "primitive_atoms": parsed.primitive_atoms if parsed else None,
        "space_group": _text(meta.get("space_group")),
        "space_group_number": meta.get("space_group_number"),
        "a": meta.get("a"),
        "b": meta.get("b"),
        "c": meta.get("c"),
        "alpha": meta.get("alpha"),
        "beta": meta.get("beta"),
        "gamma": meta.get("gamma"),
        "cell_volume": meta.get("cell_volume"),
        "z": meta.get("z"),
        "mineral": _text(meta.get("mineral")),
        "common_name": _text(meta.get("common_name")),
        "chemical_name": _text(meta.get("chemical_name")),
        "temperature_k": _first(meta.get("celltemp"), meta.get("diffrtemp")),
        "pressure_kpa": _first(meta.get("cellpressure"), meta.get("diffrpressure")),
        "method": method,
        "radiation": _text(meta.get("radiation")),
        "r_factor": _first(meta.get("robs"), meta.get("rall")),
        # COD also holds some structures from calculations.
        "experimental": method != "theoretical",
        "organic": is_organic(meta.get("formula")),
        "disordered": (not parsed.ordered) if parsed else "has disorder" in entry_flags,
        "has_coordinates": "has coordinates" in entry_flags,
        "duplicate_of": meta.get("duplicate_of"),
        "status": _text(meta.get("status")),
        "authors": _text(meta.get("authors")),
        "title": _text(meta.get("title")),
        "journal": _text(meta.get("journal")),
        "year": meta.get("year"),
        "journal_volume": _text(meta.get("journal_volume")),
        "first_page": _text(meta.get("first_page")),
        "doi": _text(meta.get("doi")),
        "revision": meta.get("revision"),
        "deposited": date.isoformat() if hasattr(date, "isoformat") else _text(date),
        "url": get_source(SOURCE).url.format(id=cod_id),
        "cif": cif if parsed else None,
    }


# What makes a COD CIF large is diffraction data riding along with the
# structure: SHELX's .res/.hkl/.fab files as text fields, and reflection or
# powder-pattern loops. ``slim_cif`` drops them from CIFs over the size cap.
BULK_ITEMS = (
    "_shelx_res_file",
    "_shelx_hkl_file",
    "_shelx_fab_file",
    "_iucr_refine_instructions_details",
    "_iucr_refine_fcf_details",
)
BULK_LOOPS = ("_refln_", "_diffrn_refln_", "_pd_meas_", "_pd_proc_", "_pd_calc_", "_pd_data_")


def slim_cif(text: str) -> Optional[str]:
    """A CIF without its diffraction data (``BULK_ITEMS``, ``BULK_LOOPS``),
    rewritten by pymatgen; None if it does not parse."""
    from pymatgen.io.cif import CifFile

    try:
        cif = CifFile.from_str(text)
    except Exception:
        return None
    for block in cif.data.values():
        for loop in [loop for loop in block.loops if loop and loop[0].startswith(BULK_LOOPS)]:
            block.loops.remove(loop)
            for key in loop:
                block.data.pop(key, None)
        for key in BULK_ITEMS:
            block.data.pop(key, None)
    return str(cif)


def read_cif(directory: Path, cod_id: int) -> Optional[str]:
    """A mirrored CIF's text (without its diffraction data when it is over
    ``MAX_CIF_BYTES``), or None if it is missing or still too large."""
    path = Path(directory) / cif_path(cod_id)
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    if len(text.encode()) > MAX_CIF_BYTES:
        text = slim_cif(text)
    return text if text and len(text.encode()) <= MAX_CIF_BYTES else None


# -------------------- Reading the dataset --------------------


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class CodDataset:
    """The COD dataset on Ouro, read through its SQL endpoint.

    ``ouro`` is an ``ouro.Ouro`` client that can read the dataset (anyone
    can: it is public).
    """

    def __init__(self, ouro: Any, dataset_id: str):
        self.ouro = ouro
        self.dataset_id = dataset_id
        self._table: Optional[str] = None
        self._count: Optional[tuple] = None

    def _request(self, method: str, path: str, **kwargs) -> Any:
        from .ouro import request

        return request(self.ouro, method, path, **kwargs)

    @property
    def table(self) -> str:
        if self._table is None:
            name = self._request("get", f"/datasets/{self.dataset_id}")["metadata"]["table_name"]
            self._table = f'datasets."{name}"'
        return self._table

    def query(self, sql: str) -> List[Dict[str, Any]]:
        return self._request("post", f"/datasets/{self.dataset_id}/query-custom", json={"query": sql}) or []

    def candidates(self, reduced_formula: str) -> List[Candidate]:
        """Entries with a CIF and this reduced formula, as novelty candidates."""
        rows = self.query(
            f"SELECT id, reduced_formula, space_group_number, num_atoms, primitive_atoms, experimental, cif "
            f"FROM {self.table} WHERE reduced_formula = {_literal(reduced_formula)} "
            "AND cif IS NOT NULL"
        )
        license = get_source(SOURCE).license
        return [
            Candidate(
                structure_id=f"{SOURCE}-{row['id']}",
                source=SOURCE,
                source_id=str(row["id"]),
                formula=row["reduced_formula"],
                space_group_number=row.get("space_group_number"),
                cif=row["cif"],
                num_atoms=row.get("num_atoms"),
                primitive_atoms=row.get("primitive_atoms"),
                license=license,
                experimental=bool(row.get("experimental")),
            )
            for row in rows
        ]

    def entry(self, cod_id: int) -> Optional[Dict[str, Any]]:
        rows = self.query(f"SELECT * FROM {self.table} WHERE id = {int(cod_id)}")
        return rows[0] if rows else None

    def count(self, max_age: float = 3600.0) -> int:
        """Entries with a CIF (what novelty checks against), counted at most
        every ``max_age`` seconds."""
        now = time.monotonic()
        if self._count is None or now - self._count[0] > max_age:
            rows = self.query(f"SELECT COUNT(*) AS n FROM {self.table} WHERE cif IS NOT NULL")
            self._count = (now, int(rows[0]["n"]) if rows else 0)
        return self._count[1]
