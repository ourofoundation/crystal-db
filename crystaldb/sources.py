"""Where structures come from: the source registry and the reference readers.

``SOURCES`` lists every provenance a structure in the database can have, the
way ``crystaldb.surfaces`` lists energy surfaces. Three kinds:

- ``published``: reference data (Materials Project, Alexandria, GNoME, and
  the Crystallography Open Database, which lives in its own dataset; see
  ``crystaldb.cod``). Prior art for novelty checks, with a license, a label
  and a link.
- ``user``: structures people submitted to Ouro's routes (``ouro``).
- ``generated``: structures a generator made on Ouro (GGen, Matra-Genoa).

Anything not registered counts as generated, so a new model's route can save
under its own source name without a crystaldb release, and its output is never
mistaken for prior art; register it only to give it a display label. A new
published dataset does need an entry (and a reader below), since only
registered published sources are prior art.

The readers yield ``SourceRecord``s: a structure as published, with the raw
(uncorrected) PBE energy computed on it and, when the source publishes them,
the run settings and its own MP2020 correction. ``scripts/import.py`` writes
records into the database as source geometries and ``pbe`` calculations, so
adding a dataset means a registry entry plus one iterator here.

Some data is licensed for non-commercial use only: all of GNoME, and the
Materials Project records MP tags "BY-NC" (its GNoME-derived ones). It is
stored like any other, and marked by its license (the source's, or the
record's in ``metadata.license``). Ouro's paid routes never read it: a
shared database copy leaves it out unless it is opened for free routes
(``crystaldb.ouro.OuroSharedDatabase(non_commercial=True)``), and a copy
that includes it never stores hull distances it shaped on other structures.
``non_commercial_sql`` is the one test, in SQL for both stores.

Where a source says whether a record was observed experimentally, its
metadata has ``experimental`` (true or false) and ``icsd_ids`` (ICSD
collection codes, ids only). Materials Project is the one that does.
"""

from __future__ import annotations

import bz2
import csv
import gc
import json
import logging
import math
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from pymatgen.core import Structure

logger = logging.getLogger(__name__)

PUBLISHED = "published"
GENERATED = "generated"
USER = "user"


@dataclass(frozen=True)
class Source:
    id: str
    kind: str  # PUBLISHED, GENERATED or USER
    label: str  # "Materials Project"
    short_label: Optional[str] = None  # "MP"
    license: Optional[str] = None
    # Page for one record, formatted with ``id`` (the record's source_id).
    url: Optional[str] = None
    # The source's home page, and the paper to cite for its data.
    home_url: Optional[str] = None
    doi: Optional[str] = None
    # Key in ``structures.metadata`` holding the source's own distance to
    # its hull (eV/atom), stored on import; see scripts/published_hulls.py.
    hull_key: Optional[str] = None

    @property
    def short(self) -> str:
        return self.short_label or self.label


SOURCES: Dict[str, Source] = {
    s.id: s
    for s in (
        Source(
            "mp",
            PUBLISHED,
            "Materials Project",
            short_label="MP",
            license="CC-BY-4.0",
            url="https://next-gen.materialsproject.org/materials/{id}",
            home_url="https://next-gen.materialsproject.org",
            doi="10.1063/1.4812323",
            hull_key="mp_energy_above_hull",
        ),
        Source(
            "alexandria",
            PUBLISHED,
            "Alexandria",
            license="CC-BY-4.0",
            home_url="https://alexandria.icams.rub.de",
            doi="10.1002/adma.202210788",
            hull_key="alexandria_e_above_hull",
        ),
        Source(
            "gnome",
            PUBLISHED,
            "GNoME",
            license="CC-BY-NC-4.0",
            home_url="https://github.com/google-deepmind/materials_discovery",
            doi="10.1038/s41586-023-06735-9",
            hull_key="gnome_e_above_hull",
        ),
        # Experimental structures, kept in a dataset of their own rather
        # than in the crystal DB's tables (they have no energies; see
        # crystaldb.cod). Prior art for novelty checks all the same.
        Source(
            "cod",
            PUBLISHED,
            "Crystallography Open Database",
            short_label="COD",
            license="CC0-1.0",
            url="https://www.crystallography.net/cod/{id}.html",
            home_url="https://www.crystallography.net/cod/",
            doi="10.1107/S0021889809016690",
        ),
        Source("ggen", GENERATED, "GGen"),
        Source("matra-genoa", GENERATED, "Matra-Genoa"),
        Source("ouro", USER, "Ouro submission"),
    )
}

USER_SOURCE = "ouro"
# Sources that are prior art. Only these, so an unregistered source is never
# mistaken for published data.
PUBLISHED_SOURCES = tuple(s.id for s in SOURCES.values() if s.kind == PUBLISHED)

# Licenses that allow non-commercial use only. Ouro's paid routes must not
# use data under them (see the module docstring). Materials Project tags its
# GNoME-derived records "BY-NC" (and the rest "BY-C").
NON_COMMERCIAL_LICENSES = ("BY-NC", "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0")


def is_non_commercial(license: Optional[str]) -> bool:
    return license in NON_COMMERCIAL_LICENSES


def non_commercial_sources() -> List[str]:
    """Sources whose every record is licensed non-commercially."""
    return sorted(s.id for s in SOURCES.values() if is_non_commercial(s.license))


def _quoted(values: Sequence[str]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


def non_commercial_sql(alias: str = "s", dialect: str = "sqlite") -> str:
    """A SQL condition, true for structures (table alias ``alias``) licensed
    non-commercially: from such a source, or with such a license in their
    metadata. Never NULL, so ``NOT`` of it is safe. ``dialect`` is
    ``sqlite`` (a local copy) or ``postgres`` (the shared datasets)."""
    if dialect == "postgres":
        license = f"COALESCE({alias}.metadata->>'license', '')"
    else:
        license = f"COALESCE(json_extract({alias}.metadata, '$.license'), '')"
    condition = f"{license} IN ({_quoted(NON_COMMERCIAL_LICENSES)})"
    sources = non_commercial_sources()
    if sources:
        condition = f"({alias}.source IN ({_quoted(sources)}) OR {condition})"
    return condition


# Materials Project's own license tags, as SPDX ids.
SPDX_LICENSES = {"BY-C": "CC-BY-4.0", "BY-NC": "CC-BY-NC-4.0"}


def record_license(source: Optional[str], license: Optional[str] = None) -> Optional[str]:
    """A stored structure's license as an SPDX id: the record's own
    (``metadata.license``) when the source sets one per record, else the
    source's."""
    license = license or get_source(source).license
    return SPDX_LICENSES.get(license, license)


def get_source(source: Optional[str]) -> Source:
    """The registry entry, or an ad-hoc generated one for an unregistered name."""
    source = source or "ggen"
    return SOURCES.get(source) or Source(source, GENERATED, source)


def is_published(source: Optional[str]) -> bool:
    return (source or "ggen") in PUBLISHED_SOURCES


def is_generated(source: Optional[str]) -> bool:
    """Made by a generator on Ouro: neither published nor a user submission."""
    return get_source(source).kind == GENERATED


def source_url(source: Optional[str], source_id: Optional[str]) -> Optional[str]:
    template = get_source(source).url
    return template.format(id=source_id) if template and source_id else None


@dataclass
class SourceRecord:
    source: str
    source_id: str
    structure: Optional[Structure]
    # Raw PBE (or PBE+U) energy per atom on ``structure``'s cell.
    energy_per_atom: Optional[float]
    # The source's own MP2020 correction, when it applies MP2020 exactly as
    # we would (Materials Project). Otherwise computed on import.
    correction_per_atom: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    # The source's license for this record when it varies per record
    # (Materials Project's "BY-C" / "BY-NC").
    license: Optional[str] = None
    space_group_number: Optional[int] = None
    space_group_symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# -------------------- Materials Project --------------------


# Slices of the summary collection, each small enough to come back as a
# plain paged list (~30k documents): an unfiltered query takes mp-api's
# DeltaTable path, which is far too slow to iterate.
_SITE_BINS = [(1, 10), (11, 20), (21, 40), (41, 80), (81, 100000)]


@dataclass(frozen=True)
class Provenance:
    """Whether MP counts a material as experimentally observed, and the
    ICSD entries it was matched to (MP's ``theoretical`` and
    ``database_IDs``). MP calls a material theoretical even when its ICSD
    ids are all for ICSD's own theoretical entries."""

    experimental: bool
    icsd_ids: Tuple[int, ...] = ()


def _icsd_ids(database_ids: Optional[Dict[str, Any]]) -> Tuple[int, ...]:
    ids = set()
    for value in (database_ids or {}).get("icsd") or []:
        digits = str(value).rsplit("-", 1)[-1]
        if digits.isdigit():
            ids.add(int(digits))
    return tuple(sorted(ids))


def _mp_summary(mpr, elements: Optional[Sequence[str]]) -> Tuple[List[str], Dict[str, Provenance]]:
    """All materials' chemical systems, and each material's provenance."""
    fields = ["material_id", "chemsys", "theoretical", "database_IDs"]
    systems: set = set()
    provenance: Dict[str, Provenance] = {}
    for n in range(1, 10):
        for sites in _SITE_BINS:
            kwargs: Dict[str, Any] = {"num_elements": (n, n), "num_sites": sites, "fields": fields}
            if elements:
                kwargs["elements"] = list(elements)
            for doc in mpr.materials.summary.search(**kwargs):
                systems.add(doc.chemsys)
                provenance[str(doc.material_id)] = Provenance(
                    experimental=doc.theoretical is False,
                    icsd_ids=_icsd_ids(doc.database_IDs),
                )
        logger.info(f"  MP summary: {len(provenance):,} materials through {n} elements")
    experimental = sum(p.experimental for p in provenance.values())
    logger.info(
        f"Materials Project: {len(provenance):,} materials ({experimental:,} experimental) "
        f"in {len(systems):,} systems"
    )
    return sorted(systems), provenance


def materials_project(
    api_key: str,
    elements: Optional[Sequence[str]] = None,
    batch_size: int = 250,
) -> Iterator[SourceRecord]:
    """GGA/GGA+U entries from Materials Project's ``GGA_GGA+U`` thermo documents.

    Materials only calculated with r2SCAN have no such entry and are skipped
    (most of MP's GNoME-derived records; GNoME's own release has their PBE
    energies). The energy and correction are MP's own (uncorrected energy
    plus the MP2020 adjustment MP applied), on the GGA structure itself.
    Each record carries MP's license for it, "BY-C" or "BY-NC", and in its
    metadata whether MP counts it as experimentally observed
    (``experimental``) and the ICSD entries MP matched it to (``icsd_ids``).
    The ids are references only: no ICSD data is read.
    """
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        all_systems, provenance = _mp_summary(mpr, elements)
        for start in range(0, len(all_systems), batch_size):
            systems = all_systems[start : start + batch_size]
            docs = mpr.materials.thermo.search(
                thermo_types=["GGA_GGA+U"],
                chemsys=systems,
                fields=[
                    "material_id",
                    "builder_meta",
                    "entries",
                    "symmetry",
                    "energy_per_atom",
                    "energy_above_hull",
                ],
            )
            for doc in docs:
                # The entry this thermo document's energy comes from.
                entries = [e for e in (doc.entries or {}).values() if e is not None]
                if not entries or doc.energy_per_atom is None:
                    continue
                entry = min(entries, key=lambda e: abs(e.energy_per_atom - doc.energy_per_atom))
                if entry.structure is None:
                    continue
                symmetry = doc.symmetry
                license = getattr(getattr(doc, "builder_meta", None), "license", None)
                origin = provenance.get(str(doc.material_id))
                yield SourceRecord(
                    source="mp",
                    source_id=str(doc.material_id),
                    license=str(getattr(license, "value", license)) if license else None,
                    structure=entry.structure,
                    energy_per_atom=entry.uncorrected_energy_per_atom,
                    correction_per_atom=entry.correction_per_atom,
                    parameters={
                        "run_type": entry.parameters.get("run_type"),
                        "is_hubbard": entry.parameters.get("is_hubbard"),
                        "hubbards": entry.parameters.get("hubbards") or {},
                    },
                    space_group_number=getattr(symmetry, "number", None),
                    space_group_symbol=getattr(symmetry, "symbol", None),
                    metadata={
                        "mp_entry_id": str(entry.entry_id),
                        "mp_energy_above_hull": doc.energy_above_hull,
                        **(
                            {"experimental": origin.experimental, "icsd_ids": list(origin.icsd_ids)}
                            if origin
                            else {}
                        ),
                    },
                )


# -------------------- Alexandria --------------------


def _alexandria_record(entry: Dict[str, Any]) -> Optional[SourceRecord]:
    data = entry.get("data") or {}
    material_id = data.get("mat_id") or entry.get("entry_id")
    structure_dict = entry.get("structure")
    if not material_id or not structure_dict:
        return None
    try:
        # Built from the arrays directly: Structure.from_dict costs ~6 ms.
        sites = structure_dict["sites"]
        structure = Structure(
            structure_dict["lattice"]["matrix"],
            [site["species"][0]["element"] for site in sites],
            [site["abc"] for site in sites],
        )
    except Exception:
        try:
            structure = Structure.from_dict(structure_dict)
        except Exception:
            return None
    energy_total = data.get("energy_total")
    sg_number = data.get("spg")
    return SourceRecord(
        source="alexandria",
        source_id=str(material_id),
        structure=structure,
        # Alexandria's energy_total is the raw energy of this cell; its
        # energy_corrected follows a different scheme from MP's, so the
        # correction is recomputed on import.
        energy_per_atom=None if energy_total is None else energy_total / len(structure),
        space_group_number=int(sg_number) if sg_number else None,
        metadata={
            "alexandria_band_gap": data.get("band_gap_ind", data.get("band_gap_dir")),
            # PBE total magnetic moment of the cell (μB)
            "alexandria_total_mag": data.get("total_mag"),
            # Distance to Alexandria's own hull (its corrections), eV/atom
            "alexandria_e_above_hull": data.get("e_above_hull"),
            "alexandria_last_modified": data.get("last_modified"),
        },
    )


def alexandria_chunks(directory: Path, pattern: str = "alexandria_*.json.bz2") -> List[Path]:
    chunks = sorted(Path(directory).glob(pattern))
    if not chunks:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")
    return chunks


def alexandria_chunk(
    path: Path, elements: Optional[Sequence[str]] = None
) -> Iterator[SourceRecord]:
    """Records in one downloaded Alexandria chunk file."""
    wanted = set(elements or [])
    with bz2.open(path, "rt") as handle:
        payload = json.load(handle)
    for entry in payload.get("entries", []):
        if wanted:
            entry_elements = set((entry.get("data") or {}).get("elements") or [])
            if not wanted <= entry_elements:
                continue
        record = _alexandria_record(entry)
        if record is not None:
            yield record


def alexandria(
    directory: Path,
    pattern: str = "alexandria_*.json.bz2",
    elements: Optional[Sequence[str]] = None,
) -> Iterator[SourceRecord]:
    """Alexandria PBE entries from the downloaded ``.json.bz2`` chunk files."""
    chunks = alexandria_chunks(directory, pattern)
    logger.info(f"Alexandria: {len(chunks)} chunk files in {directory}")
    for chunk in chunks:
        logger.info(f"  {chunk.name}")
        yield from alexandria_chunk(chunk, elements)
        gc.collect()


# -------------------- GNoME --------------------

# GNoME's release (github.com/google-deepmind/materials_discovery): a summary
# table and a zip of CIFs named by material id, downloaded from DeepMind's
# public bucket (gs://gdm_materials_discovery/gnome_data/).
GNOME_SUMMARY = "stable_materials_summary.csv"
GNOME_STRUCTURES = "by_id.zip"

# GNoME's pseudopotentials for these elements are not MP's, so its raw
# energies sit this much higher per atom of the element (eV). DeepMind
# publishes the values for shifting MP onto GNoME's scale; matched
# structures confirm them against Alexandria's PBE (median Ge 0.104, Mg
# 0.091, Ga -0.0025 per atom of the element; other systems agree to 2
# meV/atom). The importer subtracts them, so GNoME shares MP's hull.
GNOME_PSEUDOPOTENTIAL_SHIFT = {
    "Ga": -0.0028805,
    "Ge": 0.10417085,
    "Li": -0.00301278,
    "Mg": 0.0924014,
    "Na": -0.00447437,
}


def _float(value: Optional[str]) -> Optional[float]:
    """A finite number, or None (the table has nan and inf band gaps,
    which JSON cannot carry)."""
    try:
        number = float(value) if value not in (None, "") else None
    except ValueError:
        return None
    return number if number is not None and math.isfinite(number) else None


def _text(value: Optional[str]) -> Optional[str]:
    value = (value or "").strip()
    return value if value and value.lower() != "nan" else None


def gnome_rows(directory: Path) -> List[Dict[str, str]]:
    """The summary table's rows."""
    with open(Path(directory) / GNOME_SUMMARY, newline="") as handle:
        return list(csv.DictReader(handle))


def _gnome_record(row: Dict[str, str], cif: str) -> Optional[SourceRecord]:
    material_id = row.get("MaterialId")
    energy = _float(row.get("Uncorrected Energy"))  # of the cell, eV
    try:
        structure = Structure.from_str(cif, fmt="cif")
    except Exception:
        return None
    if not material_id or energy is None or len(structure) != int(row.get("NSites") or 0):
        return None
    composition = structure.composition
    energy -= sum(
        shift * composition[el] for el, shift in GNOME_PSEUDOPOTENTIAL_SHIFT.items() if el in composition
    )
    number = row.get("Space Group Number")
    return SourceRecord(
        source="gnome",
        source_id=str(material_id),
        structure=structure,
        # MP2020 is recomputed on import from the raw energy, as for Alexandria.
        energy_per_atom=energy / len(structure),
        space_group_number=int(number) if number else None,
        space_group_symbol=row.get("Space Group") or None,
        metadata={
            # Distance to GNoME's hull of all its entries with MP's and
            # OQMD's (eV/atom), and to MP's alone.
            "gnome_e_above_hull": _float(row.get("Decomposition Energy Per Atom All")),
            "gnome_e_above_mp_hull": _float(row.get("Decomposition Energy Per Atom MP")),
            "gnome_band_gap": _float(row.get("Bandgap")),
            "gnome_dimensionality": _text(row.get("Dimensionality Cheon")),
        },
    )


@lru_cache(maxsize=1)
def _gnome_archive(path: str):
    """The CIF zip and its entry names by material id, opened once per
    process: reading the index of 554k entries takes a second or two."""
    archive = zipfile.ZipFile(path)
    names = {Path(name).stem: name for name in archive.namelist() if name.upper().endswith(".CIF")}
    return archive, names


def gnome_chunk(
    directory: Path, rows: Sequence[Dict[str, str]], elements: Optional[Sequence[str]] = None
) -> Iterator[SourceRecord]:
    """Records for some summary rows, their structures read from the zip."""
    wanted = set(elements or [])
    archive, names = _gnome_archive(str(Path(directory) / GNOME_STRUCTURES))
    for row in rows:
        if wanted and not wanted <= set(json.loads(row["Elements"].replace("'", '"'))):
            continue
        name = names.get(row.get("MaterialId", ""))
        if name is None:
            continue
        record = _gnome_record(row, archive.read(name).decode())
        if record is not None:
            yield record


def gnome(directory: Path, elements: Optional[Sequence[str]] = None) -> Iterator[SourceRecord]:
    """GNoME's stable materials from the downloaded summary table and CIFs."""
    yield from gnome_chunk(directory, gnome_rows(directory), elements)


def chunk_records(records: Iterator[SourceRecord], size: int) -> Iterator[List[SourceRecord]]:
    batch: List[SourceRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
