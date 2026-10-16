"""Where structures come from: the source registry and the reference readers.

``SOURCES`` lists every provenance a structure in the database can have, the
way ``crystaldb.surfaces`` lists energy surfaces. Three kinds:

- ``published``: reference data (Materials Project, Alexandria). Prior art
  for novelty checks, with a license, a label and a link.
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
"""

from __future__ import annotations

import bz2
import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

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
            hull_key="mp_energy_above_hull",
        ),
        Source(
            "alexandria",
            PUBLISHED,
            "Alexandria",
            license="CC-BY-4.0",
            hull_key="alexandria_e_above_hull",
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

# Licenses we do not import, per source or per record: Ouro's routes are
# paid, and this data is licensed for non-commercial use only. Materials
# Project tags its GNoME-derived records "BY-NC".
NON_COMMERCIAL_LICENSES = {"BY-NC", "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0"}


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
    # (Materials Project tags GNoME-derived data "BY-NC").
    license: Optional[str] = None
    space_group_number: Optional[int] = None
    space_group_symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# -------------------- Materials Project --------------------


# Slices of the summary collection, each small enough to come back as a
# plain paged list (~30k documents): an unfiltered query takes mp-api's
# DeltaTable path, which is far too slow to iterate.
_SITE_BINS = [(1, 10), (11, 20), (21, 40), (41, 80), (81, 100000)]


def _mp_summary(mpr, elements: Optional[Sequence[str]]):
    """All materials' chemical systems, and the ids MP licenses non-commercially."""
    fields = ["material_id", "chemsys", "builder_meta"]
    systems: set = set()
    non_commercial = []
    seen = 0
    for n in range(1, 10):
        for sites in _SITE_BINS:
            kwargs: Dict[str, Any] = {"num_elements": (n, n), "num_sites": sites, "fields": fields}
            if elements:
                kwargs["elements"] = list(elements)
            for doc in mpr.materials.summary.search(**kwargs):
                seen += 1
                systems.add(doc.chemsys)
                license = getattr(getattr(doc, "builder_meta", None), "license", None)
                license = str(getattr(license, "value", license)) if license else None
                if license in NON_COMMERCIAL_LICENSES:
                    non_commercial.append((str(doc.material_id), license))
        logger.info(f"  MP summary: {seen:,} materials through {n} elements")
    logger.info(
        f"Materials Project: {seen:,} materials in {len(systems):,} systems, "
        f"{len(non_commercial):,} licensed non-commercially"
    )
    return sorted(systems), non_commercial


def materials_project(
    api_key: str,
    elements: Optional[Sequence[str]] = None,
    batch_size: int = 250,
) -> Iterator[SourceRecord]:
    """GGA/GGA+U entries from Materials Project's ``GGA_GGA+U`` thermo documents.

    Materials only calculated with r2SCAN have no such entry and are skipped.
    The energy and correction are MP's own (uncorrected energy plus the
    MP2020 adjustment MP applied), on the GGA structure itself.
    """
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        all_systems, non_commercial = _mp_summary(mpr, elements)
        # Records without a structure, so the importer retires earlier copies.
        for material_id, license in non_commercial:
            yield SourceRecord(
                source="mp",
                source_id=material_id,
                structure=None,
                energy_per_atom=None,
                license=license,
            )
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
            excluded = {material_id for material_id, _ in non_commercial}
            for doc in docs:
                if str(doc.material_id) in excluded:
                    continue
                # The entry this thermo document's energy comes from.
                entries = [e for e in (doc.entries or {}).values() if e is not None]
                if not entries or doc.energy_per_atom is None:
                    continue
                entry = min(entries, key=lambda e: abs(e.energy_per_atom - doc.energy_per_atom))
                if entry.structure is None:
                    continue
                symmetry = doc.symmetry
                license = getattr(getattr(doc, "builder_meta", None), "license", None)
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


def chunk_records(records: Iterator[SourceRecord], size: int) -> Iterator[List[SourceRecord]]:
    batch: List[SourceRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
