#!/usr/bin/env python3
"""
Build and publish the Crystallography Open Database dataset (crystaldb.cod).

``build`` reads every entry's metadata from COD's SQL server, mirrors the
inorganic entries' CIFs with rsync, parses them, and writes one row per
entry to a gzipped JSON-lines file. ``publish`` creates the Ouro dataset
from that file, or refreshes an existing one by upserting on the COD id.
Rebuilding later fetches only the CIFs that changed.

Usage:
    pip install pymysql   # and rsync on the PATH
    python scripts/cod.py build --cif-dir cod-cif --out cod_rows.jsonl.gz
    # a quick look at a slice
    python scripts/cod.py build --cif-dir cod-cif --out sample.jsonl.gz --limit 2000

    # OURO_API_KEY (and OURO_BACKEND_URL for another backend) in the environment
    python scripts/cod.py publish --rows cod_rows.jsonl.gz --create \\
        --org-id <org> --team-id <team>
    python scripts/cod.py publish --rows cod_rows.jsonl.gz --dataset-id <id>
"""

import argparse
import gzip
import json
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from crystaldb.cod import (
    BOOLEAN_COLUMNS,
    COLUMNS,
    FLOAT_COLUMNS,
    INDEXES,
    INTEGER_COLUMNS,
    dataset_row,
    fetch_metadata,
    parse_cif,
    read_cif,
    released,
    rsync_cifs,
    wants_cif,
)
from crystaldb.sources import get_source

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NAME = "Crystallography Open Database"
DESCRIPTION = """Every released entry of the [Crystallography Open Database](https://www.crystallography.net/cod/) (COD): experimentally determined crystal structures of minerals, inorganic, metal-organic and organic compounds, one row per COD entry.

- **Metadata for every entry:** formula, cell, space group, temperature and pressure, R factor, method, and the paper it was published in (journal, year, DOI).
- **The CIF itself for inorganic entries** (no carbon-hydrogen compounds), as COD publishes it; the few largest lose the reflection data and SHELX files embedded in them. `reduced_formula`, `num_atoms` and `primitive_atoms` (sites in its primitive cell) describe that CIF as pymatgen reads it.
- `experimental` is false for COD's few structures from calculations. `duplicate_of` and `status` (retracted, warnings, errors) are COD's own.
- `url` opens the entry on COD.

Ouro's Crystal DB novelty check matches structures against these CIFs alongside Materials Project, Alexandria and GNoME.

COD's data is in the public domain (CC0). Please acknowledge the original authors of each structure and cite COD: Gražulis et al., *J. Appl. Cryst.* 42, 726 (2009), [doi:10.1107/S0021889809016690](https://doi.org/10.1107/S0021889809016690)."""


def _parse(args: Tuple[str, int]) -> Tuple[int, Optional[str], Any]:
    directory, cod_id = args
    cif = read_cif(Path(directory), cod_id)
    return cod_id, cif, parse_cif(cif) if cif else None


def build(args) -> None:
    started = time.time()
    metadata = [meta for meta in fetch_metadata() if released(meta)]
    if args.limit:
        metadata = metadata[: args.limit]
    wanted = [int(meta["id"]) for meta in metadata if wants_cif(meta)]
    logger.info(f"{len(metadata):,} released entries, {len(wanted):,} with a CIF to keep")
    if not args.no_rsync:
        rsync_cifs(wanted, Path(args.cif_dir))

    parsed: Dict[int, Tuple[Optional[str], Any]] = {}
    with Pool(args.workers) as pool:
        work = ((args.cif_dir, cod_id) for cod_id in wanted)
        for done, (cod_id, cif, result) in enumerate(pool.imap_unordered(_parse, work, chunksize=64), 1):
            parsed[cod_id] = (cif, result)
            if done % 10_000 == 0:
                logger.info(f"  parsed {done:,} CIFs")
    unreadable = sum(1 for cif, result in parsed.values() if cif and result is None)
    missing = sum(1 for cif, _ in parsed.values() if cif is None)

    kept = 0
    with gzip.open(args.out, "wt") as handle:
        for meta in metadata:
            cif, result = parsed.get(int(meta["id"]), (None, None))
            row = dataset_row(meta, cif, result)
            kept += row["cif"] is not None
            handle.write(json.dumps(row, default=str) + "\n")
    logger.info(
        f"Wrote {len(metadata):,} rows to {args.out}: {kept:,} with a CIF, "
        f"{unreadable:,} CIFs pymatgen could not read, {missing:,} missing or too large, "
        f"in {time.time() - started:.0f}s"
    )


def read_rows(path: str) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def frame(rows: List[Dict[str, Any]]):
    """The rows with column types the dataset's schema should get."""
    import pandas as pd

    df = pd.DataFrame(rows, columns=list(COLUMNS))
    for column in INTEGER_COLUMNS:
        df[column] = df[column].astype("Int64")
    for column in FLOAT_COLUMNS:
        df[column] = df[column].astype("float64")
    for column in BOOLEAN_COLUMNS:
        df[column] = df[column].astype("boolean")
    return df


def publish(args) -> None:
    from ouro import Ouro

    rows = read_rows(args.rows)
    ouro = Ouro()
    source = get_source("cod")
    started = time.time()
    if args.create:
        dataset = ouro.datasets.create(
            name=NAME,
            visibility=args.visibility,
            data=frame(rows),
            description=DESCRIPTION,
            indexes=list(INDEXES),
            license_id=source.license,
            attribution={
                "originality": "third-party",
                "external_url": source.home_url,
                "doi_url": f"https://doi.org/{source.doi}",
                "citation": {
                    "doi": source.doi,
                    "title": "Crystallography Open Database – an open-access collection of crystal structures",
                    "year": 2009,
                    "venue": "Journal of Applied Crystallography",
                },
            },
            org_id=args.org_id,
            team_id=args.team_id,
        )
        logger.info(f"Created dataset {dataset.id} with {len(rows):,} rows in {time.time() - started:.0f}s")
        return
    # Upserts by id; a refresh sends every row, so each one's columns are current.
    for start in range(0, len(rows), args.batch):
        ouro.datasets.update(args.dataset_id, data=frame(rows[start : start + args.batch]), data_mode="upsert")
        logger.info(f"  {min(start + args.batch, len(rows)):,} rows sent")
    logger.info(f"Refreshed {args.dataset_id} with {len(rows):,} rows in {time.time() - started:.0f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    b = commands.add_parser("build", help="Fetch COD and write the dataset rows")
    b.add_argument("--cif-dir", required=True, help="Mirror of COD's CIFs (rsync target)")
    b.add_argument("--out", required=True, help="Rows file (.jsonl.gz)")
    b.add_argument("--limit", type=int, default=None, help="Only the first N entries")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--no-rsync", action="store_true", help="Use the CIFs already mirrored")

    p = commands.add_parser("publish", help="Create or refresh the Ouro dataset")
    p.add_argument("--rows", required=True)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--create", action="store_true")
    target.add_argument("--dataset-id")
    p.add_argument("--org-id")
    p.add_argument("--team-id")
    p.add_argument("--visibility", default="public")
    p.add_argument("--batch", type=int, default=20_000)

    args = parser.parse_args()
    build(args) if args.command == "build" else publish(args)


if __name__ == "__main__":
    main()
