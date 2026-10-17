# crystal-db

A crystal structure database in one SQLite file. It holds:

- published reference structures (Materials Project, Alexandria)
- structures made by generators
- user submissions

Each structure carries raw energies on several **energy surfaces**: PBE from the source databases, plus any number of machine-learned potentials. MP2020 corrections apply on every surface, so GGA and GGA+U phases share one convex hull.

It's the shared structure database behind [Ouro](https://ouro.foundation)'s materials services. [GGen](https://github.com/ourofoundation/ggen) uses it as its local store.

```bash
pip install "crystaldb @ git+https://github.com/ourofoundation/crystal-db.git"
# extras: [mp] Materials Project import, [relax] ORB v3 via torch-sim, [ouro] sharing as Ouro datasets
```

```python
from crystaldb import StructureDatabase, ORB_V3, PBE

db = StructureDatabase("structures.db", surface=PBE.id)
pd, e_above_hull, _ = db.compute_hull("Fe-Co-Ge")
index = db.novelty_index()
index.check(structure).status  # "known", "new polymorph" or "new composition"
```

| Module | Holds |
|---|---|
| `crystaldb.database` | Schema, structures, geometries per slot, calculations per surface, hulls, shards (extract / export / merge) |
| `crystaldb.surfaces` | Energy surfaces (PBE, Orb v3) |
| `crystaldb.corrections` | MP2020, applied the same way on every surface |
| `crystaldb.sources` | Source registry (published / generated / user) and readers for Materials Project and Alexandria |
| `crystaldb.matching` | Novelty: is a structure already published? |
| `crystaldb.references` | Fills near-hull reference energies on an MLIP surface |
| `crystaldb.intake` | Adds a structure only if no stored structure matches it |
| `crystaldb.stability` | Places a submitted structure on its system's hull: stores it if new, fills references, builds the hull |
| `crystaldb.validation` | Reads submitted CIFs and refuses ones that can't be crystals |
| `crystaldb.relaxation` | One batched ORB v3 relax recipe, plus single-points (torch-sim) |
| `crystaldb.ouro` | The tables as Ouro datasets: working copies filled from them, changes upserted back |
| `crystaldb.testing` | A fake of Ouro's dataset endpoints on SQLite, for tests |

Scripts in `scripts/` maintain a database:

| Script | What it does |
|---|---|
| `import.py` | Import reference data |
| `published_hulls.py` | Store the sources' own hull distances |
| `hulls.py` | Compute hull distances on a surface |
| `migrate_v2.py` | Convert a version 1 GGen database |

## Sources

`crystaldb.sources.SOURCES` lists every provenance a structure can have.

- **Published sources** are prior art: the novelty check matches against them, and they carry a license and a link.
- **Any other source** counts as generated: its structures are never prior art, and its P1 structures (mostly symmetry-broken leftovers) can be left out of hulls. A new generator can save structures under its own source name without a release of this package.
- **`ouro`** marks user submissions.

## Sharing one database between apps

Ouro keeps the database as five datasets, one per table (structures, geometries, calculations, runs, run_structures), with the same columns and primary keys. Postgres takes concurrent writes, so any number of apps read and write it at once:

- **Read:** `store.reader(systems=[...])` or `store.reader(formulas=[...])` fills a local `StructureDatabase` with those structures, their geometries, calculations and runs.
- **Write:** `store.working_copy(systems)` does the same; on exit, everything the copy changed is upserted back. A new structure the datasets already hold under another id (the same `(source, source_id)`, or for generated structures the same `structure_hash`) is merged into that row. A stored row's identity fields only change when the copy changed them and is newer, so a copy taken before an import can't undo it.

```python
from ouro import Ouro
from crystaldb.ouro import OuroSharedDatabase

store = OuroSharedDatabase(Ouro(api_key=...), {"structures": "<dataset id>", "geometries": ..., ...})
with store.working_copy(["Fe-Co"]) as db:
    ...
```

## Hulls and adding structures

```python
from crystaldb.references import ase_energies
from crystaldb.stability import assess

result = assess(shared, structure, energy_per_atom, ase_energies(calculator),
                surface=ORB_V3.id, source_id="my-file")
result.hull.e_above_hull, result.intake.status  # "added", "updated" or "exists"
```

`intake.add_if_new` is the storing step on its own. Ouro's Crystal DB service exposes both as routes.

## Adding a generator

Nothing needs registering. In a `working_copy([chemsys])`:

1. Call `references.fill_near_hull(db, chemsys, energies)` so the MLIP hull has every phase near the PBE hull. For example, `energies = lambda s: relaxation.singlepoint(model, s)`.
2. Relax candidates with `relaxation.relax`.
3. Save each candidate with `intake.add_if_new(db, structure, energy, source="<model>", run_id=...)`. A candidate that matches a stored structure is linked to the run instead of saved again.
4. Build the hull with `compute_hull(..., include_generated_p1=False)`.

## Adding a published dataset

1. **License first.** Records licensed in `NON_COMMERCIAL_LICENSES` aren't imported.
2. Add a `Source(..., kind=PUBLISHED, license=..., url=..., hull_key=...)` to `SOURCES`.
3. Write a reader yielding `SourceRecord`s: the raw PBE energy per atom on the published cell, and the source's own hull distance in metadata under `hull_key`.
4. Add the provider to `scripts/import.py`. A re-import refreshes rows in place, and `--retire-missing` retires records the source no longer publishes.
5. Run `scripts/published_hulls.py`, then single-points near the PBE hull for each MLIP surface.

## Adding an energy surface

Add a `Surface` to `SURFACES` and fill its energies near the PBE hull. Callers fill whatever else they need inline with `references.fill_near_hull`.

## License

MIT. Imported data keeps its own license: Materials Project and Alexandria are CC BY 4.0.
