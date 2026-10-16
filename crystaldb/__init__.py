"""crystaldb: a crystal structure database with energies on several surfaces.

Reference structures (Materials Project, Alexandria), generator output and
user submissions in one SQLite file, with raw energies per energy surface
(PBE, MLIPs), MP2020 corrections on all of them, convex hulls, novelty
matching against published data, and a single-writer layer for sharing the
file between Modal apps.
"""

from .database import ExplorationRun, StoredStructure, StructureDatabase
from .sources import SOURCES, Source, get_source, is_generated, is_published
from .surfaces import DEFAULT_SURFACE, ORB_V3, PBE, SOURCE_GEOMETRY, SURFACES, Surface

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SURFACE",
    "ExplorationRun",
    "ORB_V3",
    "PBE",
    "SOURCES",
    "SOURCE_GEOMETRY",
    "SURFACES",
    "Source",
    "StoredStructure",
    "StructureDatabase",
    "Surface",
    "get_source",
    "is_generated",
    "is_published",
]
