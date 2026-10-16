"""Reading submitted CIFs: parse, and refuse what cannot be a crystal."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from pymatgen.core import Structure

# Shorter than any real bond (H-H is 0.74 Å): sites this close are
# duplicated or overlapping, not chemistry.
MIN_DISTANCE = 0.5
SYMPREC = 0.1


class InvalidStructure(ValueError):
    """A CIF that does not describe a usable crystal."""


def read_cif(data: Union[bytes, str], allow_disordered: bool = False) -> Structure:
    """The first structure in a CIF, or ``InvalidStructure`` saying why not.

    Partial occupancies are refused unless ``allow_disordered``: an ordered
    cell is needed for an energy.
    """
    from pymatgen.io.cif import CifParser

    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    if not text.strip():
        raise InvalidStructure("CIF file is empty")
    try:
        structures = CifParser.from_str(text).parse_structures(primitive=False)
    except Exception as exc:
        message = (str(exc).strip() or exc.__class__.__name__).splitlines()[0]
        raise InvalidStructure(f"Could not read the CIF: {message}") from None
    if not structures or len(structures[0]) == 0:
        raise InvalidStructure("CIF contains no atoms")
    structure = structures[0]
    if not structure.is_ordered and not allow_disordered:
        raise InvalidStructure(
            "Structure has partial occupancies; submit an ordered cell"
        )
    overlap = closest_pair(structure)
    if overlap is not None and overlap[2] < MIN_DISTANCE:
        i, j, d = overlap
        raise InvalidStructure(
            f"Sites {structure[i].species_string}{i + 1} and "
            f"{structure[j].species_string}{j + 1} are {d:.2f} Å apart "
            f"(under {MIN_DISTANCE} Å): overlapping atoms"
        )
    return structure


def closest_pair(structure: Structure) -> Optional[Tuple[int, int, float]]:
    """The two closest sites (periodic images included) and their distance.

    Partially occupied species sharing one position (occupancies summing to
    at most 1) are one site, not an overlap.
    """
    if len(structure) < 2:
        return None
    dists = np.array(structure.distance_matrix)
    np.fill_diagonal(dists, np.inf)
    if not structure.is_ordered:
        occupancy = np.array([site.species.num_atoms for site in structure])
        shared = (dists < 1e-6) & (occupancy[:, None] + occupancy[None, :] <= 1.0 + 1e-6)
        dists[shared] = np.inf
    i, j = np.unravel_index(np.argmin(dists), dists.shape)
    if not np.isfinite(dists[i, j]):
        return None
    return int(min(i, j)), int(max(i, j)), float(dists[i, j])


def space_group(structure: Structure) -> Tuple[Optional[int], Optional[str]]:
    """International number and symbol at ``SYMPREC``, or (None, None)."""
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        analyzer = SpacegroupAnalyzer(structure, symprec=SYMPREC)
        return analyzer.get_space_group_number(), analyzer.get_space_group_symbol()
    except Exception:
        return None, None
