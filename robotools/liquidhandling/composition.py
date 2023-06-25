"""Functions for tracking fluid composition through liquid handling operations."""

from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np


def combine_composition(
    volume_A: float,
    composition_A: Optional[Mapping[str, float]],
    volume_B: float,
    composition_B: Optional[Mapping[str, float]],
) -> Optional[Dict[str, float]]:
    """Computes the composition of a liquid, created by the mixing of two liquids (A and B).

    Parameters
    ----------
    volume_A : float
        Volume of liquid A
    composition_A : dict
        Relative composition of liquid A
    volume_B : float
        Volume of liquid B
    composition_B : dict
        Relative composition of liquid B

    Returns
    -------
    composition : dict
        Composition of the new liquid created by mixing the given volumes of A and B
    """
    if composition_A is None or composition_B is None:
        return None
    # convert to volumetric fractions
    volumetric_fractions = {k: f * volume_A for k, f in composition_A.items()}
    # volumetrically add incoming fractions
    for k, f in composition_B.items():
        if not k in volumetric_fractions:
            volumetric_fractions[k] = 0
        volumetric_fractions[k] += f * volume_B
    # convert back to relative fractions
    new_composition = {k: v / (volume_A + volume_B) for k, v in volumetric_fractions.items()}
    return new_composition


def get_initial_composition(
    name: str,
    real_wells: np.ndarray,
    component_names: Mapping[str, Optional[str]],
    initial_volumes: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Creates a dictionary of initial composition arrays.

    Parameters
    ----------
    name : str
        Name of the labware - used for default component names.
    real_wells : array-like
        2D array of non-virtual wells in the labware.
    component_names : dict
        User-provided dictionary that maps real well IDs to component names.
    initial_volumes : np.ndarray
        Initial volumes of real wells.

    Returns
    -------
    composition : dict
        The component-wise dictionary of numpy arrays that describe the composition of real wells.
    """
    possible_component_wells = set(np.unique(real_wells))
    illegal_component_wells = set(component_names.keys()) - possible_component_wells
    if illegal_component_wells:
        raise ValueError(f"Invalid component name keys: {illegal_component_wells}")

    is_multiwell = len(real_wells) > 1
    composition: Dict[str, np.ndarray] = {}
    for idx, w in np.ndenumerate(real_wells):
        # Ignore None-valued component names, but don't allow naming of empty wells.
        if initial_volumes[idx] == 0:
            if component_names.get(w, None) is not None:
                raise ValueError(
                    f"A component name '{component_names[w]}' was specified for {name}.{w}, but the corresponding initial volume is 0."
                )
            continue

        # Fetch a name for identifying the liquid from this non-empty well
        cname = component_names.get(w, None)
        if cname is None:
            default_name = f"{name}.{w}" if is_multiwell else name
            cname = default_name

        # Make sure that a composition array exists
        if cname not in composition:
            composition[cname] = np.zeros_like(real_wells, dtype=float)

        # Mark this well as filled by this component
        composition[cname][idx] = 1
    return composition


def get_trough_component_names(
    name: str,
    columns: int,
    column_names: Sequence[Optional[str]],
    initial_volumes: Sequence[Union[int, float]],
) -> Dict[str, Optional[str]]:
    """Determines a fully-specified component name dictionary for a trough.

    This helper function exists to provide a different default naming pattern for troughs.
    Instead of "stocks.A01" this function defaults to "stocks.column_01" with 1-based column numbering.

    Parameters
    ----------
    name : str
        Name of the trough - used for default component names.
    columns : int
        Number of trough columns.
    column_names : array-like
        Column-wise component names.
        Must be given for all columns, but can contain None elements.
    initial_volumes : array-like
        Column-wise initial volumes.
        Used to determine if a default component name is needed.

    Returns
    -------
    component_names : dict
        The component name dictionary that maps all row A well IDs to component names.
    """
    if np.shape(column_names) != (columns,):
        raise ValueError(f"The column names {column_names} don't match the number of columns ({columns}).")
    if np.shape(initial_volumes) != (columns,):
        raise ValueError(
            f"The initial volumes {initial_volumes} don't match the number of columns ({columns})."
        )

    if any([cname is not None and ivol == 0 for cname, ivol in zip(column_names, initial_volumes)]):
        raise ValueError(
            f"Empty columns must be unnamed."
            f"\n\tcolumn_names: {column_names}"
            f"\n\tinitial_volumes: {initial_volumes}"
        )

    component_names = {}
    for c, (cname, ivol) in enumerate(zip(column_names, initial_volumes)):
        if ivol > 0 and cname is None:
            # Determine default name
            if columns > 1:
                cname = f"{name}.column_{c+1:02d}"
            else:
                cname = name
        component_names[f"A{c+1:02d}"] = cname
    return component_names
