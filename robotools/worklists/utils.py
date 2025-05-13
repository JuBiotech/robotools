"""Utility functions that are relevant for worklist commands."""
import collections
import logging
import math
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy

from robotools.evotools.types import Tip, int_to_tip
from robotools.worklists.exceptions import InvalidOperationError

from .. import liquidhandling

__all__ = (
    "prepare_aspirate_dispense_parameters",
    "optimize_partition_by",
    "partition_volume",
    "partition_by_column",
)

logger = logging.getLogger(__name__)


def prepare_aspirate_dispense_parameters(
    rack_label: str,
    position: int,
    volume: float,
    liquid_class: str = "",
    tip: Union[Tip, int, collections.abc.Iterable] = Tip.Any,
    rack_id: str = "",
    tube_id: str = "",
    rack_type: str = "",
    forced_rack_type: str = "",
    max_volume: Optional[Union[int, float]] = None,
) -> Tuple[str, int, str, str, Union[Tip, int, collections.abc.Iterable], str, str, str, str]:
    """Validates and prepares aspirate/dispense parameters.

    Parameters
    ----------
    rack_label : str
        User-defined labware name (max 32 characters)
    position : int
        Number of the well
    volume : float
        Volume in microliters (will be rounded to 2 decimal places)
    liquid_class : str, optional
        Overrides the liquid class for this step (max 32 characters)
    tip : Tip, int or Iterable of Tip / int, optional
        Tip that will be selected (Tip, 1-8 or Iterable of the former two)
    rack_id : str, optional
        Barcode of the labware (max 32 characters)
    tube_id : str, optional
        Barcode of the tube (max 32 characters)
    rack_type : str, optional
        Configuration name of the labware (max 32 characters).
        An error is raised if it missmatches with the underlying worktable.
    forced_rack_type : str, optional
        Overrides rack_type from worktable
    max_volume : int, optional
        Maximum allowed volume

    Returns
    -------
    rack_label : str
        User-defined labware name (max 32 characters)
    position : int
        Number of the well
    volume : str
        Volume in microliters (will be rounded to 2 decimal places)
    liquid_class : str
        Overrides the liquid class for this step (max 32 characters)
    tip : Tip, int or Iterable of Tip / int
        Tip that will be selected (Tip, 1-8 or Iterable of the former two)
    rack_id : str
        Barcode of the labware (max 32 characters)
    tube_id : str
        Barcode of the tube (max 32 characters)
    rack_type : str
        Configuration name of the labware (max 32 characters).
        An error is raised if it missmatches with the underlying worktable.
    forced_rack_type : str
        Overrides rack_type from worktable
    """
    # required parameters
    if rack_label is None:
        raise ValueError("Missing required parameter: rack_label")
    if not isinstance(rack_label, str) or len(rack_label) > 32 or ";" in rack_label:
        raise ValueError(f"Invalid rack_label: {rack_label}")

    if position is None:
        raise ValueError("Missing required parameter: position")
    if not isinstance(position, int) or position < 0:
        raise ValueError(f"Invalid position: {position}")

    if volume is None:
        raise ValueError("Missing required parameter: volume")
    try:
        volume = float(volume)
    except:
        raise ValueError(f"Invalid volume: {volume}")
    if volume < 0 or volume > 7158278 or numpy.isnan(volume):
        raise ValueError(f"Invalid volume: {volume}")
    if max_volume is not None and volume > max_volume:
        raise InvalidOperationError(f"Volume of {volume} exceeds max_volume.")

    # optional parameters
    if not isinstance(liquid_class, str) or ";" in liquid_class:
        raise ValueError(f"Invalid liquid_class: {liquid_class}")

    if isinstance(tip, int) and not isinstance(tip, Tip):
        # User-specified integers from 1-8 need to be converted to Tecan logic
        tip = int_to_tip(tip)

    if isinstance(tip, collections.abc.Iterable):
        tips = []
        for element in tip:
            if isinstance(element, int) and not isinstance(element, Tip):
                tips.append(int_to_tip(element))
            elif isinstance(element, Tip):
                if element == -1:
                    raise ValueError(
                        "When Iterables are used, no Tip.Any elements are allowed. Pass just one Tip.Any instead."
                    )
                tips.append(element)
            else:
                raise ValueError(
                    f"If tip is an Iterable, it may only contain int or Tip values, not {type(element)}."
                )
        tip = sum(set(tips))
    elif not isinstance(tip, Tip):
        raise ValueError(f"tip must be an int between 1 and 8, Tip or Iterable, but was {type(tip)}.")

    if not isinstance(rack_id, str) or len(rack_id) > 32 or ";" in rack_id:
        raise ValueError(f"Invalid rack_id: {rack_id}")
    if not isinstance(rack_type, str) or len(rack_type) > 32 or ";" in rack_type:
        raise ValueError(f"Invalid rack_type: {rack_type}")
    if not isinstance(forced_rack_type, str) or len(forced_rack_type) > 32 or ";" in forced_rack_type:
        raise ValueError(f"Invalid forced_rack_type: {forced_rack_type}")

    # apply rounding and corrections for the right string formatting
    volume_str = f"{numpy.round(volume, decimals=2):.2f}"
    tip = "" if tip == -1 else tip
    return rack_label, position, volume_str, liquid_class, tip, rack_id, tube_id, rack_type, forced_rack_type


def optimize_partition_by(
    source: liquidhandling.Labware,
    destination: liquidhandling.Labware,
    partition_by: str,
    label: Optional[str] = None,
) -> Literal["source", "destination"]:
    """Determines optimal partitioning settings.


    Parameters
    ----------
    source
        Source labware object.
    destination
        Destination labware object.
    partition_by
        User-provided partitioning settings.
    label
        Label of the operation (optional).

    ----------
    source (Labware): source labware object
    destination (Labware): destination labware object
    partition_by : str
    user-provided partitioning settings
    label : str
    label of the operation (optional)

    Returns
    -------
    partition_by
        Either 'source' or 'destination'
    """
    # automatic partitioning decision
    if partition_by == "auto":
        return "source"

    assert partition_by in {"source", "destination"}
    # log warnings about potentially inefficient partitioning settings
    if partition_by == "source":
        if source.is_trough and not destination.is_trough:
            logger.warning(
                'Partitioning by "source" (%s), which is a Trough while destination (%s) is not a Trough.'
                ' This is potentially inefficient. Consider using partition_by="destination".'
                " (label=%s)",
                source.name,
                destination.name,
                label,
            )
        return "source"
    elif partition_by == "destination":
        if destination.is_trough and not source.is_trough:
            logger.warning(
                'Partitioning by "destination" (%s), which is a Trough while source (%s) is not a Trough.'
                ' This is potentially inefficient. Consider using partition_by="source"'
                " (label=%s)",
                destination.name,
                source.name,
                label,
            )
        return "destination"
    raise ValueError(f"Invalid partition_by argument: {partition_by}")


def partition_volume(volume: float, *, max_volume: Union[int, float]) -> List[float]:
    """Partitions a pipetting volume into zero or more integer-valued volumes that are <= max_volume.

    Parameters
    ----------
    volume : float
        A volume to partition
    max_volume : int
        Maximum volume of a pipetting step

    Returns
    -------
    volumes : list
        Partitioned volumes
    """
    if volume == 0:
        return []
    if volume < max_volume:
        return [volume]
    isteps = math.ceil(volume / max_volume)
    step_volume = math.ceil(volume / isteps)
    volumes: List[float] = [step_volume] * (isteps - 1)
    volumes.append(volume - numpy.sum(volumes))
    return volumes


def non_repetitive_argsort(wells: Sequence[str]) -> list[int]:
    """Argsort without repeating items with the same first letter."""
    # Group wells by row
    by_row = collections.defaultdict(list)
    for iw, w in enumerate(wells):
        by_row[w[0]].append((iw, w))
    by_row_sorted = {r: by_row[r] for r in sorted(by_row)}

    # Collect the original index of the first entry
    # from each row, until all rows are empty.
    results = []
    while by_row_sorted:
        for r in list(by_row_sorted):
            row = by_row_sorted[r]
            iw, w = row.pop(0)
            results.append(iw)
            if not row:
                by_row_sorted.pop(r)
    return results


def partition_by_column(
    sources: Iterable[str],
    destinations: Iterable[str],
    volumes: Iterable[float],
    partition_by: Literal["source", "destination"],
) -> List[Tuple[List[str], List[str], List[float]]]:
    """Partitions sources/destinations/volumes by the source column and sorts within those columns.

    Parameters
    ----------
    sources : list
        The source well ids; same length as destinations and volumes
    destinations : list
        The destination well ids; same length as sources and volumes
    volumes : list
        The volumes; same length as sources and destinations
    partition_by : str
        Either 'source' or 'destination'

    Returns
    -------
    column_groups : list
        A list of (sources, destinations, volumes)
    """
    # first partition the wells into columns
    column_groups_dd: Dict[str, Tuple[List[str], List[str], List[float]]] = collections.defaultdict(
        lambda: ([], [], [])
    )
    for s, d, v in zip(sources, destinations, volumes):
        if partition_by == "source":
            group = s[1:]
        elif partition_by == "destination":
            group = d[1:]
        else:
            raise ValueError(f'Invalid `partition_by` parameter "{partition_by}""')
        column_groups_dd[group][0].append(s)
        column_groups_dd[group][1].append(d)
        column_groups_dd[group][2].append(v)
    # bring columns in the right order
    column_groups = [column_groups_dd[col] for col in sorted(column_groups_dd.keys())]
    # sort the rows within the column
    for c, (srcs, dsts, vols) in enumerate(column_groups):
        if partition_by == "source":
            order = non_repetitive_argsort(srcs)
        elif partition_by == "destination":
            order = non_repetitive_argsort(dsts)
        else:
            raise ValueError(f'Invalid `partition_by` parameter "{partition_by}""')
        column_groups[c] = (
            list(numpy.array(srcs)[order]),
            list(numpy.array(dsts)[order]),
            list(numpy.array(vols)[order]),
        )
    return column_groups
