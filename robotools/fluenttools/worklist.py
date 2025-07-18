import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

from robotools.fluenttools.utils import get_well_position
from robotools.liquidhandling.labware import Labware
from robotools.worklists.base import BaseWorklist
from robotools.worklists.utils import (
    optimize_partition_by,
    partition_by_column,
    partition_volume,
)

__all__ = ("FluentWorklist",)


class FluentWorklist(BaseWorklist):
    """Context manager for the creation of Tecan Fluent worklists."""

    def __init__(
        self,
        filepath: Optional[Union[str, Path]] = None,
        max_volume: Union[int, float] = 950,
        auto_split: bool = True,
        diti_mode: bool = False,
    ) -> None:
        super().__init__(filepath, max_volume, auto_split, diti_mode)

    def _get_well_position(self, labware: Labware, well: str) -> int:
        return get_well_position(labware, well)

    def transfer(
        self,
        source: Labware,
        source_wells: Union[str, Sequence[str], np.ndarray],
        destination: Labware,
        destination_wells: Union[str, Sequence[str], np.ndarray],
        volumes: Union[float, Sequence[float], np.ndarray],
        *,
        label: Optional[str] = None,
        wash_scheme: Literal[1, 2, 3, 4, "flush", "reuse"] = 1,
        partition_by: str = "auto",
        on_underflow: Literal["debug", "warn", "raise"] = "raise",
        **kwargs,
    ) -> None:
        """Transfer operation between two labwares.

        Parameters
        ----------
        source
            Source labware
        source_wells
            List of source well ids
        destination
            Destination labware
        destination_wells
            List of destination well ids
        volumes
            Volume(s) to transfer
        label
            Label of the operation to log into labware history
        wash_scheme
            - One of ``{1, 2, 3, 4}`` to select a wash scheme for fixed tips,
            or drop tips when using DiTis.
            - ``"flush"`` blows out tips, but does not drop DiTis, and only does a short wash with fixed tips.
            - ``"reuse"`` continues pipetting without flushing, dropping or washing.
            Passing ``None`` is deprecated, results in ``"flush"`` behavior and emits a warning.

        partition_by : str
            one of 'auto' (default), 'source' or 'destination'
                'auto': partitioning by source unless the source is a Trough
                'source': partitioning by source columns
                'destination': partitioning by destination columns
        on_underflow
            What to do about volume underflows (going below ``vmin``) in non-empty wells.

            Options:

            - ``"debug"`` mentions the underflowing wells in a log message at DEBUG level.
            - ``"warn"`` emits an :class:`~robotools.liquidhandling.exceptions.VolumeUnderflowWarning`. This `can be captured in unit tests <https://docs.pytest.org/en/stable/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests>`_.
            - ``"raise"`` raises a :class:`~robotools.liquidhandling.exceptions.VolumeUnderflowError` about underflowing wells.
        kwargs
            Additional keyword arguments to pass to aspirate and dispense.
            Most prominent example: `liquid_class`.
            Take a look at `Worklist.aspirate_well` for the full list of options.
        """
        # reformat the convenience parameters
        source_wells = np.array(source_wells).flatten("F")
        destination_wells = np.array(destination_wells).flatten("F")
        volumes = np.array(volumes).flatten("F")
        nmax = max((len(source_wells), len(destination_wells), len(volumes)))

        # Deal with deprecated behavior
        if wash_scheme is None:
            warnings.warn(
                "wash_scheme=None is deprecated. For flushing pass 'flush'.", DeprecationWarning, stacklevel=2
            )
            wash_scheme = "flush"

        if len(source_wells) == 1:
            source_wells = np.repeat(source_wells, nmax)
        if len(destination_wells) == 1:
            destination_wells = np.repeat(destination_wells, nmax)
        if len(volumes) == 1:
            volumes = np.repeat(volumes, nmax)
        lengths = (len(source_wells), len(destination_wells), len(volumes))
        if len(set(lengths)) != 1:
            raise ValueError(f"Number of source/destination/volumes must be equal. They were {lengths}")

        # automatic partitioning
        partition_by = optimize_partition_by(source, destination, partition_by, label)

        # the label applies to the entire transfer operation and is not logged at individual aspirate/dispense steps
        self.comment(label)
        nsteps = 0
        lvh_extra = 0

        for srcs, dsts, vols in partition_by_column(source_wells, destination_wells, volumes, partition_by):
            # make vector of volumes into vector of volume-lists
            vol_lists = [
                partition_volume(float(v), max_volume=self.max_volume) if self.auto_split else [v]
                for v in vols
            ]
            # transfer from this source column until all wells are done
            npartitions = max(map(len, vol_lists))
            # Count only the extra steps created by LVH
            lvh_extra += sum([len(vs) - 1 for vs in vol_lists])
            for p in range(npartitions):
                naccessed = 0
                # iterate the rows
                for s, d, vs in zip(srcs, dsts, vol_lists):
                    # transfer the next volume-fraction for this well
                    if len(vs) > p:
                        v = vs[p]
                        if v > 0:
                            vasp = self.aspirate(
                                source, s, v, label=None, on_underflow=on_underflow, **kwargs
                            )
                            self.dispense(
                                destination,
                                d,
                                v,
                                label=None,
                                compositions=[source.get_well_composition(s)],
                                vtrack=vasp,
                                **kwargs,
                            )
                            nsteps += 1
                            if wash_scheme == "flush":
                                self.flush()
                            elif wash_scheme == "reuse":
                                pass
                            else:
                                self.wash(scheme=wash_scheme)
                            naccessed += 1
                # LVH: if multiple wells are accessed, don't group across partitions
                if npartitions > 1 and naccessed > 1 and not p == npartitions - 1:
                    self.commit()
            # LVH: don't group across columns
            if npartitions > 1:
                self.commit()

        # Condense the labware logs into one operation
        # after the transfer operation completed to facilitate debugging.
        # Also include the number of extra steps because of LVH if applicable.
        if lvh_extra:
            if label:
                label = f"{label} ({lvh_extra} LVH steps)"
            else:
                label = f"{lvh_extra} LVH steps"
        if destination == source:
            source.condense_log(nsteps * 2, label=label)
        else:
            source.condense_log(nsteps, label=label)
            destination.condense_log(nsteps, label=label)
        return
