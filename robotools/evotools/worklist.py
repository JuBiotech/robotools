""" Creating worklist files for the Tecan Freedom EVO.
"""
import logging
import textwrap
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from robotools import liquidhandling
from robotools.evotools import commands
from robotools.evotools.types import Tip
from robotools.evotools.utils import get_well_position
from robotools.worklists.base import BaseWorklist
from robotools.worklists.utils import (
    optimize_partition_by,
    partition_by_column,
    partition_volume,
)

__all__ = ("EvoWorklist", "Worklist")

logger = logging.getLogger(__name__)


class EvoWorklist(BaseWorklist):
    """Context manager for the creation of Tecan EVO worklists."""

    def _get_well_position(self, labware: liquidhandling.Labware, well: str) -> int:
        return get_well_position(labware, well)

    def evo_aspirate(
        self,
        labware: liquidhandling.Labware,
        wells: Union[str, List[str]],
        labware_position: Tuple[int, int],
        tips: Union[List[Tip], List[int]],
        volumes: Union[float, List[float]],
        liquid_class: str,
        *,
        arm: int = 0,
        label: Optional[str] = None,
    ) -> None:
        """Performs aspiration from the provided labware. Is identical to the aspirate command inside the EvoWARE.
        Thus, several wells in a single column can be targeted.

        Parameters
        ----------
        labware : liquidhandling.Labware
            Source labware
        labware_position : tuple
            Grid position of the target labware on the robotic deck and site position on its carrier, e.g. labware on grid 38, site 2 -> (38,2)
        wells : list of str or iterable
            List with target well ID(s)
        tips : list
            Tip(s) that will be selected; use either a list with integers from 1 - 8 or with tip.T1 - tip.T8
        volumes : float or iterable
            Volume(s) in microliters (will be rounded to 2 decimal places); if several tips are used, these tips may aspirate individual volumes -> use list in these cases
        liquid_class : str, optional
            Overwrites the liquid class for this step (max 32 characters)
        arm : int
            Which LiHa to use, if more than one is available
        label : str
            Label of the operation to log into labware history
        """
        # diferentiate between what is needed for volume calculation and for pipetting commands
        wells_calc = np.array(wells).flatten("F")
        volumes_calc = np.array(volumes).flatten("F")
        if len(volumes_calc) == 1:
            volumes_calc = np.repeat(volumes_calc, len(wells_calc))
        labware.remove(wells_calc, volumes_calc, label)
        self.comment(label)
        cmd = commands.evo_aspirate(
            n_rows=labware.n_rows,
            n_columns=labware.n_columns,
            wells=wells,
            labware_position=labware_position,
            volume=volumes,
            liquid_class=liquid_class,
            tips=tips,
            arm=arm,
            max_volume=self.max_volume,
        )
        self.append(cmd)
        return

    def evo_dispense(
        self,
        labware: liquidhandling.Labware,
        wells: Union[str, List[str]],
        labware_position: Tuple[int, int],
        tips: Union[List[Tip], List[int]],
        volumes: Union[float, List[float]],
        liquid_class: str,
        *,
        arm: int = 0,
        label: Optional[str] = None,
        compositions: Optional[List[Optional[Dict[str, float]]]] = None,
    ) -> None:
        """Performs dispensation from the provided labware. Is identical to the dispense command inside the EvoWARE.
        Thus, several wells in a single column can be targeted.

        Parameters
        ----------
        labware : liquidhandling.Labware
            Source labware
        labware_position : tuple
            Grid position of the target labware on the robotic deck and site position on its carrier, e.g. labware on grid 38, site 2 -> (38,2)
        wells : list of str or iterable
            List with target well ID(s)
        tips : list
            Tip(s) that will be selected; use either a list with integers from 1 - 8 or with tip.T1 - tip.T8
        volumes : float or iterable
            Volume(s) in microliters (will be rounded to 2 decimal places); if several tips are used, these tips may aspirate individual volumes -> use list in these cases
        liquid_class : str, optional
            Overwrites the liquid class for this step (max 32 characters)
        arm : int
            Which LiHa to use, if more than one is available
        label : str
            Label of the operation to log into labware history
        compositions : list
            Iterable of liquid compositions
        """
        # diferentiate between what is needed for volume calculation and for pipetting commands
        wells_calc = np.array(wells).flatten("F")
        volumes_calc = np.array(volumes).flatten("F")
        if len(volumes_calc) == 1:
            volumes_calc = np.repeat(volumes_calc, len(wells_calc))
        labware.add(wells_calc, volumes_calc, label, compositions=compositions)
        self.comment(label)
        cmd = commands.evo_dispense(
            n_rows=labware.n_rows,
            n_columns=labware.n_columns,
            wells=wells,
            labware_position=labware_position,
            volume=volumes,
            liquid_class=liquid_class,
            tips=tips,
            arm=arm,
            max_volume=self.max_volume,
        )
        self.append(cmd)
        return

    def evo_wash(
        self,
        *,
        tips: Union[List[Tip], List[int]],
        waste_location: Tuple[int, int],
        cleaner_location: Tuple[int, int],
        arm: int = 0,
        waste_vol: float = 3.0,
        waste_delay: int = 500,
        cleaner_vol: float = 4.0,
        cleaner_delay: int = 500,
        airgap: int = 10,
        airgap_speed: int = 70,
        retract_speed: int = 30,
        fastwash: int = 1,
        low_volume: int = 0,
    ) -> None:
        """Command for aspirating with the EvoWARE aspirate command. As many wells in one column may be selected as your liquid handling arm has pipettes.
        This method generates the full command (as can be observed when opening a .esc file with an editor) and calls upon other functions to create the code string
        specifying the target wells.

        Parameters
        ----------
        tips : list
            Tip(s) that will be selected; use either a list with integers from 1 - 8 or with tip.T1 - tip.T8
        waste_location : tuple
            Tuple with grid position (1-67) and site number (0-127) of waste as integers
        cleaner_location : tuple
            Tuple with grid position (1-67) and site number (0-127) of cleaner as integers
        arm : int
            number of the LiHa performing the action: 0 = LiHa 1, 1 = LiHa 2
        waste_vol: float
            Volume in waste in mL (0-100)
        waste_delay : int
            Delay before closing valves in waste in ms (0-1000)
        cleaner_vol: float
            Volume in cleaner in mL (0-100)
        cleaner_delay : int
            Delay before closing valves in cleaner in ms (0-1000)
        airgap : int
            Volume of airgap in µL which is aspirated after washing the tips (system trailing airgap) (0-100)
        airgap_speed : int
            Speed of airgap aspiration in µL/s (1-1000)
        retract_speed : int
            Retract speed in mm/s (1-100)
        fastwash : int
            Use fast-wash module = 1, don't use it = 0
        low_volume : int
            Use pinch valves = 1, don't use them = 0
        """
        cmd = commands.evo_wash(
            tips=tips,
            waste_location=waste_location,
            cleaner_location=cleaner_location,
            arm=arm,
            waste_vol=waste_vol,
            waste_delay=waste_delay,
            cleaner_vol=cleaner_vol,
            cleaner_delay=cleaner_delay,
            airgap=airgap,
            airgap_speed=airgap_speed,
            retract_speed=retract_speed,
            fastwash=fastwash,
            low_volume=low_volume,
        )
        self.append(cmd)
        return

    def transfer(
        self,
        source: liquidhandling.Labware,
        source_wells: Union[str, Sequence[str], np.ndarray],
        destination: liquidhandling.Labware,
        destination_wells: Union[str, Sequence[str], np.ndarray],
        volumes: Union[float, Sequence[float], np.ndarray],
        *,
        label: Optional[str] = None,
        wash_scheme: int = 1,
        partition_by: str = "auto",
        **kwargs,
    ) -> None:
        """Transfer operation between two labwares.

        Parameters
        ----------
        source : liquidhandling.Labware
            Source labware
        source_wells : str or iterable
            List of source well ids
        destination : liquidhandling.Labware
            Destination labware
        destination_wells : str or iterable
            List of destination well ids
        volumes : float or iterable
            Volume(s) to transfer
        label : str
            Label of the operation to log into labware history
        wash_scheme : int
            Wash scheme to apply after every tip use
        partition_by : str
            one of 'auto' (default), 'source' or 'destination'
                'auto': partitioning by source unless the source is a Trough
                'source': partitioning by source columns
                'destination': partitioning by destination columns
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

        if len(source_wells) == 1:
            source_wells = np.repeat(source_wells, nmax)
        if len(destination_wells) == 1:
            destination_wells = np.repeat(destination_wells, nmax)
        if len(volumes) == 1:
            volumes = np.repeat(volumes, nmax)
        lengths = (len(source_wells), len(destination_wells), len(volumes))
        assert (
            len(set(lengths)) == 1
        ), f"Number of source/destination/volumes must be equal. They were {lengths}"

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
                            self.aspirate(source, s, v, label=None, **kwargs)
                            self.dispense(
                                destination,
                                d,
                                v,
                                label=None,
                                compositions=[source.get_well_composition(s)],
                                **kwargs,
                            )
                            nsteps += 1
                            if wash_scheme is not None:
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


class Worklist(EvoWorklist):
    def __init__(self, *args, **kwargs) -> None:
        msg = textwrap.dedent(
            """
            Robotools now distunguishes between EVO- and Fluent-compatible worklists.
            You created a 'Worklist', which will stop working in a future release.
            Instead please switch to one of the following options:
            1.) `robotools.EvoWorklist(...)` for EVO-compatible worklists.
            2.) `robotools.FluentWorklist(...)` for Fluent-compatible worklists.
            3.) `robotools.BaseWorklist(...)` for cross-compatible worklists with fewer features.
            """
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
