""" Creating worklist files for the Tecan Freedom EVO.
"""
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy

from robotools import liquidhandling
from robotools.evotools.types import Tip
from robotools.liquidhandling import Labware
from robotools.worklists.exceptions import CompatibilityError, InvalidOperationError
from robotools.worklists.utils import prepare_aspirate_dispense_parameters

__all__ = ("BaseWorklist",)

logger = logging.getLogger(__name__)


class BaseWorklist(list):
    """Context manager for the creation of Worklists."""

    def __init__(
        self,
        filepath: Optional[Union[str, Path]] = None,
        max_volume: Union[int, float] = 950,
        auto_split: bool = True,
    ) -> None:
        """Creates a worklist writer.

        Parameters
        ----------
        filepath
            Optional filename/filepath to write when the context is exited (must include a .gwl extension)
        max_volume : int
            Maximum aspiration volume in µL
        auto_split : bool
            If `True`, large volumes in transfer operations are automatically splitted.
            If set to `False`, `InvalidOperationError` is raised when a pipetting volume exceeds `max_volume`.
        """
        self._filepath: Optional[Path] = None
        if filepath is not None:
            self._filepath = Path(filepath)
        if max_volume is None:
            raise ValueError("The `max_volume` parameter is required.")
        self.max_volume = max_volume
        self.auto_split = auto_split
        super().__init__()

    @property
    def filepath(self) -> Optional[Path]:
        """Path to which the worklist will write, if specified."""
        if self._filepath is not None:
            return Path(self._filepath)
        return None

    def __enter__(self) -> "BaseWorklist":
        self.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._filepath:
            self.save(self._filepath)
        return

    def _get_well_position(self, labware: Labware, well: str) -> int:
        """Internal method to resolve the well number for a given labware well."""
        raise TypeError(
            "The use of a specific worklist type (typically EvoWorklist or FluentWorklist) is required for this operation."
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Writes the worklist to the filepath.

        Parameters
        ----------
        filepath
            File name or path to write (must include a .gwl extension)
        """
        filepath = Path(filepath)
        assert ".gwl" in filepath.name.lower(), "The filename did not contain the .gwl extension."
        filepath.unlink(missing_ok=True)
        with open(filepath, "w", newline="\r\n", encoding="latin_1") as file:
            file.write("\n".join(self))
        return

    def comment(self, comment: Optional[str]) -> None:
        """Adds a comment.

        Parameters
        ----------
        comment : str
            A single- or multi-line comment. Be nice and avoid special characters.
        """
        if not comment:
            return
        if ";" in comment:
            raise ValueError("Illegal semicolon in comment.")
        for cline in comment.split("\n"):
            cline = cline.strip()
            if cline:
                self.append(f"C;{cline}")
        return

    def wash(self, scheme: int = 1) -> None:
        """Washes fixed tips or replaces DiTis.

        Washes/replaces the tip that was used by the preceding aspirate record(s).

        Parameters
        ----------
        scheme : int
            Number indicating the wash scheme (default: 1)
        """
        if not scheme in {1, 2, 3, 4}:
            raise ValueError("scheme must be either 1, 2, 3 or 4")
        self.append(f"W{scheme};")
        return

    def decontaminate(self) -> None:
        """Decontamination wash consists of a decontamination wash followed by a normal wash."""
        self.append("WD;")
        return

    def flush(self) -> None:
        """Discards the contents of the tips WITHOUT WASHING or DROPPING of tips."""
        self.append("F;")
        return

    def commit(self) -> None:
        """Inserts a 'break' that forces the execution of aspirate/dispense operations at this point.

        WARNING: may be unreliable

        If you don’t specify a Break record, Freedom EVOware normally executes
        pipetting commands in groups to optimize the efficiency. For example, if
        you have specified four tips in the Worklist command, Freedom EVOware
        will queue Aspirate records until four of them are ready for execution.
        This allows pipetting to take place using all four tips at the same time.
        Specify the Break record if you want to execute all of the currently queued
        commands without waiting. You can use the Break record e.g. to create a
        worklist which pipettes using only one tip at a time (even if you chose
        more than one tip in the tip selection).
        """
        self.append("B;")
        return

    def set_diti(self, diti_index: int) -> None:
        """Switches the DiTi types within the worklist.

        IMPORTANT: As the DiTi index in worklists is 1-based you have to increase the shown DiTi index by one.

        Choose the required DiTi type by specifying the DiTi index.
        Freedom EVOware automatically assigns a unique index to each DiTi type.
        The DiTi index is shown in the Edit Labware dialog box for the DiTi labware (Well dimensions tab).

        The Set DiTi Type record can only be used at the very beginning of the
        worklist or directly after a Break record. A Break record always resets
        the DiTi type to the type selected in the Worklist command. Accordingly,
        if your worklist contains a Break record, you may need to specify the
        Set DiTi Type record again.

        Parameters
        ----------
        diti_index : int
            Type of DiTis to use in subsequent steps
        """
        if not (len(self) == 0 or self[-1][0] == "B"):
            raise InvalidOperationError(
                "DiTi type can only be switched at the beginning or after a Break/commit step. Read the docstring."
            )
        self.append(f"S;{diti_index}")
        return

    def aspirate_well(
        self,
        rack_label: str,
        position: int,
        volume: float,
        *,
        liquid_class: str = "",
        tip: Union[Tip, int, Iterable] = Tip.Any,
        rack_id: str = "",
        tube_id: str = "",
        rack_type: str = "",
        forced_rack_type: str = "",
    ) -> None:
        """Command for aspirating with a single tip.

        Each Aspirate record specifies the aspiration parameters for a single tip (the next unused tip from the tip selection you have specified).

        Parameters
        ----------
        rack_label : str
            User-defined labware name (max 32 characters)
        position : int
            Number of the well
        volume : float
            Volume in microliters (will be rounded to 2 decimal places)
        liquid_class : str, optional
            Overwrites the liquid class for this step (max 32 characters)
        tip : Tip, int or Iterable of Tip / int, optional
            Tip that will be selected (Tip, 1-8 or Iterable of the former two)
        rack_id : str, optional
            Barcode of the labware (max 32 characters)
        tube_id : str, optional
            Barcode of the tube (max 32 characters)
        rack_type : str
            Configuration name of the labware (max 32 characters).
            An error is raised if it missmatches with the underlying worktable.
        forced_rack_type : str, optional
            Overrides rack_type from worktable
        """
        (
            rack_label,
            position,
            volume_s,
            liquid_class,
            tip,
            rack_id,
            tube_id,
            rack_type,
            forced_rack_type,
        ) = prepare_aspirate_dispense_parameters(
            rack_label,
            position,
            volume,
            liquid_class,
            tip,
            rack_id,
            tube_id,
            rack_type,
            forced_rack_type,
            max_volume=self.max_volume,
        )
        tip_type = ""
        self.append(
            f"A;{rack_label};{rack_id};{rack_type};{position};{tube_id};{volume_s};{liquid_class};{tip_type};{tip};{forced_rack_type}"
        )
        return

    def dispense_well(
        self,
        rack_label: str,
        position: int,
        volume: float,
        *,
        liquid_class: str = "",
        tip: Union[Tip, int] = Tip.Any,
        rack_id: str = "",
        tube_id: str = "",
        rack_type: str = "",
        forced_rack_type: str = "",
    ) -> None:
        """Command for dispensing with a single tip.

        Each Dispense record specifies the dispense parameters for a single tip.
        It uses the same tip which was used by the preceding Aspirate record.

        Parameters
        ----------
        rack_label : str
            User-defined labware name (max 32 characters)
        position : int
            Number of the well
        volume : float
            Volume in microliters (will be rounded to 2 decimal places)
        liquid_class : str, optional
            Overwrites the liquid class for this step (max 32 characters)
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
        """
        (
            rack_label,
            position,
            volume_s,
            liquid_class,
            tipv,
            rack_id,
            tube_id,
            rack_type,
            forced_rack_type,
        ) = prepare_aspirate_dispense_parameters(
            rack_label,
            position,
            volume,
            liquid_class,
            tip,
            rack_id,
            tube_id,
            rack_type,
            forced_rack_type,
            max_volume=self.max_volume,
        )
        tip_type = ""
        self.append(
            f"D;{rack_label};{rack_id};{rack_type};{position};{tube_id};{volume_s};{liquid_class};{tip_type};{tipv};{forced_rack_type}"
        )
        return

    def reagent_distribution(
        self,
        src_rack_label: str,
        src_start: int,
        src_end: int,
        dst_rack_label: str,
        dst_start: int,
        dst_end: int,
        *,
        volume: float,
        diti_reuse: int = 1,
        multi_disp: int = 1,
        exclude_wells: Optional[Iterable[int]] = None,
        liquid_class: str = "",
        direction: str = "left_to_right",
        src_rack_id: str = "",
        src_rack_type: str = "",
        dst_rack_id: str = "",
        dst_rack_type: str = "",
    ) -> None:
        """Transfers from a Trough into many destination wells using multi-pipetting.

        ⚠ This is the low-level version. Use ``.distribute()`` for a more user-friendly signature. ⚠

        Parameters
        ----------
        src_rack_label : str
            Name of the source labware on the worktable
        src_start : int
            First well to be used in the source labware
        end_start : int
            Last well to be used in the source labware
        src_rack_label : str
            Name of the destination labware on the worktable
        src_start : int
            First well to be used in the destination labware
        end_start :int
            Last well to be used in the destination labware
        volume : float
            Microliters to dispense into each destination
        diti_reuse : int
            Number of allowed re-uses for disposable tips
        multi_disp : int
            Maximum number of allowed multi-dispenses
        exclude_wells : list
            Numbers of destination wells to skip
        liquid_class : str
            Liquid class to use for the operation
        direction : str
            Moving direction on the destination ('left_to_right' or 'right_to_left')
        src_rack_id : str, optional
            Barcode of the source labware
        src_rack_type : str, optional
            Configuration name of the source labware
        dst_rack_id : str, optional
            Barcode of the destination labware
        dst_rack_type : str, optional
            Configuration name of the destination labware
        """
        # check & convert arguments
        if not direction in {"left_to_right", "right_to_left"}:
            raise ValueError(f'"direction" must be either "left_to_right" or "right_to_left"')
        direction_i = 0 if direction == "left_to_right" else 1

        if exclude_wells is None:
            exclude_list = []
        else:
            exclude_list = list(exclude_wells)
        if len(exclude_list) > 0:
            # check that all excluded wells fall in the range
            dst_range = set(range(dst_start, dst_end + 1))
            invalid_exclusion_wells = set(exclude_list).difference(dst_range)
            if len(invalid_exclusion_wells) > 0:
                raise ValueError(
                    f"The excluded wells {invalid_exclusion_wells} are not in the destination interval [{dst_start},{dst_end}]"
                )
            # condense into ;-separated text
            exclude_str = ";" + ";".join(map(str, sorted(exclude_list)))
        else:
            exclude_str = ""

        src_args = (src_rack_label, 1, volume, "", Tip.Any, src_rack_id, "", src_rack_type, "")
        (
            src_rack_label,
            _,
            _,
            _,
            _,
            src_rack_id,
            _,
            src_rack_type,
            _,
        ) = prepare_aspirate_dispense_parameters(*src_args, max_volume=self.max_volume)

        dst_args = (dst_rack_label, 1, volume, "", Tip.Any, dst_rack_id, "", dst_rack_type, "")
        (
            dst_rack_label,
            _,
            _,
            _,
            _,
            dst_rack_id,
            _,
            dst_rack_type,
            _,
        ) = prepare_aspirate_dispense_parameters(*dst_args, max_volume=self.max_volume)

        # automatically decrease multi_disp to support the large volume
        # at the expense of more washing
        if multi_disp * volume > self.max_volume:
            logger.warning(
                "Decreasing `multi_disp` to account for a large dispense volume. The number of washs will increase."
            )
            multi_disp = math.floor(self.max_volume / volume)

        src_parameters = f"{src_rack_label};{src_rack_id};{src_rack_type};{src_start};{src_end}"
        dst_parameters = f"{dst_rack_label};{dst_rack_id};{dst_rack_type};{dst_start};{dst_end}"
        self.append(
            f"R;{src_parameters};{dst_parameters};{volume};{liquid_class};{diti_reuse};{multi_disp};{direction_i}{exclude_str}"
        )
        return

    def aspirate(
        self,
        labware: liquidhandling.Labware,
        wells: Union[str, Sequence[str], numpy.ndarray],
        volumes: Union[float, Sequence[float], numpy.ndarray],
        *,
        label: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Performs aspiration from the provided labware.

        Parameters
        ----------
        labware : liquidhandling.Labware
            Source labware
        wells : str or iterable
            List of well ids
        volumes : float or iterable
            Volume(s) to aspirate
        label : str
            Label of the operation to log into labware history
        kwargs
            Additional keyword arguments to pass to `aspirate_well`.
            Most prominent example: `liquid_class`.
            Take a look at `Worklist.aspirate_well` for the full list of options.
        """
        wells = numpy.array(wells).flatten("F")
        volumes = numpy.array(volumes).flatten("F")
        if len(volumes) == 1:
            volumes = numpy.repeat(volumes, len(wells))
        labware.remove(wells, volumes, label)
        self.comment(label)
        for well, volume in zip(wells, volumes):
            if volume > 0:
                self.aspirate_well(labware.name, self._get_well_position(labware, well), volume, **kwargs)
        return

    def dispense(
        self,
        labware: liquidhandling.Labware,
        wells: Union[str, Sequence[str], numpy.ndarray],
        volumes: Union[float, Sequence[float], numpy.ndarray],
        *,
        label: Optional[str] = None,
        compositions: Optional[List[Optional[Dict[str, float]]]] = None,
        **kwargs,
    ) -> None:
        """Performs dispensing into the provided labware.

        Parameters
        ----------
        labware : liquidhandling.Labware
            Source labware
        wells : str or iterable
            List of well ids
        volumes : float or iterable
            Volume(s) to dispense
        label : str
            Label of the operation to log into labware history
        compositions : list
            Iterable of liquid compositions
        kwargs
            Additional keyword arguments to pass to `dispense_well`.
            Most prominent example: `liquid_class`.
            Take a look at `Worklist.dispense_well` for the full list of options.
        """
        wells = numpy.array(wells).flatten("F")
        volumes = numpy.array(volumes).flatten("F")
        if len(volumes) == 1:
            volumes = numpy.repeat(volumes, len(wells))
        labware.add(wells, volumes, label, compositions=compositions)
        self.comment(label)
        for well, volume in zip(wells, volumes):
            if volume > 0:
                self.dispense_well(labware.name, self._get_well_position(labware, well), volume, **kwargs)
        return

    def transfer(
        self,
        source: liquidhandling.Labware,
        source_wells: Union[str, Sequence[str], numpy.ndarray],
        destination: liquidhandling.Labware,
        destination_wells: Union[str, Sequence[str], numpy.ndarray],
        volumes: Union[float, Sequence[float], numpy.ndarray],
        *,
        label: Optional[str] = None,
        wash_scheme: int = 1,
        partition_by: str = "auto",
        **kwargs,
    ):
        raise CompatibilityError(
            "The transfer method is Fluent/Evo-specific, but this object is of the generic BaseWorklist type."
            " Use an EvoWorklist or FluentWorklist for device-specific methods."
        )

    def distribute(
        self,
        source: liquidhandling.Labware,
        source_column: int,
        destination: liquidhandling.Labware,
        destination_wells: Union[str, Sequence[str], numpy.ndarray],
        *,
        volume: float,
        diti_reuse: int = 1,
        multi_disp: int = 1,
        liquid_class: str = "",
        label: str = "",
        direction: str = "left_to_right",
        src_rack_id: str = "",
        src_rack_type: str = "",
        dst_rack_id: str = "",
        dst_rack_type: str = "",
    ) -> None:
        """Transfers from a Trough into many destination wells using multi-pipetting.

        Does NOT support large volume operations.

        Parameters
        ----------
        source : liquidhandling.Labware
            Source labware with virtual_rows (a Trough)
        source_column : int
            0-based column number of the reagent in the source labware
        destination : liquidhandling.Labware
            Destination labware
        destination_wells : array-like
            List or array of destination wells
        volume : float
            Microliters to dispense into each destination
        multi_disp : int
            Maximum number of allowed multi-dispenses
        liquid_class : str
            Liquid class to use for the operation
        label : str
            Label of the operation
        diti_reuse : int
            Number of allowed re-uses for disposable tips
        direction : str
            Moving direction on the destination ('left_to_right' or 'right_to_left')
        src_rack_id : str, optional
            Barcode of the source labware
        src_rack_type : str, optional
            Configuration name of the source labware
        dst_rack_id : str, optional
            Barcode of the destination labware
        dst_rack_type : str
            Configuration name of the destination labware
        """
        if source.virtual_rows is None:
            raise ValueError(
                f'Reagent distribution only works with Trough sources. "{source.name}" is not a Trough.'
            )

        if volume > self.max_volume:
            raise InvalidOperationError(
                f"Reagent distribution only works with volumes smaller than the diluter volume ({self.max_volume} µl)"
            )

        # always use the entire first column of the source
        src_start = 1 + source.n_rows * source_column
        src_end = src_start + source.n_rows - 1

        # transform destination wells into range + mask
        destination_wells = numpy.array(destination_wells).flatten("F")
        dst_wells = list(sorted([self._get_well_position(destination, w) for w in destination_wells]))
        dst_start, dst_end = dst_wells[0], dst_wells[-1]
        excluded_dst_wells = set(range(dst_start, dst_end + 1)).difference(dst_wells)

        # hand over to low-level command implementation
        self.comment(label)
        self.reagent_distribution(
            source.name,
            src_start,
            src_end,
            destination.name,
            dst_start,
            dst_end,
            volume=volume,
            diti_reuse=diti_reuse,
            multi_disp=multi_disp,
            exclude_wells=excluded_dst_wells,
            liquid_class=liquid_class,
            direction=direction,
            src_rack_id=src_rack_id,
            src_rack_type=src_rack_type,
            dst_rack_id=dst_rack_id,
            dst_rack_type=dst_rack_type,
        )

        # update volume tracking
        n_dst = len(dst_wells)
        source.remove(source.wells[0, source_column], volume * n_dst, label=label)
        src_composition = source.get_well_composition(source.wells[0, source_column])
        destination.add(destination_wells, volume, label=label, compositions=[src_composition] * n_dst)
        return

    def __repr__(self) -> str:
        return "\n".join(self)

    def __str__(self) -> str:
        return self.__repr__()
