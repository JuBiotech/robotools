""" Creating worklist files for the Tecan Freedom EVO.
"""
import collections
import enum
import logging
import math
import os
import typing

import numpy

from .. import liquidhandling

logger = logging.getLogger("evotools")


class Labwares(str, enum.Enum):
    """Built-in EVOware labware identifiers."""

    SystemLiquid = "Systemliquid"


class Tip(enum.IntEnum):
    Any = -1
    T1 = 1
    T2 = 2
    T3 = 4
    T4 = 8
    T5 = 16
    T6 = 32
    T7 = 64
    T8 = 128


class InvalidOperationError(Exception):
    pass


def _prepare_aspirate_dispense_parameters(
    rack_label: str,
    position: int,
    volume: float,
    liquid_class: str = "",
    tip: typing.Union[Tip, int, collections.abc.Iterable] = Tip.Any,
    rack_id: str = "",
    tube_id: str = "",
    rack_type: str = "",
    forced_rack_type: str = "",
    max_volume: typing.Optional[int] = None,
) -> typing.Tuple[str, int, float, str, typing.Union[Tip, int, collections.abc.Iterable], str, str, str, str]:
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
    volume : float
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
        raise ValueError("Missing required paramter: rack_label")
    if not isinstance(rack_label, str) or len(rack_label) > 32 or ";" in rack_label:
        raise ValueError(f"Invalid rack_label: {rack_label}")

    if position is None:
        raise ValueError("Missing required paramter: position")
    if not isinstance(position, int) or position < 0:
        raise ValueError(f"Invalid position: {position}")

    if volume is None:
        raise ValueError("Missing required paramter: volume")
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

    def _int_to_tip(tip_int: int):
        """Asserts a Tecan Tip class to an int between 1 and 8."""
        if not 1 <= tip_int <= 8:
            raise ValueError(
                f"Tip is {tip} with type {type(tip)}, but should be an int between 1 and 8 for _int_to_tip conversion."
            )
        if tip_int == 1:
            return Tip.T1
        elif tip_int == 2:
            return Tip.T2
        elif tip_int == 3:
            return Tip.T3
        elif tip_int == 4:
            return Tip.T4
        elif tip_int == 5:
            return Tip.T5
        elif tip_int == 6:
            return Tip.T6
        elif tip_int == 7:
            return Tip.T7
        elif tip_int == 8:
            return Tip.T8

    if isinstance(tip, int) and not isinstance(tip, Tip):
        # User-specified integers from 1-8 need to be converted to Tecan logic
        tip = _int_to_tip(tip)

    if isinstance(tip, collections.abc.Iterable):
        tips = []
        for element in tip:
            if isinstance(element, int) and not isinstance(element, Tip):
                tips.append(_int_to_tip(element))
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
    volume = f"{numpy.round(volume, decimals=2):.2f}"
    tip = "" if tip == -1 else tip
    return rack_label, position, volume, liquid_class, tip, rack_id, tube_id, rack_type, forced_rack_type


def _optimize_partition_by(
    source: liquidhandling.Labware,
    destination: liquidhandling.Labware,
    partition_by: str,
    label: typing.Optional[str] = None,
) -> str:
    """Determines optimal partitioning settings.

    Parameters
    ----------
    source (Labware): source labware object
    destination (Labware): destination labware object
    partition_by : str
    user-provided partitioning settings
    label : str
    label of the operation (optional)

    Returns
    -------
    partition_by : str
        Either 'source' or 'destination'
    """
    if not partition_by in {"auto", "source", "destination"}:
        raise ValueError(f"Invalid partition_by argument: {partition_by}")
    # automatic partitioning decision
    if partition_by == "auto":
        if source.is_trough and not destination.is_trough:
            logger.debug(f"")
            partition_by = "destination"
        else:
            partition_by = "source"
    else:
        # log warnings about potentially inefficient partitioning settings
        if partition_by == "source" and source.is_trough and not destination.is_trough:
            logger.warning(
                f'Partitioning by "source" ({source.name}), which is a Trough while destination ({destination.name}) is not a Trough.'
                ' This is potentially inefficient. Consider using partition_by="destination".'
                f" (label={label})"
            )
        elif partition_by == "destination" and destination.is_trough and not source.is_trough:
            logger.warning(
                f'Partitioning by "destination" ({destination.name}), which is a Trough while source ({source.name}) is not a Trough.'
                ' This is potentially inefficient. Consider using partition_by="source"'
                f" (label={label})"
            )
    return partition_by


def _partition_volume(volume: float, *, max_volume: int) -> typing.List[float]:
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
    volumes: typing.List[float] = [step_volume] * (isteps - 1)
    volumes.append(volume - numpy.sum(volumes))
    return volumes


def _partition_by_column(
    sources: typing.Iterable[str],
    destinations: typing.Iterable[str],
    volumes: typing.Iterable[float],
    partition_by: str,
) -> typing.Dict[int, typing.Tuple[typing.List[str], typing.List[str], typing.List[float]]]:
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
    column_groups = collections.defaultdict(lambda: ([], [], []))
    for s, d, v in zip(sources, destinations, volumes):
        if partition_by == "source":
            group = s[1:]
        elif partition_by == "destination":
            group = d[1:]
        else:
            raise ValueError(f'Invalid `partition_by` parameter "{partition_by}""')
        column_groups[group][0].append(s)
        column_groups[group][1].append(d)
        column_groups[group][2].append(v)
    # bring columns in the right order
    column_groups = [column_groups[col] for col in sorted(column_groups.keys())]
    # sort the rows within the column
    for c, (srcs, dsts, vols) in enumerate(column_groups):
        if partition_by == "source":
            order = numpy.argsort(srcs)
        elif partition_by == "destination":
            order = numpy.argsort(dsts)
        else:
            raise ValueError(f'Invalid `partition_by` parameter "{partition_by}""')
        column_groups[c] = (
            list(numpy.array(srcs)[order]),
            list(numpy.array(dsts)[order]),
            list(numpy.array(vols)[order]),
        )
    return column_groups


class Worklist(list):
    """Context manager for the creation of Worklists."""

    def __init__(self, filepath: str = None, max_volume: int = 950, auto_split: bool = True) -> None:
        """Creates a worklist writer.

        Parameters
        ----------
        filepath : str
            Optional filename/filepath to write when the context is exited (must include a .gwl extension)
        max_volume : int
            Maximum aspiration volume in µL
        auto_split : bool
            If `True`, large volumes in transfer operations are automatically splitted.
            If set to `False`, `InvalidOperationError` is raised when a pipetting volume exceeds `max_volume`.
        """
        self._filepath = filepath
        if max_volume is None:
            raise ValueError("The `max_volume` parameter is required.")
        self.max_volume = max_volume
        self.auto_split = auto_split
        super().__init__()

    def __enter__(self) -> None:
        self.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._filepath:
            self.save(self._filepath)
        return

    def save(self, filepath: str) -> None:
        """Writes the worklist to the filepath.

        Parameters
        ----------
        filepath : str
            File name or path to write (must include a .gwl extension)
        """
        assert ".gwl" in filepath.lower(), "The filename did not contain the .gwl extension."
        if os.path.exists(filepath):
            os.remove(filepath)
        with open(filepath, "w", newline="\r\n") as file:
            file.write("\n".join(self))
        return

    def comment(self, comment: typing.Optional[str]) -> None:
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
        tip: typing.Union[Tip, int, typing.Iterable] = Tip.Any,
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
        args = (
            rack_label,
            position,
            volume,
            liquid_class,
            tip,
            rack_id,
            tube_id,
            rack_type,
            forced_rack_type,
        )
        (
            rack_label,
            position,
            volume,
            liquid_class,
            tip,
            rack_id,
            tube_id,
            rack_type,
            forced_rack_type,
        ) = _prepare_aspirate_dispense_parameters(*args, max_volume=self.max_volume)
        tip_type = ""
        self.append(
            f"A;{rack_label};{rack_id};{rack_type};{position};{tube_id};{volume};{liquid_class};{tip_type};{tip};{forced_rack_type}"
        )
        return

    def dispense_well(
        self,
        rack_label: str,
        position: int,
        volume: float,
        *,
        liquid_class: str = "",
        tip: typing.Union[Tip, int] = Tip.Any,
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
        args = (
            rack_label,
            position,
            volume,
            liquid_class,
            tip,
            rack_id,
            tube_id,
            rack_type,
            forced_rack_type,
        )
        (
            rack_label,
            position,
            volume,
            liquid_class,
            tip,
            rack_id,
            tube_id,
            rack_type,
            forced_rack_type,
        ) = _prepare_aspirate_dispense_parameters(*args, max_volume=self.max_volume)
        tip_type = ""
        self.append(
            f"D;{rack_label};{rack_id};{rack_type};{position};{tube_id};{volume};{liquid_class};{tip_type};{tip};{forced_rack_type}"
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
        exclude_wells: typing.Optional[typing.Iterable[int]] = None,
        liquid_class: str = "",
        direction: str = "left_to_right",
        src_rack_id: str = "",
        src_rack_type: str = "",
        dst_rack_id: str = "",
        dst_rack_type: str = "",
    ) -> None:
        """Transfers from a Trough into many destination wells using multi-pipetting.

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
        direction = 0 if direction == "left_to_right" else 1

        if exclude_wells is None:
            exclude_wells = []
        if len(exclude_wells) > 0:
            # check that all excluded wells fall in the range
            dst_range = set(range(dst_start, dst_end + 1))
            invalid_exclusion_wells = set(exclude_wells).difference(dst_range)
            if len(invalid_exclusion_wells) > 0:
                raise ValueError(
                    f"The excluded wells {invalid_exclusion_wells} are not in the destination interval [{dst_start},{dst_end}]"
                )
            # condense into ;-separated text
            exclude_wells = ";" + ";".join(map(str, sorted(exclude_wells)))
        else:
            exclude_wells = ""

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
        ) = _prepare_aspirate_dispense_parameters(*src_args, max_volume=self.max_volume)

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
        ) = _prepare_aspirate_dispense_parameters(*dst_args, max_volume=self.max_volume)

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
            f"R;{src_parameters};{dst_parameters};{volume};{liquid_class};{diti_reuse};{multi_disp};{direction}{exclude_wells}"
        )
        return

    def aspirate(
        self,
        labware: liquidhandling.Labware,
        wells: typing.Union[str, typing.Sequence[str], numpy.ndarray],
        volumes: typing.Union[float, typing.Sequence[float], numpy.ndarray],
        *,
        label: typing.Optional[str] = None,
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
                self.aspirate_well(labware.name, labware.positions[well], volume, **kwargs)
        return

    def dispense(
        self,
        labware: liquidhandling.Labware,
        wells: typing.Union[str, typing.Sequence[str], numpy.ndarray],
        volumes: typing.Union[float, typing.Sequence[float], numpy.ndarray],
        *,
        label: typing.Optional[str] = None,
        compositions: typing.Optional[typing.List[typing.Optional[typing.Dict[str, float]]]] = None,
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
                self.dispense_well(labware.name, labware.positions[well], volume, **kwargs)
        return

    def transfer(
        self,
        source: liquidhandling.Labware,
        source_wells: typing.Union[str, typing.Sequence[str], numpy.ndarray],
        destination: liquidhandling.Labware,
        destination_wells: typing.Union[str, typing.Sequence[str], numpy.ndarray],
        volumes: typing.Union[float, typing.Sequence[float], numpy.ndarray],
        *,
        label: typing.Optional[str] = None,
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
        source_wells = numpy.array(source_wells).flatten("F")
        destination_wells = numpy.array(destination_wells).flatten("F")
        volumes = numpy.array(volumes).flatten("F")
        nmax = max((len(source_wells), len(destination_wells), len(volumes)))

        if len(source_wells) == 1:
            source_wells = numpy.repeat(source_wells, nmax)
        if len(destination_wells) == 1:
            destination_wells = numpy.repeat(destination_wells, nmax)
        if len(volumes) == 1:
            volumes = numpy.repeat(volumes, nmax)
        lengths = (len(source_wells), len(destination_wells), len(volumes))
        assert (
            len(set(lengths)) == 1
        ), f"Number of source/destination/volumes must be equal. They were {lengths}"

        # automatic partitioning
        partition_by = _optimize_partition_by(source, destination, partition_by, label)

        # the label applies to the entire transfer operation and is not logged at individual aspirate/dispense steps
        self.comment(label)
        nsteps = 0
        lvh_extra = 0

        for srcs, dsts, vols in _partition_by_column(source_wells, destination_wells, volumes, partition_by):
            # make vector of volumes into vector of volume-lists
            vols = [
                _partition_volume(float(v), max_volume=self.max_volume) if self.auto_split else [v]
                for v in vols
            ]
            # transfer from this source column until all wells are done
            npartitions = max(map(len, vols))
            # Count only the extra steps created by LVH
            lvh_extra += sum([len(vs) - 1 for vs in vols])
            for p in range(npartitions):
                naccessed = 0
                # iterate the rows
                for s, d, vs in zip(srcs, dsts, vols):
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

    def distribute(
        self,
        source: liquidhandling.Labware,
        source_column: int,
        destination: liquidhandling.Labware,
        destination_wells: typing.Union[str, typing.Sequence[str], numpy.ndarray],
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
        dst_wells = list(sorted([destination.positions[w] for w in destination_wells]))
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
