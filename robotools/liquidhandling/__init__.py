import logging
import typing
import warnings

import numpy

logger = logging.getLogger("liquidhandling")


class VolumeOverflowError(Exception):
    """Error that indicates the planned overflow of a well."""

    def __init__(
        self,
        labware: str,
        well: str,
        current: float,
        change: float,
        threshold: float,
        label: typing.Optional[str] = None,
    ) -> None:
        if label:
            super().__init__(
                f'Too much volume for "{labware}".{well}: {current} + {change} > {threshold} in step {label}'
            )
        else:
            super().__init__(f'Too much volume for "{labware}".{well}: {current} + {change} > {threshold}')


class VolumeUnderflowError(Exception):
    """Error that indicates the planned underflow of a well."""

    def __init__(
        self,
        labware: str,
        well: str,
        current: float,
        change: float,
        threshold: float,
        label: typing.Optional[str] = None,
    ) -> None:
        if label:
            super().__init__(
                f'Too little volume in "{labware}".{well}: {current} - {change} < {threshold} in step {label}'
            )
        else:
            super().__init__(f'Too little volume in "{labware}".{well}: {current} - {change} < {threshold}')


def _combine_composition(
    volume_A: float,
    composition_A: typing.Optional[typing.Dict[str, float]],
    volume_B: float,
    composition_B: typing.Optional[typing.Dict[str, float]],
) -> typing.Optional[typing.Dict[str, float]]:
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


def _get_initial_composition(
    name: str,
    real_wells: numpy.ndarray,
    component_names: typing.Dict[str, typing.Union[str, None]],
    initial_volumes: numpy.ndarray,
) -> typing.Dict[str, typing.Union[str, None]]:
    """Creates a dictionary of initial composition arrays.

    Parameters
    ----------
    name : str
        Name of the labware - used for default component names.
    real_wells : array-like
        2D array of non-virtual wells in the labware.
    component_names : dict
        User-provided dictionary that maps real well IDs to component names.
    initial_volumes : numpy.ndarray
        Initial volumes of real wells.

    Returns
    -------
    composition : dict
        The component-wise dictionary of numpy arrays that describe the composition of real wells.
    """
    possible_component_wells = set(numpy.unique(real_wells))
    illegal_component_wells = set(component_names.keys()) - possible_component_wells
    if illegal_component_wells:
        raise ValueError(f"Invalid component name keys: {illegal_component_wells}")

    is_multiwell = len(real_wells) > 1
    composition = {}
    for idx, w in numpy.ndenumerate(real_wells):
        # Ignore None-valued component names, but don't allow naming of empty wells.
        if initial_volumes[idx] == 0:
            if component_names.get(w, None) is not None:
                raise ValueError(
                    f"A component name '{component_names[w]}' was specified for {name}.{w}, but the corresponding initial volume is 0."
                )
            continue

        # Fetch a name for identifying the liquid from this non-empty well
        default_name = f"{name}.{w}" if is_multiwell else name
        cname = component_names.get(w, default_name)

        # Make sure that a composition array exists
        if cname not in composition:
            composition[cname] = numpy.zeros_like(real_wells, dtype=float)

        # Mark this well as filled by this component
        composition[cname][idx] = 1
    return composition


class Labware:
    """Represents an array of liquid cavities."""

    @property
    def history(self) -> typing.List[typing.Tuple[typing.Optional[str], numpy.ndarray]]:
        """List of label/volumes history."""
        return list(zip(self._labels, self._history))

    @property
    def report(self) -> str:
        """A printable report of the labware history."""
        report = self.name
        for label, state in self.history:
            if label:
                report += f"\n{label}"
            report += f"\n{numpy.round(state, decimals=1)}"
            report += "\n"
        return report

    @property
    def volumes(self) -> numpy.ndarray:
        """Current volumes in the labware."""
        return self._volumes.copy()

    @property
    def wells(self) -> numpy.ndarray:
        """Array of well ids."""
        return self._wells

    @property
    def indices(self) -> typing.Dict[str, typing.Tuple[int, int]]:
        """Mapping of well-ids to numpy indices."""
        return self._indices

    @property
    def positions(self) -> typing.Dict[str, int]:
        """Mapping of well-ids to EVOware-compatible position numbers."""
        return self._positions

    @property
    def n_rows(self) -> int:
        return len(self.row_ids)

    @property
    def n_columns(self) -> int:
        return len(self.column_ids)

    @property
    def is_trough(self) -> bool:
        return self.virtual_rows != None

    @property
    def composition(self) -> typing.Dict[str, numpy.ndarray]:
        """Relative composition of the liquids.

        This dictionary maps liquid names (keys) to arrays of relative amounts in each well.
        """
        return self._composition

    def __init__(
        self,
        name: str,
        rows: int,
        columns: int,
        *,
        min_volume: float,
        max_volume: float,
        initial_volumes: typing.Optional[typing.Union[float, numpy.ndarray]] = None,
        virtual_rows: typing.Optional[int] = None,
        component_names: typing.Optional[typing.Dict[str, str]] = None,
    ) -> None:
        """Creates a `Labware` object.

        Parameters
        ----------
        name : str
            Label that the labware is identified by.
        rows : int
            Number of rows in the labware
        columns : int
            Number of columns in the labware
        min_volume : float
            Filling volume that must remain after an aspirate operation.
        max_volume : float
            Maximum volume that must not be exceeded after a dispense.
        initial_volumes : float, array-like, optional
            Initial filling volume of the wells (default: 0)
        virtual_rows : int, optional
            When specified to a positive number, the `Labware` is treated as a trough.
            Must be used in combination with `rows=1`.
            For example: A `Labware` with virtual rows can be accessed with 6 Tips,
            but has just one row in the `volumes` array.
        component_names : dict, optional
            A dictionary that names the content of non-empty real wells for composition tracking.
        """
        # sanity checking
        if not isinstance(rows, int) or rows < 1:
            raise ValueError(f"Invalid rows: {rows}")
        if not isinstance(columns, int) or columns < 1:
            raise ValueError(f"Invalid columns: {columns}")
        if min_volume is None or min_volume < 0:
            raise ValueError(f"Invalid min_volume: {min_volume}")
        if max_volume is None or max_volume <= min_volume:
            raise ValueError(f"Invalid max_volume: {max_volume}")
        if virtual_rows is not None and rows != 1:
            raise ValueError("When using virtual_rows, the number of rows must be == 1")
        if virtual_rows is not None and virtual_rows < 1:
            raise ValueError(f"Invalid virtual_rows: {virtual_rows}")
        if virtual_rows and not isinstance(self, Trough):
            warnings.warn(
                "Troughs should be created with the robotools.Trough class.",
                UserWarning,
                stacklevel=2,
            )

        # explode convenience parameters
        if initial_volumes is None:
            initial_volumes = 0
        initial_volumes = numpy.array(initial_volumes)
        if initial_volumes.shape == ():
            initial_volumes = numpy.full((rows, columns), initial_volumes)
        else:
            initial_volumes = initial_volumes.reshape((rows, columns))
        assert initial_volumes.shape == (
            rows,
            columns,
        ), f"Invalid shape of initial_volumes: {initial_volumes.shape}"
        if numpy.any(initial_volumes < 0):
            raise ValueError("initial_volume cannot be negative")
        if numpy.any(initial_volumes > max_volume):
            raise ValueError("initial_volume cannot be above max_volume")

        # initialize properties
        self.name = name
        self.row_ids = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: rows if not virtual_rows else virtual_rows])
        self.column_ids = list(range(1, columns + 1))
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.virtual_rows = virtual_rows

        # generate arrays/mappings of well ids
        self._wells = numpy.array(
            [[f"{row}{column:02d}" for column in self.column_ids] for row in self.row_ids]
        )
        if virtual_rows is None:
            self._indices = {
                f"{row}{column:02d}": (r, c)
                for r, row in enumerate(self.row_ids)
                for c, column in enumerate(self.column_ids)
            }
            self._positions = {
                f"{row}{column:02d}": 1 + c * rows + r
                for r, row in enumerate(self.row_ids)
                for c, column in enumerate(self.column_ids)
            }
        else:
            self._indices = {
                f"{vrow}{column:02d}": (0, c)
                for vr, vrow in enumerate(self.row_ids)
                for c, column in enumerate(self.column_ids)
            }
            self._positions = {
                f"{vrow}{column:02d}": 1 + c * virtual_rows + vr
                for vr, vrow in enumerate(self.row_ids)
                for c, column in enumerate(self.column_ids)
            }

        # initialize state variables
        self._volumes = initial_volumes.copy().astype(float)
        self._history: typing.List[numpy.ndarray] = [self.volumes]
        self._labels: typing.List[typing.Optional[str]] = ["initial"]
        self._composition = _get_initial_composition(
            name,
            real_wells=self.wells[[0], :] if virtual_rows else self.wells,
            component_names=component_names or {},
            initial_volumes=initial_volumes,
        )
        super().__init__()

    def get_well_composition(self, well: str) -> typing.Dict[str, float]:
        """Retrieves the relative composition of a well.

        Parameters
        ----------
        well : str
            ID of the well for which to retrieve the composition.

        Returns
        -------
        composition : dict
            Keys: liquid names
            Values: relative amount
        """
        if self._composition is None:
            return None
        idx = self.indices[well]
        well_comp = {k: f[idx] for k, f in self.composition.items() if f[idx] > 0}
        return well_comp

    def add(
        self,
        wells: typing.Sequence[str],
        volumes: typing.Union[float, typing.Sequence[float], numpy.ndarray],
        label: typing.Optional[str] = None,
        compositions: typing.Optional[typing.List[typing.Optional[typing.Dict[str, float]]]] = None,
    ) -> None:
        """Adds volumes to wells.

        Parameters
        ----------
        wells : iterable of str
            Well ids
        volumes : float, iterable of float
            Scalar or iterable of volumes
        label : str
            Description of the operation
        compositions : iterable
            List of composition dictionaries ({ name : relative amount })
        """
        wells = numpy.array(wells).flatten("F")
        volumes = numpy.array(volumes).flatten("F")
        if len(volumes) == 1:
            volumes = numpy.repeat(volumes, len(wells))
        assert len(volumes) == len(wells), "Number of volumes must equal the number of wells"
        assert numpy.all(volumes >= 0), "Volumes must be positive or zero."
        if compositions is not None:
            assert len(compositions) == len(
                wells
            ), "Well compositions must be given for either all or none of the wells."
        else:
            compositions = [None] * len(wells)

        for well, volume, composition in zip(wells, volumes, compositions):
            idx = self.indices[well]
            v_original = self._volumes[idx]
            v_new = v_original + volume

            if v_new > self.max_volume:
                raise VolumeOverflowError(self.name, well, v_original, volume, self.max_volume, label)

            self._volumes[idx] = v_new

            if composition is not None and self._composition is not None:
                assert isinstance(composition, dict), "Well compositions must be given as dicts"
                # update the volumentric composition for this well
                original_composition = self.get_well_composition(well)
                new_composition = _combine_composition(v_original, original_composition, volume, composition)
                for k, f in new_composition.items():
                    if not k in self._composition:
                        # a new liquid is being added
                        self._composition[k] = numpy.zeros_like(self.volumes)
                    self._composition[k][idx] = f

        self.log(label)
        return

    def remove(
        self,
        wells: typing.Sequence[str],
        volumes: typing.Union[float, typing.Sequence[float], numpy.ndarray],
        label: typing.Optional[str] = None,
    ) -> None:
        """Removes volumes from wells.

        Parameters
        ----------
        wells : iterable of str
            Well ids
        volumes : float, iterable of float
            Scalar or iterable of volumes
        label : str
            Description of the operation
        """
        wells = numpy.array(wells).flatten("F")
        volumes = numpy.array(volumes).flatten("F")
        if len(volumes) == 1:
            volumes = numpy.repeat(volumes, len(wells))
        assert len(volumes) == len(wells), "Number of volumes must number of wells"
        assert numpy.all(volumes >= 0), "Volumes must be positive or zero."
        for well, volume in zip(wells, volumes):
            idx = self.indices[well]
            v_original = self._volumes[idx]
            v_new = v_original - volume

            if v_new < self.min_volume:
                raise VolumeUnderflowError(self.name, well, v_original, volume, self.min_volume, label)

            self._volumes[idx] -= volume
        self.log(label)
        return

    def log(self, label: typing.Optional[str]) -> None:
        """Logs the current volumes to the history.

        Parameters
        ----------
        label : str
            A label to insert in the history.
        """
        self._history.append(self.volumes)
        self._labels.append(label)
        return

    def condense_log(self, n: int, label: typing.Optional[str] = "last") -> None:
        """Condense the last n log entries.

        Parameters
        ----------
        n : int
            Number of log entries to condense
        label : str
            'first', 'last' or label of the condensed entry (default: label of the last entry in the condensate)
        """
        if label == "first":
            label = self._labels[len(self._labels) - n]
        if label == "last":
            label = self._labels[-1]
        state = self._history[-1]
        # cut away the history
        self._labels = self._labels[:-n]
        self._history = self._history[:-n]
        # append the last state
        self._labels.append(label)
        self._history.append(state)
        return

    def __repr__(self) -> None:
        return f"{self.name}\n{numpy.round(self.volumes, decimals=1)}"

    def __str__(self) -> None:
        return self.__repr__()


def _get_trough_component_names(
    name: str,
    columns: int,
    column_names: typing.Sequence[typing.Union[str, None]],
    initial_volumes: typing.Sequence[typing.Union[int, float]],
) -> typing.Dict[str, typing.Union[str, None]]:
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
    if numpy.shape(column_names) != (columns,):
        raise ValueError(f"The column names {column_names} don't match the number of columns ({columns}).")
    if numpy.shape(initial_volumes) != (columns,):
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


class Trough(Labware):
    """Special type of labware that can be accessed by many pipette tips in parallel."""

    def __init__(
        self,
        name: str,
        virtual_rows: int,
        columns: int,
        *,
        min_volume: float,
        max_volume: float,
        initial_volumes: typing.Union[float, numpy.ndarray] = 0,
        column_names: typing.Optional[typing.Sequence[typing.Union[str, None]]] = None,
    ) -> None:
        """Creates a `Labware` object.

        Parameters
        ----------
        name : str
            Label that the labware is identified by.
        virtual_rows : int, optional
            Number of tips that may access the trough in parallel.
            For example: A `Labware` with virtual rows can be accessed with 6 Tips,
            but has just one row in the `volumes` array.
        columns : int
            Number of columns in the labware
        min_volume : float
            Filling volume that must remain after an aspirate operation.
        max_volume : float
            Maximum volume that must not be exceeded after a dispense.
        initial_volumes : float, array-like, optional
            Initial filling volume of the wells (default: 0)
        column_names : array-like, optional
            A list/tuple of names for the column-wise contents of the troughs.
            If provided, these names are used for composition tracking.
        """
        # Convert lazily scalar-valued parameters to lists
        if column_names is None:
            column_names = [None] * columns
        if isinstance(column_names, str):
            column_names = [column_names]

        if isinstance(initial_volumes, (int, float)):
            initial_volumes = [initial_volumes] * columns

        # Determine component names with a different default pattern compared to Labware
        component_names = _get_trough_component_names(name, columns, column_names, initial_volumes)

        super().__init__(
            name=name,
            rows=1,
            columns=columns,
            min_volume=min_volume,
            max_volume=max_volume,
            initial_volumes=initial_volumes,
            virtual_rows=virtual_rows,
            component_names=component_names,
        )
