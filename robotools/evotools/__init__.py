import collections
import enum
import logging
import math
import numpy
import os

from .. import liquidhandling


logger = logging.getLogger('evotools')


class Labwares(str, enum.Enum):
    SystemLiquid = 'Systemliquid'


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


def _prepate_aspirate_dispense_parameters(rack_label:str, position:int, volume:float,
        liquid_class:str='',
        tip:Tip=Tip.Any,
        rack_id:str='', tube_id:str='',
        rack_type:str='', forced_rack_type:str='',
        max_volume:int=None):
    """Validates and prepares aspirate/dispense parameters.

    Args:
        rack_label (str): user-defined labware name (max 32 characters)
        position (int): number of the well
        volume (float): volume in microliters (will be rounded to 2 decimal places)
        liquid_class (str): (optional) overwrites the liquid class for this step (max 32 characters)
        tip (Tip or int): (optional) tip that will be selected (Tip or 1-8)
        rack_id (str): (optional) barcode of the labware (max 32 characters)
        tube_id (str): (optional) barcode of the tube (max 32 characters)
        rack_type (str): (optional) configuration name of the labware (max 32 characters).
            An error is raised if it missmatches with the underlying worktable.
        forced_rack_type (str): (optional) overrides rack_type from worktable
        max_volume (int): (optional) maximum allowed volume
    """
    # required parameters
    if rack_label is None:
        raise ValueError('Missing required paramter: rack_label')
    if not isinstance(rack_label, str) or len(rack_label) > 32 or ';' in rack_label:
        raise ValueError(f'Invalid rack_label: {rack_label}')

    if position is None:
        raise ValueError('Missing required paramter: position')
    if not isinstance(position, int) or position < 0:
        raise ValueError(f'Invalid position: {position}')

    if volume is None:
        raise ValueError('Missing required paramter: volume')
    try:
        volume = float(volume)
    except:
        raise ValueError(f'Invalid volume: {volume}')
    if volume < 0 or volume > 7158278 or numpy.isnan(volume):
        raise ValueError(f'Invalid volume: {volume}')
    if max_volume is not None and volume > max_volume:
        raise InvalidOperationError(f'Volume of {volume} exceeds max_volume.')

    # optional parameters
    if not isinstance(liquid_class, str) or len(liquid_class) > 32 or ';' in liquid_class:
        raise ValueError(f'Invalid liquid_class: {liquid_class}')
    if isinstance(tip, int) and not isinstance(tip, Tip):
        if tip == 1:
            tip = Tip.T1
        elif tip == 2:
            tip = Tip.T2
        elif tip == 3:
            tip = Tip.T3
        elif tip == 4:
            tip = Tip.T4
        elif tip == 5:
            tip = Tip.T5
        elif tip == 6:
            tip = Tip.T6
        elif tip == 7:
            tip = Tip.T7
        elif tip == 8:
            tip = Tip.T8
    if not isinstance(tip, Tip):
        raise ValueError(f'Invalid tip: {tip}')
    if not isinstance(rack_id, str) or len(rack_id) > 32 or ';' in rack_id:
        raise ValueError(f'Invalid rack_id: {rack_id}')
    if not isinstance(rack_type, str) or len(rack_type) > 32 or ';' in rack_type:
        raise ValueError(f'Invalid rack_type: {rack_type}')
    if not isinstance(forced_rack_type, str) or len(forced_rack_type) > 32 or ';' in forced_rack_type:
        raise ValueError(f'Invalid forced_rack_type: {forced_rack_type}')

    # apply rounding and corrections for the right string formatting
    volume = f'{numpy.round(volume, decimals=2):.2f}'
    tip = '' if tip == -1 else tip
    return rack_label, position, volume, liquid_class, tip, rack_id, tube_id, rack_type, forced_rack_type


def _partition_volume(volume:float, *, max_volume:int):
    """Partitions a pipetting volume into zero or more integer-valued volumes that are <= max_volume.

    Args:
        volume (float): a volume to partition
        max_volume (int): maximum volume of a pipetting step

    Returns:
        volumes (list): partitioned volumes
    """
    if volume == 0:
        return []
    if volume < max_volume:
        return [volume]
    isteps = math.ceil(volume / max_volume)
    step_volume = math.ceil(volume / isteps)
    volumes = [step_volume] * (isteps - 1)
    volumes.append(volume - numpy.sum(volumes))
    return volumes


def _partition_by_column(sources, destinations, volumes):
    """Partitions sources/destinations/volumes by the source column.

    Returns:
        list of (sources, destinations, volumnes)
    """
    column_groups = collections.defaultdict(lambda: ([], [], []))
    for s, d, v in zip(sources, destinations, volumes):
        # group by source columns
        group = s[1:]
        column_groups[group][0].append(s)
        column_groups[group][1].append(d)
        column_groups[group][2].append(v)

    column_groups = [
        column_groups[col]
        for col in sorted(column_groups.keys())
    ]
    return column_groups


class Worklist(list):
    def __init__(self, filepath:str=None, max_volume:int=950, auto_split:bool=True):
        """Creates a worklist writer.

        Args:
            filepath (str): optional filename/filepath to write when the context is exited (must include a .gwl extension)
            max_volume (int): maximum aspiration volume in µl
            auto_split (bool): If True, large volumes in transfer operations are automatically splitted.
                If set to False, InvalidOperationError is raised when a pipetting volume exceeds max_volume.
        """
        self._filepath = filepath
        if max_volume is None:
            raise ValueError('The `max_volume` parameter is required.')
        self.max_volume = max_volume
        self.auto_split = auto_split
        return super().__init__()
    
    def __enter__(self):
        self.clear()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._filepath:
            self.save(self._filepath)
        return
    
    def save(self, filepath:str):
        """Writes the worklist to the filepath.

        Args:
            filepath (str): file name or path to write (must include a .gwl extension)
        """
        assert '.gwl' in filepath.lower(), 'The filename did not contain the .gwl extension.'
        if os.path.exists(filepath):
            os.remove(filepath)
        with open(filepath, 'w', newline='\r\n') as file:
            file.write('\n'.join(self))
        return
    
    def comment(self, comment:str):
        """Adds a comment.
        
        Args:
            comment (str): A single- or multi-line comment. Be nice and avoid special characters.
        """
        if not comment:
            return
        if ';' in comment:
            raise ValueError('Illegal semicolon in comment.')
        for cline in comment.split('\n'):
            cline = cline.strip()
            if cline:
                self.append(f'C;{cline}')
        return
    
    def wash(self, scheme:int=1):
        """Washes fixed tips or replaces DiTis.

        Washes/replaces the tip that was used by the preceding aspirate record(s).
        
        Args:
            scheme (int): number indicating the wash scheme (default: 1)
        """
        if not scheme in {1,2,3,4}:
            raise ValueError('scheme must be either 1, 2, 3 or 4')
        self.append(f'W{scheme};')
        return
    
    def decontaminate(self):
        """Decontamination wash consists of a decontamination wash followed by a normal wash."""
        self.append('WD;')
        return
    
    def flush(self):
        """Discards the contents of the tips WITHOUT WASHING or DROPPING of tips."""
        self.append('F;')
        return
    
    def commit(self):
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
        self.append('B;')
        return
    
    def set_diti(self, diti_index:int):
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
        
        Args:
            diti_index (int): type of DiTis to use in subsequent steps
        """
        if not (len(self) == 0 or self[-1][0] == 'B'):
            raise InvalidOperationError('DiTi type can only be switched at the beginning or after a Break/commit step. Read the docstring.')
        self.append(f'S;{diti_index}')
        return
    
    def aspirate_well(self, rack_label:str, position:int, volume:float, *,
            liquid_class:str='', tip:Tip=Tip.Any,
            rack_id:str='', tube_id:str='',
            rack_type:str='', forced_rack_type:str=''):
        """Command for aspirating with a single tip.

        Each Aspirate record specifies the aspiration parameters for a single tip (the next unused tip from the tip selection you have specified).

        Args:
            rack_label (str): user-defined labware name (max 32 characters)
            position (int): number of the well
            volume (float): volume in microliters (will be rounded to 2 decimal places)
            liquid_class (str): (optional) overwrites the liquid class for this step (max 32 characters)
            tip (Tip or int): (optional) tip that will be selected (Tip or 1-8)
            rack_id (str): (optional) barcode of the labware (max 32 characters)
            tube_id (str): (optional) barcode of the tube (max 32 characters)
            rack_type (str): (optional) configuration name of the labware (max 32 characters).
                An error is raised if it missmatches with the underlying worktable.
            forced_rack_type (str): (optional) overrides rack_type from worktable
        """
        args = (rack_label, position, volume, liquid_class, tip, rack_id, tube_id, rack_type, forced_rack_type)
        (rack_label, position, volume, liquid_class, tip, rack_id, tube_id, rack_type, forced_rack_type) = _prepate_aspirate_dispense_parameters(*args, max_volume=self.max_volume)
        tip_type = ''
        self.append(
            f'A;{rack_label};{rack_id};{rack_type};{position};{tube_id};{volume};{liquid_class};{tip_type};{tip};{forced_rack_type}'
        )
        return
    
    def dispense_well(self, rack_label:str, position:int, volume:float, *,
            liquid_class:str='', tip:Tip=Tip.Any,
            rack_id:str='', tube_id:str='',
            rack_type:str='', forced_rack_type:str=''):
        """Command for dispensing with a single tip.

        Each Dispense record specifies the dispense parameters for a single tip.
        It uses the same tip which was used by the preceding Aspirate record.
        
        Args:
            rack_label (str): user-defined labware name (max 32 characters)
            position (int): number of the well
            volume (float): volume in microliters (will be rounded to 2 decimal places)
            liquid_class (str): (optional) overwrites the liquid class for this step (max 32 characters)
            tip (Tip or int): (optional) tip that will be selected (Tip or 1-8)
            rack_id (str): (optional) barcode of the labware (max 32 characters)
            tube_id (str): (optional) barcode of the tube (max 32 characters)
            rack_type (str): (optional) configuration name of the labware (max 32 characters).
                An error is raised if it missmatches with the underlying worktable.
            forced_rack_type (str): (optional) overrides rack_type from worktable
        """
        args = (rack_label, position, volume, liquid_class, tip, rack_id, tube_id, rack_type, forced_rack_type)
        (rack_label, position, volume, liquid_class, tip, rack_id, tube_id, rack_type, forced_rack_type) = _prepate_aspirate_dispense_parameters(*args, max_volume=self.max_volume)
        tip_type = ''
        self.append(
            f'D;{rack_label};{rack_id};{rack_type};{position};{tube_id};{volume};{liquid_class};{tip_type};{tip};{forced_rack_type}'
        )
        return
        
    def _reagent_distribution(self):
        raise NotImplementedError()
    
    def aspirate(self, labware:liquidhandling.Labware, wells:list, volumes:float, *, label=None, **kwargs):
        """Performs aspiration from the provided labware.

        Args:
            labware (liquidhandling.Labware): source labware
            wells (str or iterable): list of well ids
            volumes (float or iterable): volume(s) to aspirate
            label (str): label of the operation to log into labware history
            kwargs: additional keyword arguments to pass to `aspirate_well`
        """
        wells = numpy.array(wells).flatten('F')
        volumes = numpy.array(volumes).flatten('F')
        if len(volumes) == 1:
            volumes = numpy.repeat(volumes, len(wells))
        labware.remove(wells, volumes, label)
        self.comment(label)
        for well, volume in zip(wells, volumes):
            if volume > 0:
                self.aspirate_well(labware.name, labware.positions[well], volume, **kwargs)
        return

    def dispense(self, labware:liquidhandling.Labware, wells:list, volumes:float, *, label=None, **kwargs):
        """Performs dispensing from the provided labware.

        Args:
            labware (liquidhandling.Labware): source labware
            wells (str or iterable): list of well ids
            volumes (float or iterable): volume(s) to dispense
            label (str): label of the operation to log into labware history
            kwargs: additional keyword arguments to pass to `dispense_well`
        """
        wells = numpy.array(wells).flatten('F')
        volumes = numpy.array(volumes).flatten('F')
        if len(volumes) == 1:
            volumes = numpy.repeat(volumes, len(wells))
        labware.add(wells, volumes, label)
        self.comment(label)
        for well, volume in zip(wells, volumes):
            if volume > 0:
                self.dispense_well(labware.name, labware.positions[well], volume, **kwargs)
        return
    
    def transfer(self, source, source_wells, destination, destination_wells, volumes, *, label=None, wash_scheme=1, **kwargs):
        """Transfer operation between two labwares.

        Args:
            source (liquidhandling.Labware): source labware
            source_wells (str or iterable): list of source well ids
            destination (liquidhandling.Labware): destination labware
            destination_wells (str or iterable): list of destination well ids
            volumes (float or iterable): volume(s) to transfer
            label (str): label of the operation to log into labware history
            wash_scheme (int): wash scheme to apply after every every
            kwargs: additional keyword arguments to pass to aspirate and dispense
        """
        # reformat the convenience parameters
        source_wells = numpy.array(source_wells).flatten('F')
        destination_wells = numpy.array(destination_wells).flatten('F')
        volumes = numpy.array(volumes).flatten('F')
        nmax = max((len(source_wells), len(destination_wells), len(volumes)))
        
        if len(source_wells) == 1:
            source_wells = numpy.repeat(source_wells, nmax)
        if len(destination_wells) == 1:
            destination_wells = numpy.repeat(destination_wells, nmax)
        if len(volumes) == 1:
            volumes = numpy.repeat(volumes, nmax)
        lengths = (len(source_wells), len(destination_wells), len(volumes))
        assert len(set(lengths)) == 1, f'Number of source/destination/volumes must be equal. They were {lengths}'
        
        # the label applies to the entire transfer operation and is not logged at individual aspirate/dispense steps
        self.comment(label)
        nsteps = 0

        for srcs, dsts, vols in _partition_by_column(source_wells, destination_wells, volumes):
            # make vector of volumes into vector of volume-lists
            vols = [
                _partition_volume(float(v), max_volume=self.max_volume) if self.auto_split else [v]
                for v in vols
            ]
            # transfer from this source column until all wells are done
            npartitions = max(map(len, vols))
            for p in range(npartitions):
                naccessed = 0
                # iterate the rows
                for s, d, vs in zip(srcs, dsts, vols):
                    # transfer the next volume-fraction for this well
                    if len(vs) > p:
                        v = vs[p]
                        if v > 0:
                            self.aspirate(source, s, v, label=None, **kwargs)
                            self.dispense(destination, d, v, label=None, **kwargs)
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

        # condense the labware logs into one operation
        # this is done after creating the worklist to facilitate debugging
        source.condense_log(nsteps, label=label)
        destination.condense_log(nsteps, label=label)
        return
        
    def __repr__(self):
        return '\n'.join(self)
    
    def __str__(self):
        return self.__repr__()
