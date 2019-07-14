import numpy
import logging

logger = logging.getLogger('liquidhandling')


class VolumeOverflowError(Exception):
    pass


class VolumeUnderflowError(Exception):
    pass


def _combine_composition(original_volume:float, original_composition:dict, added_volume:float, added_composition:dict):
    """Computes the composition of a liquid, created by the mixing of two liquids (A and B).

    Args:
        volume_A (float): volume of liquid A
        composition_A (dict): relative composition of liquid A
        volume_B (float): volume of liquid B
        composition_B (dict): relative composition of liquid B

    Returns:
        composition (dict): composition of the new liquid created by mixing the given volumes of A and B
    """
    # convert to volumetric fractions
    volumetric_fractions = {
        k : f * original_volume
        for k, f in original_composition.items()
    }
    # volumetrically add incoming fractions
    for k, f in added_composition.items():
        if not k in volumetric_fractions:
            volumetric_fractions[k] = 0
        volumetric_fractions[k] += f * added_volume
    # convert back to relative fractions
    new_composition = {
        k : v / (original_volume + added_volume)
        for k, v in volumetric_fractions.items()
    }
    return new_composition


class Labware(object):
    @property
    def history(self):
        """List of label/volumes history."""
        return list(zip(self._labels, self._history))

    @property
    def report(self):
        """A printable report of the labware history."""
        report = self.name
        for label, state in self.history:
            if label:
                report += f'\n{label}'
            report += f'\n{numpy.round(state, decimals=1)}'
            report += '\n'
        return report
        
    @property
    def volumes(self):
        """Current volumes in the labware."""
        return self._volumes.copy()
    
    @property
    def wells(self) -> numpy.ndarray:
        """Array of well ids."""
        return self._wells
    
    @property
    def indices(self) -> dict:
        """Mapping of well-ids to numpy indices."""
        return self._indices
    
    @property
    def positions(self) -> dict:
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
    def composition(self) -> dict:
        """Relative composition of the liquids.
        
        This dictionary maps liquid names (keys) to arrays of relative amounts in each well.
        """
        return self._composition
    
    def __init__(self, name:str, rows:int, columns:int, *, min_volume:float, max_volume:float, initial_volumes:float=None, virtual_rows:int=None):
        # sanity checking
        if not isinstance(rows, int) or rows < 1:
            raise ValueError(f'Invalid rows: {rows}')
        if not isinstance(columns, int) or columns < 1:
            raise ValueError(f'Invalid columns: {columns}')
        if min_volume is None or min_volume < 0:
            raise ValueError(f'Invalid min_volume: {min_volume}')
        if max_volume is None or max_volume <= min_volume:
            raise ValueError(f'Invalid max_volume: {max_volume}')
        if virtual_rows is not None and rows != 1:
            raise ValueError('When using virtual_rows, the number of rows must be == 1')
        if virtual_rows is not None and virtual_rows < 1:
            raise ValueError(f'Invalid virtual_rows: {virtual_rows}')
                
        # explode convenience parameters
        if initial_volumes is None:
            initial_volumes = 0
        initial_volumes = numpy.array(initial_volumes)
        if initial_volumes.shape == ():
            initial_volumes = numpy.full((rows, columns), initial_volumes)
        else:
            initial_volumes = initial_volumes.reshape((rows, columns))
        assert initial_volumes.shape == (rows, columns), f'Invalid shape of initial_volumes: {initial_volumes.shape}'
        if numpy.any(initial_volumes < 0):
            raise ValueError('initial_volume cannot be negative')
        if numpy.any(initial_volumes > max_volume):
            raise ValueError('initial_volume cannot be above max_volume')
        
        # initialize properties
        self.name = name
        self.row_ids = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:rows if not virtual_rows else virtual_rows])
        self.column_ids = list(range(1, columns + 1))
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.virtual_rows = virtual_rows

        # generate arrays/mappings of well ids
        self._wells = numpy.array([
            [f'{row}{column:02d}' for column in self.column_ids]
            for row in self.row_ids
        ])
        if virtual_rows is None:
            self._indices = {
                f'{row}{column:02d}' : (r, c)
                for r, row in enumerate(self.row_ids)
                for c, column in enumerate(self.column_ids)
            }
            self._positions = {
                f'{row}{column:02d}' : 1 + c * rows + r
                for r, row in enumerate(self.row_ids)
                for c, column in enumerate(self.column_ids)
            }
        else:
            self._indices = {
                f'{vrow}{column:02d}' : (0, c)
                for vr, vrow in enumerate(self.row_ids)
                for c, column in enumerate(self.column_ids)
            }
            self._positions = {
                f'{vrow}{column:02d}' : 1 + c * virtual_rows + vr
                for vr, vrow in enumerate(self.row_ids)
                for c, column in enumerate(self.column_ids)
            }
        
        # initialize state variables
        self._volumes = initial_volumes.copy().astype(float)
        self._history = [self.volumes]
        self._labels = ['initial']
        self._composition = {
            self.name: numpy.ones_like(self.volumes)
        } if numpy.any(initial_volumes > 0) else {}
        return
    
    def get_well_composition(self, well:str) -> dict:
        """Retrieves the relative composition of a well.
        
        Keys: liquid names
        Values: relative amount
        """
        idx = self.indices[well]
        well_comp = {
            k : f[idx]
            for k, f in self.composition.items()
        }
        return well_comp

    def add(self, wells, volumes:float, label:str=None, compositions:list=None):
        """Adds volumes to wells.

        Args:
            wells: iterable of well ids
            volumes (int or float): scalar or iterable of volumes
            label (str): description of the operation
            compositions (iterable): list of composition dictionaries ({ name : relative amount })
        """
        wells = numpy.array(wells).flatten('F')
        volumes = numpy.array(volumes).flatten('F')
        if len(volumes) == 1:
            volumes = numpy.repeat(volumes, len(wells))
        assert len(volumes) == len(wells), 'Number of volumes must equal the number of wells'
        assert numpy.all(volumes >= 0), 'Volumes must be positive or zero.'
        if compositions is not None:
            assert len(compositions) == len(wells), 'Well compositions must be given for either all or none of the wells.'
        else:
            compositions = [None] * len(wells)

        for well, volume, composition in zip(wells, volumes, compositions):
            idx = self.indices[well]
            v_original = self._volumes[idx]
            v_new = v_original + volume

            if v_new > self.max_volume:
                raise VolumeOverflowError(f'Step "{label}": {self.name}.{well} has exceeded the maximum volume')

            self._volumes[idx] = v_new

            if composition is not None:
                assert isinstance(composition, dict), 'Well compositions must be given as dicts'
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
    
    def remove(self, wells, volumes:float, label=None):
        """Removes volumes from wells.

        Args:
            wells: iterable of well ids
            volumes (int or float): scalar or iterable of volumes
            label (str): description of the operation
        """
        wells = numpy.array(wells).flatten('F')
        volumes = numpy.array(volumes).flatten('F')
        if len(volumes) == 1:
            volumes = numpy.repeat(volumes, len(wells))
        assert len(volumes) == len(wells), 'Number of volumes must number of wells'
        assert numpy.all(volumes >= 0), 'Volumes must be positive or zero.'
        for well, volume in zip(wells, volumes):
            self._volumes[self.indices[well]] -= volume
            if self._volumes[self.indices[well]] < self.min_volume:
                raise VolumeUnderflowError(f'Step "{label}": {self.name}.{well} has undershot the minimum volume')
        self.log(label)
        return
    
    def log(self, label):
        """Logs the current volumes to the history."""
        self._history.append(self.volumes)
        self._labels.append(label)
        return

    def condense_log(self, n:int, label='last'):
        """Condense the last n log entries.

        Args:
            n (int): number of log entries to condense
            label (str): 'first', 'last' or label of the condensed entry (default: label of the last entry in the condensate)
        """
        if label == 'first':
            label = self._labels[len(self._labels)-n]
        if label == 'last':
            label = self._labels[-1]
        state = self._history[-1]
        # cut away the history
        self._labels = self._labels[:-n]
        self._history = self._history[:-n]
        # append the last state
        self._labels.append(label)
        self._history.append(state)
        return
        
    def __repr__(self):
        return f'{self.name}\n{numpy.round(self.volumes, decimals=1)}'
    
    def __str__(self):
        return self.__repr__()
