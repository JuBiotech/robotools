import numpy
import logging

logger = logging.getLogger('liquidhandling')


class VolumeOverflowError(Exception):
    pass


class VolumeUnderflowError(Exception):
    pass


class Labware(object):
    @property
    def history(self):
        return list(zip(self._labels, self._history))
        
    @property
    def current(self):
        return self._current.copy()
    
    @property
    def wells(self):
        return self._wells
    
    @property
    def indices(self):
        return self._indices
    
    @property
    def numbers(self):
        return self._numbers
    
    def __init__(self, name, rows, columns, min_volume, max_volume, initial_volumes=None):
        # explode convenience parameters
        if not initial_volumes:
            initial_volumes = 0
        if numpy.isscalar(initial_volumes):
            initial_volumes = numpy.full((rows, columns), initial_volumes)
        assert initial_volumes.shape == (rows, columns)
        
        # initialize properties
        self.name = name
        self.row_ids = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:rows]
        self.column_ids = list(range(1, columns + 1))
        self.min_volume = min_volume
        self.max_volume = max_volume
        
        # generate arrays/mappings of well ids
        self._wells = numpy.array([
            [f'{row}{column:02d}' for column in self.column_ids]
            for row in self.row_ids
        ])
        self._indices = {
            f'{row}{column:02d}' : (r, c)
            for r, row in enumerate(self.row_ids)
            for c, column in enumerate(self.column_ids)
        }
        self._numbers = {
            f'{row}{column:02d}' : 1 + c * rows + r
            for r, row in enumerate(self.row_ids)
            for c, column in enumerate(self.column_ids)
        }
        
        # initialize state variables
        self._current = initial_volumes.copy()
        self._history = [self.current]
        self._labels = ['initial']
        return
    
    def add(self, wells, volumes, label=None):
        wells = numpy.array(wells).flatten()
        if numpy.isscalar(volumes):
            volumes = numpy.repeat(volumes, len(wells))
        assert len(volumes) == len(wells), 'Number of volumes must number of wells'
        assert numpy.all(volumes >= 0), 'Volumes must be positive or zero.'
        for well, volume in zip(wells, volumes):
            self._current[self.indices[well]] += volume
            if self._current[self.indices[well]] > self.max_volume:
                raise VolumeOverflowError(f'Step "{label}": {well} has exceeded the maximum volume')
        self.log(label)
        return
    
    def remove(self, wells, volumes, label=None):
        wells = numpy.array(wells).flatten()
        if not hasattr(volumes, '__iter__'):
            volumes = numpy.repeat(volumes, len(wells))
        assert len(volumes) == len(wells), 'Number of volumes must number of wells'
        assert numpy.all(volumes >= 0), 'Volumes must be positive or zero.'
        for well, volume in zip(wells, volumes):
            self._current[self.indices[well]] -= volume
            if self._current[self.indices[well]] < self.min_volume:
                raise VolumeUnderflowError(f'Step "{label}": {well} has undershot the minimum volume')
        self.log(label)
        return
    
    def log(self, label):
        self._history.append(self.current)
        self._labels.append(label)
        return
    
    def report(self):
        report = self.name
        for label, state in self.history:
            if label:
                report += f'\n{label}'
            report += f'\n{numpy.round(state, decimals=1)}'
            report += '\n'
        return report
    
    def __repr__(self):
        return f'{self.name}\n{numpy.round(self.current, decimals=1)}'
    
    def __str__(self):
        return self.__repr__()
    
    
