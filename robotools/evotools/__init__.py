import numpy
import logging
import os

from .. import liquidhandling


logger = logging.getLogger('evotools')


class Worklist(list):
    def __init__(self):
        self._lines = None
        return super().__init__()
    
    def __enter__(self):
        self._lines = []
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return
    
    def save(self, filepath):
        """Writes the worklist to the filename.

        Args:
            filepath (str): file name or path to write (must include a .gwl extension)
        """
        assert '.gwl' in filepath.lower(), 'The filename did not contain the .gwl extension.'
        if os.path.exists(filepath):
            os.remove(filepath)
        with open(filepath, 'w') as file:
            file.writelines(self.lines)
        return
    
    def comment(self, comment):
        """Adds a comment line. Multiline is supported"""
        for cline in comment.split('\n'):
            self.append(f'C;{comment.strip()}')
        return
    
    def wash(self, scheme=1):
        """Washes fixed tips or replaces DiTis.


        
        Args:
            scheme (int): number indicating the wash scheme (default: 1)
        """
        assert scheme in {1,2,3,4}, 'scheme must be either 1, 2, 3 or 4'
        self.append(f'W{scheme};')
        return
    
    def decontaminate(self):
        """Decontamination wash consists of a decontamination wash followed by a normal wash."""
        self.append('WD;')
        return
    
    def flush(self):
        self.append('F;')
        return
    
    def force(self):
        """Inserts a 'break' that forces the execution of aspirate/dispense operations at this point.
        
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
    
    def set_diti(self, diti_index):
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
        assert len(self._lines) == 0 or self._lines[-1][0] == 'B', 'DiTi type can only be switched at the beginning or after a Break/force step. Read the docstring.'
        self.append(f'S;{diti_index}')
        return
    
    def _aspirate(self, rack_label, 
                  position, volume,
                  liquid_class,
                  tip_type, tip_mask,
                  rack_id='', tube_id='',
                  rack_type='', forced_rack_type=False):
        """Command for aspirating with a single tip.

        Each Aspirate record specifies the aspiration parameters for a single tip (the next unused tip from the tip selection you have specified).

        Args:
            rack_label (str): name of the source labware
            position (int): number of the source well
            rack_id (str): barcode of the source labware
            tube_id (str): barcode of the source tube
            rack_type (str):
        """
        self._lines.append(
            f'A;{rack_label};{rack_id};{rack_type};{position};{tube_id};volume;{liquid_class};{tip_type};{tip_mask};{forced_rack_type}'
        )
        return
    
    def _dispense(self):
        raise NotImplementedError()
        
    def _reagent_distribution(self):
        raise NotImplementedError()
    
    def aspirate(self, labware, volumes):
        return
    
    def transfer(self, source, source_wells, destination, destination_wells, volumes, label=None):
        # reformat the convenience parameters
        source_wells = numpy.array(source_wells).flatten()
        destination_wells = numpy.array(destination_wells).flatten()
        n_source = len(source_wells)
        n_destination = len(destination_wells)
        if numpy.isscalar(volumes):
            volumes = numpy.repeat(volumes, max(n_source, n_destination))
        n_vol = len(volumes)
        
        if not n_source == n_destination:
            assert n_source == 1 or n_destination == 1, 'Number of source & destination wells must be equal or 1'
            
        if n_source == 1:
            source.remove(source_wells, numpy.sum(volumes), label=label)
        else:
            source.remove(source_wells, volumes, label=label)
        
        if n_destination == 1:
            destination.add(destination_wells, numpy.sum(volumes), label=label)
        else:
            destination.add(destination_wells, volumes, label=label)

        return
        