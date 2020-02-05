"""Module with robot-agnostic utilities."""
import collections
import numpy
from . import evotools
from . import liquidhandling


def get_trough_wells(n: int, trough_wells: list) -> list:
    """Creates a list that re-uses trough wells if needed.

    When n > trough.virtual_rows, the available wells are repeated.
    
    Args:
        n (int): number of trough wells to work with
        trough_wells (list): trough well IDs that may be used

    Returns:
        wells (list): n virtual wells in the trough
    """
    if not isinstance(n, int):
        raise TypeError('n must be int')
    if not isinstance(trough_wells, (list, tuple, numpy.ndarray)):
        raise TypeError('trough_wells must be a tuple, list or 1-D numpy array.')
    if n < 0:
        raise ValueError('n must be ≥ 0.')
    if len(trough_wells) == 0:
        raise ValueError('trough_wells must contain at least 1 element.')

    trough_wells = list(numpy.array(trough_wells).flatten('F'))
    n_available = len(trough_wells)
    n_repeat = n // n_available + 1
    return (trough_wells * n_repeat)[:n]


class DilutionPlan:
    """Represents the result of a dilution series planning."""
    def __init__(self, *, xmin:float, xmax:float, R:int, C:int, stock:float, mode:str, vmax:float, min_transfer:float):
        """Plans a regularly-spaced dilution series with in very few steps.
    
        Args:
            xmin (float): lowest concentration value in the result
            xmax (float): highest concentration in the result
            R (int): number of rows in the MTP
            C (int): number of colums in the MTP
            stock (float): stock concentration (must be >= xmax)
            mode (str): either 'log' or 'linear'
            vmax (float): scalar or vector-valued (C,) maximum volume [µL] in the dilution series
            min_transfer (float): minimum allowed volume [µL] for transfer steps
        """
        # process arguments
        if stock < xmax:
            raise ValueError(f'Stock concentration ({stock}) must be >= xmax ({xmax})')
        N = R * C
        
        vmax = numpy.atleast_1d(vmax)
        if len(vmax) == 1:
            vmax = numpy.repeat(vmax, C)
        if not len(vmax) == C:
            raise ValueError('The `vmax` argument must be scalar or of length `C`.')
        
        # determine target concentrations
        if mode == 'log':
            ideal_targets = numpy.exp(numpy.linspace(numpy.log(xmax), numpy.log(xmin), N))
        elif mode == 'linear':
            ideal_targets = numpy.linspace(xmax, xmin, N)
        else:
            raise ValueError('mode must be either "log" or "linear".')
        
        ideal_targets = ideal_targets.reshape((R, C), order='F')
    
        # collect preparation instructions for each columns
        # (column, dilution steps, prepared from, transfer volumes)
        instructions = []
        actual_targets = []
    
        # transfer from stock until the volume is too low
        for c in range(C):
            vtransfer = numpy.round(vmax[c] * ideal_targets[:,c] / stock, 0)
            if all(vtransfer >= min_transfer):
                instructions.append(
                    (c, 0, 'stock', vtransfer)
                )
                # compute the actually achieved target concentration
                actual_targets.append(vtransfer / vmax[c] * stock)
            else:
                break
        
        # prepare remaining columns by diluting existing ones
        for c in range(len(instructions), C):
            # find the first source column that can be used (with sufficient transfer volume)
            for src_c in range(0, len(instructions)):
                _, src_df, _, _ = instructions[src_c]
                vtransfer = numpy.ceil(vmax[c] * ideal_targets[:,c] / actual_targets[src_c])
                # take the leftmost column (least dilution steps) where the minimal transfer volume is exceeded
                if all(vtransfer >= min_transfer):
                    instructions.append(
                        # increment the dilution step counter
                        (c, src_df+1, src_c, vtransfer)
                    )
                    # compute the actually achieved target concentration
                    actual_targets.append(vtransfer * actual_targets[src_c] / vmax[c])
                    break
            
        if len(actual_targets) < C:
            message = f'Impossible with current settings.' \
                f' Only {len(instructions)}/{C} colums can be prepared.'
            if mode == 'linear':
                message += ' Try switching to "log" mode.'
            raise ValueError(message)
    
        self.R = R
        self.C = C
        self.N = R * C
        self.ideal_x = ideal_targets
        self.x = numpy.array(actual_targets).T
        self.xmin = numpy.min(actual_targets)
        self.xmax = numpy.max(actual_targets)
        self.instructions = instructions
        self.vmax = vmax
        self.v_stock = numpy.sum([
            v
            for _, dsteps, src, v in instructions
            if dsteps == 0
        ])
        self.v_diluent = numpy.sum(R * vmax) - self.v_stock
        self.max_steps = max([
            dsteps
            for _, dsteps, _, _ in instructions
        ])

    def __repr__(self):
        output = f'Serial dilution plan ({self.xmin:.5f} to {self.xmax:.2f})' \
            f' from at least {self.v_stock} µL stock and {self.v_diluent} µL diluent:'
        for c, dsteps, src, vtransfer in self.instructions:
            output += f'\r\n\tPrepare column {c} with {vtransfer} µL from '
            if dsteps == 0:
                output += 'stock'
            else:
                output += f'column {src}'
            output += f' and fill up to {self.vmax[c]} µL'
            if dsteps > 0:
                output += f' ({dsteps} serial dilutions)'
        return output

    def to_worklist(
            self, *,
            worklist: evotools.Worklist,
            stock: liquidhandling.Labware, stock_column: int=0,
            diluent: liquidhandling.Labware, diluent_column: int=0,
            dilution_plate: liquidhandling.Labware,
            destination_plate: liquidhandling.Labware=None,
            v_destination: float=None,
            mix_threshold: float=0.05,
            lc_stock_trough: str='Trough_Water_FD_AspLLT',
            lc_diluent_trough: str='Trough_Water_FD_AspLLT',
            lc_mix: str='Water_FD_AspZmax-1_Mix',
            lc_transfer: str='Water_FD_AspZmax-1',
        ):
        """Writes the `DilutionPlan` to a `Worklist`.

        The stock is assumed to be non-sedimenting (e.g. by stirring), but all aspirations from freshly
        diluted wells are done right away.
        Mixing is done after dilution and before transfer whenever the diluted volume is more than
        `mix_threshold * self.vmax`. The volume aspirated for mixing is 80% of `self.vmax` but
        maxes out at the `Worklist.max_volume`.

        Stock and diluent troughs may have less rows than the dilution plate.

        Args:
            wl (Worklist): a Worklist that will be appended
            stock (Labware): a trough containing the highly concentrated stock solution
            stock_column (int): 0-based column number of the stock solution in the `stock` labware
            diluent (Labware): a trough containing the diluent for the dilution series
            diluent_column (int): 0-based column number of the diluent solution in the `stock` labware        
            dilution_plate (Labware): an (empty) labware to use for the dilution series (begins in top left corner)
            destination_plate (Labware, optional): an (empty) labware to transfer to
            v_destination (float): volume [µl] to transfer to the `destination_plate` (if set)
            mix_threshold (float): maximum fraction of total dilution volume (self.vmax) that may be diluted without subsequent mixing (defaults to 0.05 or 5%)
            lc_stock_trough (str): liquid class to use for transfer of stock solution to the dilution plate
            lc_diluent_trough (str): liquid class to use for transfer of diluent to dilution plate
            lc_mix (str): liquid class for mixing steps
            lc_transfer (str): liquid class for transfers within the `dilution_plate` and to the `destination_plate`
        """
        if dilution_plate.n_rows < self.R:
            raise ValueError(f'Dilution plate "{dilution_plate.name}" has not enough rows for this dilution plan.')
        if dilution_plate.n_columns < self.C:
            raise ValueError(f'Dilution plate "{dilution_plate.name}" has not enough columns for this dilution plan.')
        if destination_plate and destination_plate.n_rows < self.R:
            raise ValueError(f'Destination plate "{destination_plate.name}" has not enough rows for this dilution plan.')
        if destination_plate and destination_plate.n_columns < self.C:
            raise ValueError(f'Destination plate "{destination_plate.name}" has not enough columns for this dilution plan.')
        if not stock.is_trough:
            raise ValueError(f'The stock labware "{stock.name}" must be a trough.')
        if not diluent.is_trough:
            raise ValueError(f'The diluent labware "{diluent.name}" must be a trough.')

        stock_wells = stock.wells[:, stock_column]
        diluent_wells = diluent.wells[:, diluent_column]

        # beforehand, we need to know which other columns have to be prepared from
        # a given column. This way, we can transfer to them right after diluting/mixing.
        serial_dilution_from_to = collections.defaultdict(list)
        for col, _, src, v in self.instructions:
            if src != 'stock':
                serial_dilution_from_to[src].append((col, v))

        # now prepare column by column
        for col, _, src, v_src in self.instructions:
            # this is the first transfer in the entire procedure
            if src == 'stock':
                worklist.transfer(
                    stock, get_trough_wells(self.R, stock_wells),
                    dilution_plate, dilution_plate.wells[:self.R, col],
                    volumes=v_src,
                    liquid_class=lc_stock_trough,
                    label=f'Distribute from stock'
                )
                worklist.commit()
            else:
                # transfers for serial dilution are done after the mixing step
                # at this point this has already happened
                if not numpy.allclose(dilution_plate.volumes[:self.R, col], v_src):
                    raise Exception(f'Column {col} volume not as expected.')

            # the column already contains the higher-concentrated standards
            # now it's time to dilute it
            worklist.transfer(
                diluent, get_trough_wells(self.R, diluent_wells),
                dilution_plate, dilution_plate.wells[:self.R, col],
                volumes=self.vmax[col] - v_src,
                liquid_class=lc_diluent_trough,
                label=f'Dilute column {col}'
            )
            worklist.commit()

            # mixing time!
            if numpy.any(v_src > mix_threshold * self.vmax[col]):
                worklist.transfer(
                    dilution_plate, dilution_plate.wells[:self.R, col],
                    dilution_plate, dilution_plate.wells[:self.R, col],
                    volumes=min(worklist.max_volume, self.vmax[col] * 0.8),
                    liquid_class=lc_mix,
                    label=f'Mix column {col} with 80% of its volume'
                )
                worklist.commit()

            # transfer to other columns that will be prepared from this one
            for dst, v_dst in serial_dilution_from_to[col]:
                worklist.transfer(
                    dilution_plate, dilution_plate.wells[:self.R, col],
                    dilution_plate, dilution_plate.wells[:self.R, dst],
                    volumes=v_dst,
                    liquid_class=lc_transfer,
                    label=f'Transfer columns {col} -> {dst} for later dilution step'
                )
                worklist.commit()

            # transfer to a destination is optional
            if destination_plate:
                worklist.transfer(
                    dilution_plate, dilution_plate.wells[:self.R, col],
                    destination_plate, destination_plate.wells[:self.R, col],
                    volumes=v_destination,
                    liquid_class=lc_transfer,
                    label=f'Transfer column {col} to the destination plate'
                )
                worklist.commit()

        return
