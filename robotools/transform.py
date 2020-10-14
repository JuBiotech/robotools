import numpy


def make_well_index_dict(R:int, C:int):
    """Create a dictionary mapping well IDs to their numpy indices.
    
    Args:
        R (int): number of rows
        C (int): number of columns

    Returns:
        indices (dict): mapping of IDs to numpy-style indices
    """
    return {
        f'{row}{column:02d}' : (r, c)
        for r, row in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:R])
        for c, column in enumerate(range(1, C+1))
    }


def make_well_array(R:int, C:int):
    """Create a numpy array of well IDs.
    
    Args:
        R (int): number of rows
        C (int): number of columns

    Returns:
        array (numpy.array): array of well IDs
    """
    return numpy.array([
        [f'{row}{column:02d}' for column in range(1, C+1)]
        for row in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:R]
    ])


class WellShifter(object):
    def __init__(self, shape_A:tuple, shape_B:tuple, shifted_A01:str):
        """Create a helper object for shifting wells around.

        Args:
            shape_A: (n_rows, n_cols) of the source labware
            shape_B: (n_rows, n_cols) of the destination labware
            shifted_A01 (str): well ID on B where the A01 from A ends up
        """
        self.shifted_A01 = shifted_A01
        self.shape_A = shape_A
        self.shape_B = shape_B
        self.indices_A = make_well_index_dict(*self.shape_A)
        self.indices_B = make_well_index_dict(*self.shape_B)
        self.wells_A = make_well_array(*self.shape_A)
        self.wells_B = make_well_array(*self.shape_B)
        self.dr, self.dc = self.indices_B[shifted_A01]

        if shape_A[0] + self.dr > shape_B[0]:
            raise ValueError(f'Invalid shift parameterization. Not enough rows in destination.')
        if shape_A[1] + self.dc > shape_B[1]:
            raise ValueError(f'Invalid shift parameterization. Not enough columns in destination.')
        return

    def shift(self, wells):
        """Apply the forward-transformation.

        Args:
            wells (array-like): list or array of well ids on A

        Returns:
            shifted (array-like): array of well ids on B (same shape)
        """
        wells = numpy.array(wells)
        wells_shape = wells.shape

        shifted = []
        for well in wells.flatten():
            r, c = self.indices_A[well]
            shifted.append(self.wells_B[r+self.dr, c+self.dc])
        return numpy.array(shifted).reshape(wells_shape)

    def unshift(self, wells):
        """Apply the reverse-transformation.

        Args:
            wells (array-like): list or array of well ids on B

        Returns:
            original (array-like): array of well ids on A (same shape)
        """
        wells = numpy.array(wells)
        wells_shape = wells.shape

        shifted = []
        for well in wells.flatten():
            r, c = self.indices_B[well]
            shifted.append(self.wells_A[r-self.dr, c-self.dc])
        return numpy.array(shifted).reshape(wells_shape)


class WellRotator(object):
    def __init__(self, original_shape:tuple):
        """Create a helper object for shifting wells around.

        Args:
            original_shape: (n_rows, n_cols) of all wells in the source labware
        """
        self.original_shape = original_shape
        self.rotated_shape = original_shape[::-1]
        self.original_indices = make_well_index_dict(*self.original_shape)
        self.rotated_indices = make_well_index_dict(*self.rotated_shape)
        self.original_wells = make_well_array(*self.original_shape)
        self.rotated_wells = make_well_array(*self.rotated_shape)
        return

    def rotate_ccw(self, wells):
        """Rotate the given wells counterclockwise.

        Args:
            wells (array-like): list or array of well ids

        Returns:
            rotated (array-like): array of well ids
        """
        wells = numpy.array(wells)
        wells_shape = wells.shape

        rotated = []
        for well in wells.flatten():
            r, c = self.original_indices[well]
            rotated.append(self.rotated_wells[self.original_shape[1] - c - 1, r])
        return numpy.array(rotated).reshape(wells_shape)

    def rotate_cw(self, wells):
        """Rotate the given wells clockwise.

        Args:
            wells (array-like): list or array of well ids

        Returns:
            rotated (array-like): array of well ids
        """
        wells = numpy.array(wells)
        wells_shape = wells.shape

        rotated = []
        for well in wells.flatten():
            r, c = self.original_indices[well]
            rotated.append(self.rotated_wells[c, self.original_shape[0] - r - 1])
        return numpy.array(rotated).reshape(wells_shape)

