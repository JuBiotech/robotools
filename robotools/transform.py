from typing import Dict, Literal, Tuple

import numpy
from numpy.typing import ArrayLike


def make_well_index_dict(R: int, C: int) -> Dict[str, Tuple[int, int]]:
    """Create a dictionary mapping well IDs to their numpy indices.

    Parameters
    ----------
    R : int
        Number of rows
    C : int
        Number of columns

    Returns
    -------
    indices : dict
        Mapping of IDs to numpy-style indices
    """
    return {
        f"{row}{column:02d}": (r, c)
        for r, row in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:R])
        for c, column in enumerate(range(1, C + 1))
    }


def make_well_array(R: int, C: int) -> numpy.ndarray:
    """Create a numpy array of well IDs.

    Parameters
    ----------
    R : int
        Number of rows
    C : int
        Number of columns

    Returns
    -------
    array : ndarray
        Array of well IDs
    """
    return numpy.array(
        [[f"{row}{column:02d}" for column in range(1, C + 1)] for row in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:R]]
    )


class WellShifter:
    """Helper object to shift a set of well IDs within a MTP."""

    def __init__(self, shape_A: Tuple[int, int], shape_B: Tuple[int, int], shifted_A01: str) -> None:
        """Create a helper object for shifting wells around.

        Parameters
        ----------
        shape_A : tuple
            (n_rows, n_cols) of the source labware
        shape_B : tuple
            (n_rows, n_cols) of the destination labware
        shifted_A01 : str
            Well ID on B where the A01 from A ends up
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
            raise ValueError(f"Invalid shift parameterization. Not enough rows in destination.")
        if shape_A[1] + self.dc > shape_B[1]:
            raise ValueError(f"Invalid shift parameterization. Not enough columns in destination.")

    def shift(self, wells: ArrayLike) -> numpy.ndarray:
        """Apply the forward-transformation.

        Parameters
        ----------
        wells : array-like
            List or array of well ids on A

        Returns
        -------
        shifted : ndarray
            Array of well ids on B (same shape)
        """
        wells = numpy.array(wells)
        wells_shape = wells.shape

        shifted = []
        for well in wells.flatten():
            r, c = self.indices_A[well]
            shifted.append(self.wells_B[r + self.dr, c + self.dc])
        return numpy.array(shifted).reshape(wells_shape)

    def unshift(self, wells: ArrayLike) -> numpy.ndarray:
        """Apply the reverse-transformation.

        Parameters
        ----------
        wells : array-like
            List or array of well ids on B

        Returns
        -------
        original : ndarray
            Array of well ids on A (same shape)
        """
        wells = numpy.array(wells)
        wells_shape = wells.shape

        shifted = []
        for well in wells.flatten():
            r, c = self.indices_B[well]
            shifted.append(self.wells_A[r - self.dr, c - self.dc])
        return numpy.array(shifted).reshape(wells_shape)


class WellRotator:
    """Helper object to rotate a set of well IDs within a MTP."""

    def __init__(self, original_shape: Tuple[int, int]) -> None:
        """Create a helper object for shifting wells around.

        Parameters
        ----------
        original_shape : tuple
            (n_rows, n_cols) of all wells in the source labware
        """
        self.original_shape = original_shape
        self.rotated_shape = original_shape[::-1]
        self.original_indices = make_well_index_dict(*self.original_shape)
        self.rotated_indices = make_well_index_dict(*self.rotated_shape)
        self.original_wells = make_well_array(*self.original_shape)
        self.rotated_wells = make_well_array(*self.rotated_shape)
        super().__init__()

    def rotate_ccw(self, wells: ArrayLike) -> numpy.ndarray:
        """Rotate the given wells counterclockwise.

        Parameters
        ----------
        wells : array-like
            List or array of well ids

        Returns
        -------
        rotated : ndarray
            Array of well ids
        """
        wells = numpy.array(wells)
        wells_shape = wells.shape

        rotated = []
        for well in wells.flatten():
            r, c = self.original_indices[well]
            rotated.append(self.rotated_wells[self.original_shape[1] - c - 1, r])
        return numpy.array(rotated).reshape(wells_shape)

    def rotate_cw(self, wells: ArrayLike) -> numpy.ndarray:
        """Rotate the given wells clockwise.

        Parameters
        ----------
        wells : array-like
            List or array of well ids

        Returns
        -------
        rotated : ndarray
            Array of well ids
        """
        wells = numpy.array(wells)
        wells_shape = wells.shape

        rotated = []
        for well in wells.flatten():
            r, c = self.original_indices[well]
            rotated.append(self.rotated_wells[c, self.original_shape[0] - r - 1])
        return numpy.array(rotated).reshape(wells_shape)


class WellRandomizer:
    """Helper object to randomize a set of well IDs within a MTP."""

    def __init__(
        self,
        original_shape: Tuple[int, int],
        random_seed: int,
        *,
        mode: Literal["full", "row", "column"] = "full",
    ) -> None:
        """Create a helper object for randomizing wells.

        Parameters
        ----------
        original_shape
            (n_rows, n_cols) of all wells in the source labware
        random_seed
            Integer for defined and reproduceable randomization
        mode
            To switch between `"full"` randomization,
            or randomization only within each `"row"`,
            or randomization only within each `"column"`.
        """
        self.original_shape = original_shape
        self.random_seed = random_seed
        self.rng = numpy.random.RandomState(self.random_seed)
        self.lookup: Dict[str, str] = {}
        full = make_well_array(*self.original_shape)
        if mode == "full":
            self.original_wells = full.flatten()
            self.randomized_wells = []
            self.randomized_wells = self.rng.permutation(self.original_wells).tolist()
            self.lookup = {owell: rwell for owell, rwell in zip(self.original_wells, self.randomized_wells)}
        elif mode == "row":
            for r in range(self.original_shape[0]):
                rowwells = full[r, :]
                randomized = self.rng.permutation(rowwells)
                for owell, rwell in zip(rowwells, randomized):
                    self.lookup[owell] = rwell
        elif mode == "column":
            self.original_wells = []
            for c in range(self.original_shape[1]):
                columnwells = full[:, c]
                randomized = self.rng.permutation(columnwells)
                for owell, rwell in zip(columnwells, randomized):
                    self.lookup[owell] = rwell
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        self.lookup_reverse = {rwell: owell for owell, rwell in self.lookup.items()}
        super().__init__()

    def randomize_wells(self, wells: ArrayLike) -> numpy.ndarray:
        """Randomize the given wells with the random state and assignment specified in __init__.

        Parameters
        ----------
        wells : array-like
            List or array of well ids

        Returns
        -------
        randomized : ndarray
            Array of well ids
        """

        input_wells = numpy.array(wells)
        wells_shape = input_wells.shape
        randomized_output_wells = [self.lookup.get(well) for well in input_wells]

        return numpy.array(randomized_output_wells).reshape(wells_shape)

    def derandomize_wells(self, wells: ArrayLike) -> numpy.ndarray:
        """Derandomize the given wells with the random state and assignment specified in __init__.

        Parameters
        ----------
        wells : array-like
            List or array of well ids

        Returns
        -------
        derandomized_output_wells : ndarray
            Array of well ids
        """
        input_wells = numpy.array(wells)
        wells_shape = input_wells.shape
        derandomized_output_wells = [self.lookup_reverse.get(well) for well in input_wells]

        return numpy.array(derandomized_output_wells).reshape(wells_shape)
