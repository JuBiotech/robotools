"""Type hints for internal use, including aliases for NumPy arrays.

Note: NumPy 2.6 adds npt.Array2D, but we like to keep some backwards-compat.
"""

from typing import TypeAlias

import numpy as np

Bools2D: TypeAlias = np.ndarray[tuple[int, int], np.bool]
Floats2D: TypeAlias = np.ndarray[tuple[int, int], np.float64]
Strings2D: TypeAlias = np.ndarray[tuple[int, int], np.str_]
