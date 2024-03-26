import warnings

import numpy as np
import pytest

from robotools.liquidhandling.exceptions import (
    VolumeOverflowError,
    VolumeUnderflowError,
)
from robotools.liquidhandling.labware import Labware, Trough


class TestStandardLabware:
    def test_init(self) -> None:
        plate = Labware("TestPlate", 2, 3, min_volume=50, max_volume=250, initial_volumes=30)
        assert plate.name == "TestPlate"
        assert plate.is_trough == False
        assert plate.row_ids == tuple("AB")
        assert plate.column_ids == [1, 2, 3]
        assert plate.n_rows == 2
        assert plate.n_columns == 3
        assert plate.min_volume == 50
        assert plate.max_volume == 250
        assert len(plate.history) == 1
        np.testing.assert_array_equal(plate.volumes, np.array([[30, 30, 30], [30, 30, 30]]))
        exp = {
            "A01": (0, 0),
            "A02": (0, 1),
            "A03": (0, 2),
            "B01": (1, 0),
            "B02": (1, 1),
            "B03": (1, 2),
        }
        assert plate.indices == exp
        with pytest.warns(DeprecationWarning, match="in favor of model-specific"):
            assert plate.positions == {
                "A01": 1,
                "A02": 3,
                "A03": 5,
                "B01": 2,
                "B02": 4,
                "B03": 6,
            }
        return

    def test_invalid_init(self) -> None:
        with pytest.raises(ValueError):
            Labware("A", 0, 3, min_volume=10, max_volume=250)
        with pytest.raises(ValueError):
            Labware("A", 3, 0, min_volume=10, max_volume=250)
        with pytest.raises(ValueError):
            Labware("A", 3, 4, min_volume=10, max_volume=250, virtual_rows=2)
        with pytest.raises(ValueError):
            Labware("A", 1, 4, min_volume=10, max_volume=250, virtual_rows=0)
        return

    def test_volume_limits(self) -> None:
        with pytest.raises(ValueError):
            Labware("A", 3, 4, min_volume=-30, max_volume=100)
        with pytest.raises(ValueError):
            Labware("A", 3, 4, min_volume=100, max_volume=70)
        with pytest.raises(ValueError):
            Labware("A", 3, 4, min_volume=10, max_volume=70, initial_volumes=100)
        with pytest.raises(ValueError):
            Labware("A", 3, 4, min_volume=10, max_volume=70, initial_volumes=-10)
        Labware("A", 3, 4, min_volume=10, max_volume=70, initial_volumes=50)
        return

    def test_initial_volumes(self) -> None:
        plate = Labware("TestPlate", 1, 3, min_volume=50, max_volume=250, initial_volumes=[20, 30, 40])
        np.testing.assert_array_equal(
            plate.volumes,
            np.array(
                [
                    [20, 30, 40],
                ]
            ),
        )
        return

    def test_logging(self) -> None:
        plate = Labware("TestPlate", 2, 3, min_volume=50, max_volume=250)
        plate.add(plate.wells, 25)
        plate.add(plate.wells, 25)
        plate.add(plate.wells, 25)
        plate.add(plate.wells, 25)
        assert len(plate.history) == 5
        return

    def test_log_condensation_first(self) -> None:
        plate = Labware("TestPlate", 2, 3, min_volume=50, max_volume=250)
        plate.add(plate.wells, 25, label="A")
        plate.add(plate.wells, 25, label="B")
        plate.add(plate.wells, 25, label="C")
        plate.add(plate.wells, 25, label="D")
        assert len(plate.history) == 5

        # condense the last two as 'D'
        plate.condense_log(2, label="last")
        assert len(plate.history) == 4
        assert plate.history[-1][0] == "D"
        np.testing.assert_array_equal(
            plate.history[-1][1],
            np.array(
                [
                    [100, 100, 100],
                    [100, 100, 100],
                ]
            ),
        )

        # condense the last three as 'A'
        plate.condense_log(3, label="first")
        assert len(plate.history) == 2
        assert plate.history[-1][0] == "A"
        np.testing.assert_array_equal(
            plate.history[-1][1],
            np.array(
                [
                    [100, 100, 100],
                    [100, 100, 100],
                ]
            ),
        )

        # condense the remaining two as 'prepared'
        plate.condense_log(3, label="prepared")
        assert len(plate.history) == 1
        assert plate.history[-1][0] == "prepared"
        np.testing.assert_array_equal(
            plate.history[-1][1],
            np.array(
                [
                    [100, 100, 100],
                    [100, 100, 100],
                ]
            ),
        )
        return

    def test_add_valid(self) -> None:
        plate = Labware("TestPlate", 4, 6, min_volume=100, max_volume=250)
        wells = ["A01", "A02", "B04"]
        plate.add(wells, 150)
        plate.add(wells, 3.5)
        assert len(plate.history) == 3
        for well in wells:
            assert plate.volumes[plate.indices[well]] == 153.5
        return

    def test_add_too_much(self) -> None:
        plate = Labware("TestPlate", 4, 6, min_volume=100, max_volume=250)
        wells = ["A01", "A02", "B04"]
        with pytest.raises(VolumeOverflowError):
            plate.add(wells, 500)
        return

    def test_remove_valid(self) -> None:
        plate = Labware("TestPlate", 2, 3, min_volume=50, max_volume=250, initial_volumes=200)
        wells = ["A01", "A02", "B03"]
        plate.remove(wells, 50)
        assert len(plate.history) == 2
        np.testing.assert_array_equal(plate.volumes, np.array([[150, 150, 200], [200, 200, 150]]))
        return

    def test_remove_too_much(self) -> None:
        plate = Labware("TestPlate", 4, 6, min_volume=100, max_volume=250)
        wells = ["A01", "A02", "B04"]
        with pytest.raises(VolumeUnderflowError):
            plate.remove(wells, 500)
        assert len(plate.history) == 1
        return


class TestTroughLabware:
    def test_warns_on_api(self) -> None:
        with pytest.warns(UserWarning, match="Troughs should be created with"):
            Labware("test", rows=1, columns=2, min_volume=100, max_volume=3000, virtual_rows=4)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Trough("test", virtual_rows=6, columns=2, min_volume=100, max_volume=3000)
        return

    def test_init_trough(self) -> None:
        trough = Trough("TestTrough", 5, 4, min_volume=1000, max_volume=50 * 1000, initial_volumes=30 * 1000)
        assert trough.name == "TestTrough"
        assert trough.is_trough
        assert trough.row_ids == tuple("ABCDE")
        assert trough.column_ids == [1, 2, 3, 4]
        assert trough.min_volume == 1000
        assert trough.max_volume == 50 * 1000
        assert len(trough.history) == 1
        np.testing.assert_array_equal(
            trough.volumes, np.array([[30 * 1000, 30 * 1000, 30 * 1000, 30 * 1000]])
        )
        assert trough.indices == {
            "A01": (0, 0),
            "A02": (0, 1),
            "A03": (0, 2),
            "A04": (0, 3),
            "B01": (0, 0),
            "B02": (0, 1),
            "B03": (0, 2),
            "B04": (0, 3),
            "C01": (0, 0),
            "C02": (0, 1),
            "C03": (0, 2),
            "C04": (0, 3),
            "D01": (0, 0),
            "D02": (0, 1),
            "D03": (0, 2),
            "D04": (0, 3),
            "E01": (0, 0),
            "E02": (0, 1),
            "E03": (0, 2),
            "E04": (0, 3),
        }
        with pytest.warns(DeprecationWarning, match="in favor of model-specific"):
            assert trough.positions == {
                "A01": 1,
                "A02": 6,
                "A03": 11,
                "A04": 16,
                "B01": 2,
                "B02": 7,
                "B03": 12,
                "B04": 17,
                "C01": 3,
                "C02": 8,
                "C03": 13,
                "C04": 18,
                "D01": 4,
                "D02": 9,
                "D03": 14,
                "D04": 19,
                "E01": 5,
                "E02": 10,
                "E03": 15,
                "E04": 20,
            }
        return

    def test_initial_volumes(self) -> None:
        trough = Trough(
            "TestTrough",
            5,
            4,
            min_volume=1000,
            max_volume=50 * 1000,
            initial_volumes=[30 * 1000, 20 * 1000, 20 * 1000, 20 * 1000],
        )
        np.testing.assert_array_equal(
            trough.volumes,
            np.array(
                [
                    [30 * 1000, 20 * 1000, 20 * 1000, 20 * 1000],
                ]
            ),
        )
        return

    def test_trough_add_valid(self) -> None:
        trough = Trough("TestTrough", 3, 4, min_volume=100, max_volume=250)
        # adding into the first column (which is actually one well)
        trough.add(["A01", "B01"], 50)
        np.testing.assert_array_equal(trough.volumes, np.array([[100, 0, 0, 0]]))
        # adding to the last row (separate wells)
        trough.add(["C01", "C02", "C03"], 50)
        np.testing.assert_array_equal(trough.volumes, np.array([[150, 50, 50, 0]]))
        assert len(trough.history) == 3
        return

    def test_trough_add_too_much(self) -> None:
        trough = Trough("TestTrough", 3, 4, min_volume=100, max_volume=1000)
        # adding into the first column (which is actually one well)
        with pytest.raises(VolumeOverflowError):
            trough.add(["A01", "B01"], 600)
        return

    def test_trough_remove_valid(self) -> None:
        trough = Trough("TestTrough", 3, 4, min_volume=1000, max_volume=30000, initial_volumes=3000)
        # adding into the first column (which is actually one well)
        trough.remove(["A01", "B01"], 50)
        np.testing.assert_array_equal(trough.volumes, np.array([[2900, 3000, 3000, 3000]]))
        # adding to the last row (separate wells)
        trough.remove(["C01", "C02", "C03"], 50)
        np.testing.assert_array_equal(trough.volumes, np.array([[2850, 2950, 2950, 3000]]))
        assert len(trough.history) == 3
        return

    def test_trough_remove_too_much(self) -> None:
        trough = Trough("TestTrough", 3, 4, min_volume=1000, max_volume=30 * 1000, initial_volumes=3000)
        # adding into the first column (which is actually one well)
        with pytest.raises(VolumeUnderflowError):
            trough.remove(["A01", "B01"], 2000)
        return
