import numpy as np
import pytest

from robotools.transform import WellRandomizer, WellRotator, WellShifter


class TestWellShifter:
    def test_identity_transform(self) -> None:
        A = (6, 8)
        B = (8, 12)
        shifter = WellShifter(A, B, shifted_A01="A01")

        original = ["A01", "C03", "D06", "F08"]
        expected = ["A01", "C03", "D06", "F08"]
        shifted = shifter.shift(original)
        np.testing.assert_array_equal(expected, shifter.shift(original))
        np.testing.assert_array_equal(shifter.unshift(shifted), original)
        return

    def test_center_shift(self) -> None:
        A = (6, 8)
        B = (8, 12)
        shifter = WellShifter(A, B, shifted_A01="B03")

        original = ["A01", "C03", "D06", "F08"]
        expected = ["B03", "D05", "E08", "G10"]
        shifted = shifter.shift(original)
        np.testing.assert_array_equal(expected, shifter.shift(original))
        np.testing.assert_array_equal(shifter.unshift(shifted), original)
        return

    def test_boundcheck(self) -> None:
        A = (6, 8)
        B = (8, 12)

        with pytest.raises(ValueError):
            WellShifter(A, B, shifted_A01="E03")

        with pytest.raises(ValueError):
            WellShifter(A, B, shifted_A01="B06")
        return


class TestWellRotator:
    def test_init(self) -> None:
        rotator = WellRotator(original_shape=(7, 3))
        assert rotator.original_shape == (7, 3)
        assert rotator.rotated_shape == (3, 7)
        return

    def test_clockwise(self) -> None:
        A = (6, 8)
        rotator = WellRotator(A)

        original = ["A01", "C03", "D06", "F08", "B04"]
        expected = ["A06", "C04", "F03", "H01", "D05"]
        rotated = rotator.rotate_cw(original)
        np.testing.assert_array_equal(expected, rotated)
        return

    def test_counterclockwise(self) -> None:
        A = (6, 8)
        rotator = WellRotator(A)

        original = ["A01", "C03", "D06", "F08", "B04"]
        expected = ["H01", "F03", "C04", "A06", "E02"]
        rotated = rotator.rotate_ccw(original)
        np.testing.assert_array_equal(expected, rotated)
        return


class TestWellRandomizer:
    def test_init(self):
        randomizer = WellRandomizer(original_shape=(1, 4), random_seed=13)
        assert randomizer.original_shape == (1, 4)
        assert randomizer.random_seed == 13
        np.testing.assert_array_equal(randomizer.randomized_wells, ["A02", "A04", "A01", "A03"])
        return

    def test_randomize_wells(self) -> None:
        A = (6, 8)
        S = 13
        randomizer = WellRandomizer(A, S)
        original = ["A01", "A02", "A03", "A04", "A05", "A06"]
        expected = ["A01", "F02", "A05", "A07", "B07", "D06"]
        randomized = randomizer.randomize_wells(original)
        np.testing.assert_array_equal(expected, randomized)
        return

    def test_derandomize_wells(self) -> None:
        A = (6, 8)
        S = 13
        randomizer = WellRandomizer(A, S)
        original = ["A01", "F02", "A05", "A07", "B07", "D06"]
        expected = ["A01", "A02", "A03", "A04", "A05", "A06"]
        derandomized = randomizer.derandomize_wells(original)
        np.testing.assert_array_equal(expected, derandomized)
        return

    def test_derandomize_wells_bug_29(self) -> None:
        A = (6, 8)
        S = 13
        randomizer = WellRandomizer(A, S)
        original = ["A01", "F02", "A05", "A07", "B07", "D06"][::-1]
        expected = ["A01", "A02", "A03", "A04", "A05", "A06"][::-1]
        derandomized = randomizer.derandomize_wells(original)
        np.testing.assert_array_equal(expected, derandomized)
        return

    def test_randomize_wells_in_row(self) -> None:
        A = (6, 8)
        S = 13
        randomizer = WellRandomizer(A, S, mode="row")
        original = ["A01", "A02", "A03", "B01", "C02", "B04"]
        expected = ["A02", "A05", "A04", "B08", "C05", "B06"]
        randomized = randomizer.randomize_wells(original)
        np.testing.assert_array_equal(expected, randomized)
        return

    def test_derandomize_wells_in_row(self) -> None:
        A = (6, 8)
        S = 13
        randomizer = WellRandomizer(A, S, mode="row")
        original = ["A02", "A05", "A04", "B08", "C05", "B06"]
        expected = ["A01", "A02", "A03", "B01", "C02", "B04"]
        randomized = randomizer.derandomize_wells(original)
        np.testing.assert_array_equal(expected, randomized)
        return

    def test_randomize_wells_in_column(self) -> None:
        A = (6, 8)
        S = 13
        randomizer = WellRandomizer(A, S, mode="column")
        original = ["A01", "A02", "A03", "B01", "C02", "B04"]
        expected = ["B01", "D02", "B03", "D01", "A02", "E04"]
        randomized = randomizer.randomize_wells(original)
        np.testing.assert_array_equal(expected, randomized)
        return

    def test_derandomize_wells_in_column(self) -> None:
        A = (6, 8)
        S = 13
        randomizer = WellRandomizer(A, S, mode="column")
        original = ["B01", "D02", "B03", "D01", "A02", "E04"]
        expected = ["A01", "A02", "A03", "B01", "C02", "B04"]
        randomized = randomizer.derandomize_wells(original)
        np.testing.assert_array_equal(expected, randomized)
        return
