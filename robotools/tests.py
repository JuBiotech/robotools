import logging
import os
import tempfile
import unittest

import numpy
import pytest

import robotools
from robotools import evotools, liquidhandling, transform


class TestStandardLabware(unittest.TestCase):
    def test_init(self) -> None:
        plate = liquidhandling.Labware("TestPlate", 2, 3, min_volume=50, max_volume=250, initial_volumes=30)
        self.assertEqual(plate.name, "TestPlate")
        self.assertFalse(plate.is_trough)
        self.assertEqual(plate.row_ids, tuple("AB"))
        self.assertEqual(plate.column_ids, [1, 2, 3])
        self.assertEqual(plate.n_rows, 2)
        self.assertEqual(plate.n_columns, 3)
        self.assertEqual(plate.min_volume, 50)
        self.assertEqual(plate.max_volume, 250)
        self.assertEqual(len(plate.history), 1)
        numpy.testing.assert_array_equal(plate.volumes, numpy.array([[30, 30, 30], [30, 30, 30]]))
        self.assertDictEqual(
            plate.indices,
            {
                "A01": (0, 0),
                "A02": (0, 1),
                "A03": (0, 2),
                "B01": (1, 0),
                "B02": (1, 1),
                "B03": (1, 2),
            },
        )
        self.assertDictEqual(
            plate.positions,
            {
                "A01": 1,
                "A02": 3,
                "A03": 5,
                "B01": 2,
                "B02": 4,
                "B03": 6,
            },
        )
        return

    def test_invalid_init(self) -> None:
        with self.assertRaises(ValueError):
            liquidhandling.Labware("A", 0, 3, min_volume=10, max_volume=250)
        with self.assertRaises(ValueError):
            liquidhandling.Labware("A", 3, 0, min_volume=10, max_volume=250)
        with self.assertRaises(ValueError):
            liquidhandling.Labware("A", 3, 4, min_volume=10, max_volume=250, virtual_rows=2)
        with self.assertRaises(ValueError):
            liquidhandling.Labware("A", 1, 4, min_volume=10, max_volume=250, virtual_rows=0)
        return

    def test_volume_limits(self) -> None:
        with self.assertRaises(ValueError):
            liquidhandling.Labware("A", 3, 4, min_volume=-30, max_volume=100)
        with self.assertRaises(ValueError):
            liquidhandling.Labware("A", 3, 4, min_volume=100, max_volume=70)
        with self.assertRaises(ValueError):
            liquidhandling.Labware("A", 3, 4, min_volume=10, max_volume=70, initial_volumes=100)
        with self.assertRaises(ValueError):
            liquidhandling.Labware("A", 3, 4, min_volume=10, max_volume=70, initial_volumes=-10)
        liquidhandling.Labware("A", 3, 4, min_volume=10, max_volume=70, initial_volumes=50)
        return

    def test_initial_volumes(self) -> None:
        plate = liquidhandling.Labware(
            "TestPlate", 1, 3, min_volume=50, max_volume=250, initial_volumes=[20, 30, 40]
        )
        numpy.testing.assert_array_equal(
            plate.volumes,
            numpy.array(
                [
                    [20, 30, 40],
                ]
            ),
        )
        return

    def test_logging(self) -> None:
        plate = liquidhandling.Labware("TestPlate", 2, 3, min_volume=50, max_volume=250)
        plate.add(plate.wells, 25)
        plate.add(plate.wells, 25)
        plate.add(plate.wells, 25)
        plate.add(plate.wells, 25)
        self.assertEqual(len(plate.history), 5)
        return

    def test_log_condensation_first(self) -> None:
        plate = liquidhandling.Labware("TestPlate", 2, 3, min_volume=50, max_volume=250)
        plate.add(plate.wells, 25, label="A")
        plate.add(plate.wells, 25, label="B")
        plate.add(plate.wells, 25, label="C")
        plate.add(plate.wells, 25, label="D")
        self.assertEqual(len(plate.history), 5)

        # condense the last two as 'D'
        plate.condense_log(2, label="last")
        self.assertEqual(len(plate.history), 4)
        self.assertEqual(plate.history[-1][0], "D")
        numpy.testing.assert_array_equal(
            plate.history[-1][1],
            numpy.array(
                [
                    [100, 100, 100],
                    [100, 100, 100],
                ]
            ),
        )

        # condense the last three as 'A'
        plate.condense_log(3, label="first")
        self.assertEqual(len(plate.history), 2)
        self.assertEqual(plate.history[-1][0], "A")
        numpy.testing.assert_array_equal(
            plate.history[-1][1],
            numpy.array(
                [
                    [100, 100, 100],
                    [100, 100, 100],
                ]
            ),
        )

        # condense the remaining two as 'prepared'
        plate.condense_log(3, label="prepared")
        self.assertEqual(len(plate.history), 1)
        self.assertEqual(plate.history[-1][0], "prepared")
        numpy.testing.assert_array_equal(
            plate.history[-1][1],
            numpy.array(
                [
                    [100, 100, 100],
                    [100, 100, 100],
                ]
            ),
        )
        return

    def test_add_valid(self) -> None:
        plate = liquidhandling.Labware("TestPlate", 4, 6, min_volume=100, max_volume=250)
        wells = ["A01", "A02", "B04"]
        plate.add(wells, 150)
        plate.add(wells, 3.5)
        self.assertEqual(len(plate.history), 3)
        for well in wells:
            assert plate.volumes[plate.indices[well]] == 153.5
        return

    def test_add_too_much(self) -> None:
        plate = liquidhandling.Labware("TestPlate", 4, 6, min_volume=100, max_volume=250)
        wells = ["A01", "A02", "B04"]
        with self.assertRaises(liquidhandling.VolumeOverflowError):
            plate.add(wells, 500)
        return

    def test_remove_valid(self) -> None:
        plate = liquidhandling.Labware("TestPlate", 2, 3, min_volume=50, max_volume=250, initial_volumes=200)
        wells = ["A01", "A02", "B03"]
        plate.remove(wells, 50)
        self.assertEqual(len(plate.history), 2)
        numpy.testing.assert_array_equal(plate.volumes, numpy.array([[150, 150, 200], [200, 200, 150]]))
        return

    def test_remove_too_much(self) -> None:
        plate = liquidhandling.Labware("TestPlate", 4, 6, min_volume=100, max_volume=250)
        wells = ["A01", "A02", "B04"]
        with self.assertRaises(liquidhandling.VolumeUnderflowError):
            plate.remove(wells, 500)
        self.assertEqual(len(plate.history), 1)
        return


class TestTroughLabware(unittest.TestCase):
    def test_warns_on_api(self) -> None:
        with pytest.warns(UserWarning, match="Troughs should be created with"):
            robotools.Labware("test", rows=1, columns=2, min_volume=100, max_volume=3000, virtual_rows=4)

        with pytest.warns(None) as record:
            robotools.Trough("test", virtual_rows=6, columns=2, min_volume=100, max_volume=3000)
        assert len(record) == 0
        return

    def test_init_trough(self) -> None:
        trough = liquidhandling.Trough(
            "TestTrough", 5, 4, min_volume=1000, max_volume=50 * 1000, initial_volumes=30 * 1000
        )
        self.assertEqual(trough.name, "TestTrough")
        self.assertTrue(trough.is_trough)
        self.assertEqual(trough.row_ids, tuple("ABCDE"))
        self.assertEqual(trough.column_ids, [1, 2, 3, 4])
        self.assertEqual(trough.min_volume, 1000)
        self.assertEqual(trough.max_volume, 50 * 1000)
        self.assertEqual(len(trough.history), 1)
        numpy.testing.assert_array_equal(
            trough.volumes, numpy.array([[30 * 1000, 30 * 1000, 30 * 1000, 30 * 1000]])
        )
        self.assertDictEqual(
            trough.indices,
            {
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
            },
        )
        self.assertDictEqual(
            trough.positions,
            {
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
            },
        )
        return

    def test_initial_volumes(self) -> None:
        trough = liquidhandling.Trough(
            "TestTrough",
            5,
            4,
            min_volume=1000,
            max_volume=50 * 1000,
            initial_volumes=[30 * 1000, 20 * 1000, 20 * 1000, 20 * 1000],
        )
        numpy.testing.assert_array_equal(
            trough.volumes,
            numpy.array(
                [
                    [30 * 1000, 20 * 1000, 20 * 1000, 20 * 1000],
                ]
            ),
        )
        return

    def test_trough_add_valid(self) -> None:
        trough = liquidhandling.Trough("TestTrough", 3, 4, min_volume=100, max_volume=250)
        # adding into the first column (which is actually one well)
        trough.add(["A01", "B01"], 50)
        numpy.testing.assert_array_equal(trough.volumes, numpy.array([[100, 0, 0, 0]]))
        # adding to the last row (separate wells)
        trough.add(["C01", "C02", "C03"], 50)
        numpy.testing.assert_array_equal(trough.volumes, numpy.array([[150, 50, 50, 0]]))
        self.assertEqual(len(trough.history), 3)
        return

    def test_trough_add_too_much(self) -> None:
        trough = liquidhandling.Trough("TestTrough", 3, 4, min_volume=100, max_volume=1000)
        # adding into the first column (which is actually one well)
        with self.assertRaises(liquidhandling.VolumeOverflowError):
            trough.add(["A01", "B01"], 600)
        return

    def test_trough_remove_valid(self) -> None:
        trough = liquidhandling.Trough(
            "TestTrough", 3, 4, min_volume=1000, max_volume=30000, initial_volumes=3000
        )
        # adding into the first column (which is actually one well)
        trough.remove(["A01", "B01"], 50)
        numpy.testing.assert_array_equal(trough.volumes, numpy.array([[2900, 3000, 3000, 3000]]))
        # adding to the last row (separate wells)
        trough.remove(["C01", "C02", "C03"], 50)
        numpy.testing.assert_array_equal(trough.volumes, numpy.array([[2850, 2950, 2950, 3000]]))
        self.assertEqual(len(trough.history), 3)
        return

    def test_trough_remove_too_much(self) -> None:
        trough = liquidhandling.Trough(
            "TestTrough", 3, 4, min_volume=1000, max_volume=30 * 1000, initial_volumes=3000
        )
        # adding into the first column (which is actually one well)
        with self.assertRaises(liquidhandling.VolumeUnderflowError):
            trough.remove(["A01", "B01"], 2000)
        return


class TestWorklist(unittest.TestCase):
    def test_context(self) -> None:
        with evotools.Worklist() as worklist:
            self.assertIsNotNone(worklist)
        return

    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(rack_label=None, position=1, volume=15)
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(rack_label=15, position=1, volume=15)
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="thisisaveryverylongracklabelthatexceedsthemaximumlength", position=1, volume=15
            )
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="rack label; with semicolon", position=1, volume=15
            )
        evotools._prepare_aspirate_dispense_parameters(rack_label="valid rack label", position=1, volume=15)
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=None, volume=15)
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position="3", volume=15)
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=-1, volume=15)
        evotools._prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=15)
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=None)
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume="nan")
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=float("nan")
            )
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=-15.4)
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume="bla")
        evotools._prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume="15")
        evotools._prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=20)
        evotools._prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=23.78)
        evotools._prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=numpy.array(23.4)
        )

        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, liquid_class=None
            )
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, liquid_class="liquid;class"
            )
        evotools._prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, liquid_class="valid liquid class"
        )

        _, _, _, _, tip, _, _, _, _ = evotools._prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=4
        )
        assert tip == 8
        _, _, _, _, tip, _, _, _, _ = evotools._prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=evotools.Tip.T5
        )
        assert tip == 16
        _, _, _, _, tip, _, _, _, _ = evotools._prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=(evotools.Tip.T4, 4)
        )
        assert tip == 8
        _, _, _, _, tip, _, _, _, _ = evotools._prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=[evotools.Tip.T1, 4]
        )
        assert tip == 9
        _, _, _, _, tip, _, _, _, _ = evotools._prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=[1, 4]
        )
        assert tip == 9
        _, _, _, _, tip, _, _, _, _ = evotools._prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=[1, evotools.Tip.T4]
        )
        assert tip == 9
        _, _, _, _, tip, _, _, _, _ = evotools._prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=evotools.Tip.Any
        )
        assert tip == ""

        with pytest.raises(ValueError, match="no Tip.Any elements are allowed"):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, tip=(evotools.Tip.T1, evotools.Tip.Any)
            )
        with pytest.raises(ValueError, match="tip must be an int between 1 and 8, Tip or Iterable"):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, tip=None
            )
        with pytest.raises(ValueError, match="it may only contain int or Tip values"):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, tip=[1, 2.6]
            )
        with pytest.raises(ValueError, match="should be an int between 1 and 8 for _int_to_tip"):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, tip=12
            )

        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, rack_id=None
            )
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, rack_id="invalid;rack"
            )
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough",
                position=1,
                volume=15,
                rack_id="thisisaveryverylongrackthatexceedsthemaximumlength",
            )
        evotools._prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, rack_id="1235464"
        )
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, rack_type=None
            )
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, rack_type="invalid;rack type"
            )
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough",
                position=1,
                volume=15,
                rack_type="thisisaveryverylongracktypethatexceedsthemaximumlength",
            )
        evotools._prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, rack_type="valid rack type"
        )
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, forced_rack_type=None
            )
        with self.assertRaises(ValueError):
            evotools._prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, forced_rack_type="invalid;forced rack type"
            )
        evotools._prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, forced_rack_type="valid forced rack type"
        )
        return

    def test_comment(self) -> None:
        with evotools.Worklist() as wl:
            # empty and None comments should be ignored
            wl.comment("")
            wl.comment(None)
            # this will be the first actual comment
            wl.comment("This is a simple comment")
            with self.assertRaises(ValueError):
                wl.comment("It must not contain ; semicolons")
            wl.comment(
                """
            But it may very well be
            a multiline comment
            """
            )
            self.assertEqual(
                wl, ["C;This is a simple comment", "C;But it may very well be", "C;a multiline comment"]
            )
        return

    def test_wash(self) -> None:
        with evotools.Worklist() as wl:
            wl.wash()
            with self.assertRaises(ValueError):
                wl.wash(scheme=15)
            with self.assertRaises(ValueError):
                wl.wash(scheme="2")
            wl.wash(scheme=1)
            wl.wash(scheme=2)
            wl.wash(scheme=3)
            wl.wash(scheme=4)
            self.assertEqual(
                wl,
                [
                    "W1;",
                    "W1;",
                    "W2;",
                    "W3;",
                    "W4;",
                ],
            )
        return

    def test_decontaminate(self) -> None:
        with evotools.Worklist() as wl:
            wl.decontaminate()
            self.assertEqual(
                wl,
                [
                    "WD;",
                ],
            )
        return

    def test_flush(self) -> None:
        with evotools.Worklist() as wl:
            wl.flush()
            self.assertEqual(
                wl,
                [
                    "F;",
                ],
            )
        return

    def test_commit(self) -> None:
        with evotools.Worklist() as wl:
            wl.commit()
            self.assertEqual(
                wl,
                [
                    "B;",
                ],
            )
        return

    def test_set_diti(self) -> None:
        with evotools.Worklist() as wl:
            wl.set_diti(diti_index=1)
            with self.assertRaises(evotools.InvalidOperationError):
                wl.set_diti(diti_index=2)
            wl.commit()
            wl.set_diti(diti_index=2)
            self.assertEqual(
                wl,
                [
                    "S;1",
                    "B;",
                    "S;2",
                ],
            )
        return

    def test_aspirate_single(self) -> None:
        with evotools.Worklist() as wl:
            wl.aspirate_well("WaterTrough", 1, 200)
            self.assertEqual(wl[-1], "A;WaterTrough;;;1;;200.00;;;;")
            wl.aspirate_well(
                "WaterTrough", 1, 200, rack_id="12345", rack_type="my_rack_id", tube_id="my_tube_id"
            )
            self.assertEqual(wl[-1], "A;WaterTrough;12345;my_rack_id;1;my_tube_id;200.00;;;;")
            wl.aspirate_well(
                "WaterTrough", 1, 200, liquid_class="my_liquid_class", tip=8, forced_rack_type="forced_rack"
            )
            self.assertEqual(wl[-1], "A;WaterTrough;;;1;;200.00;my_liquid_class;;128;forced_rack")
        return

    def test_dispense_single(self) -> None:
        with evotools.Worklist() as wl:
            wl.dispense_well("WaterTrough", 1, 200)
            self.assertEqual(wl[-1], "D;WaterTrough;;;1;;200.00;;;;")
            wl.dispense_well(
                "WaterTrough", 1, 200, rack_id="12345", rack_type="my_rack_id", tube_id="my_tube_id"
            )
            self.assertEqual(wl[-1], "D;WaterTrough;12345;my_rack_id;1;my_tube_id;200.00;;;;")
            wl.dispense_well(
                "WaterTrough", 1, 200, liquid_class="my_liquid_class", tip=8, forced_rack_type="forced_rack"
            )
            self.assertEqual(wl[-1], "D;WaterTrough;;;1;;200.00;my_liquid_class;;128;forced_rack")
        return

    def test_aspirate_systemliquid(self) -> None:
        with evotools.Worklist() as wl:
            wl.aspirate_well(evotools.Labwares.SystemLiquid, 1, 200)
            self.assertEqual(wl[-1], "A;Systemliquid;;;1;;200.00;;;;")
        return

    def test_save(self) -> None:
        tf = tempfile.mktemp() + ".gwl"
        error = None
        try:
            with evotools.Worklist() as worklist:
                worklist.flush()
                worklist.save(tf)
                self.assertTrue(os.path.exists(tf))
                # also check that the file can be overwritten if it exists already
                worklist.save(tf)
            self.assertTrue(os.path.exists(tf))
            with open(tf) as file:
                lines = file.readlines()
                self.assertEqual(lines, ["F;"])
        except Exception as ex:
            error = ex
        finally:
            os.remove(tf)
        self.assertFalse(os.path.exists(tf))
        if error:
            raise error
        return

    def test_autosave(self) -> None:
        tf = tempfile.mktemp() + ".gwl"
        error = None
        try:
            with evotools.Worklist(tf) as worklist:
                worklist.flush()
            self.assertTrue(os.path.exists(tf))
            with open(tf) as file:
                lines = file.readlines()
                self.assertEqual(lines, ["F;"])
        except Exception as ex:
            error = ex
        finally:
            os.remove(tf)
        self.assertFalse(os.path.exists(tf))
        if error:
            raise error
        return


class TestStandardLabwareWorklist(unittest.TestCase):
    def test_aspirate(self) -> None:
        source = liquidhandling.Labware(
            "SourceLW", rows=3, columns=3, min_volume=10, max_volume=200, initial_volumes=200
        )
        with evotools.Worklist() as wl:
            wl.aspirate(source, ["A01", "A02", "C02"], 50, label=None)
            wl.aspirate(source, ["A03", "B03", "C03"], [10, 20, 30.5], label="second aspirate")
            self.assertEqual(
                wl,
                [
                    "A;SourceLW;;;1;;50.00;;;;",
                    "A;SourceLW;;;4;;50.00;;;;",
                    "A;SourceLW;;;6;;50.00;;;;",
                    "C;second aspirate",
                    "A;SourceLW;;;7;;10.00;;;;",
                    "A;SourceLW;;;8;;20.00;;;;",
                    "A;SourceLW;;;9;;30.50;;;;",
                ],
            )
            self.assertTrue(
                numpy.array_equal(
                    source.volumes,
                    [
                        [150, 150, 190],
                        [200, 200, 180],
                        [200, 150, 169.5],
                    ],
                )
            )
            self.assertEqual(len(source.history), 3)
        return

    def test_aspirate_2d_volumes(self) -> None:
        source = liquidhandling.Labware(
            "SourceLW", rows=2, columns=3, min_volume=10, max_volume=200, initial_volumes=200
        )
        with evotools.Worklist() as wl:
            wl.aspirate(
                source,
                source.wells[:, :2],
                volumes=numpy.array(
                    [
                        [20, 30],
                        [15.3, 17.53],
                    ]
                ),
            )
            self.assertEqual(
                wl,
                [
                    "A;SourceLW;;;1;;20.00;;;;",
                    "A;SourceLW;;;2;;15.30;;;;",
                    "A;SourceLW;;;3;;30.00;;;;",
                    "A;SourceLW;;;4;;17.53;;;;",
                ],
            )
            self.assertTrue(
                numpy.array_equal(source.volumes, [[180, 170, 200], [200 - 15.3, 200 - 17.53, 200]])
            )
            self.assertEqual(len(source.history), 2)
        return

    def test_dispense(self) -> None:
        destination = liquidhandling.Labware(
            "DestinationLW", rows=2, columns=3, min_volume=10, max_volume=200
        )
        with evotools.Worklist() as wl:
            wl.dispense(destination, ["A01", "A02", "A03"], 150, label=None)
            wl.dispense(destination, ["B01", "B02", "B03"], [10, 20, 30.5], label="second dispense")
            self.assertEqual(
                wl,
                [
                    "D;DestinationLW;;;1;;150.00;;;;",
                    "D;DestinationLW;;;3;;150.00;;;;",
                    "D;DestinationLW;;;5;;150.00;;;;",
                    "C;second dispense",
                    "D;DestinationLW;;;2;;10.00;;;;",
                    "D;DestinationLW;;;4;;20.00;;;;",
                    "D;DestinationLW;;;6;;30.50;;;;",
                ],
            )
            self.assertTrue(
                numpy.array_equal(
                    destination.volumes,
                    [
                        [150, 150, 150],
                        [10, 20, 30.5],
                    ],
                )
            )
            self.assertEqual(len(destination.history), 3)
        return

    def test_dispense_2d_volumes(self) -> None:
        destination = liquidhandling.Labware(
            "DestinationLW", rows=2, columns=3, min_volume=10, max_volume=200
        )
        with evotools.Worklist() as wl:
            wl.dispense(
                destination,
                destination.wells[:, :2],
                volumes=numpy.array(
                    [
                        [20, 30],
                        [15.3, 17.53],
                    ]
                ),
            )
            self.assertEqual(
                wl,
                [
                    "D;DestinationLW;;;1;;20.00;;;;",
                    "D;DestinationLW;;;2;;15.30;;;;",
                    "D;DestinationLW;;;3;;30.00;;;;",
                    "D;DestinationLW;;;4;;17.53;;;;",
                ],
            )
            numpy.testing.assert_array_equal(destination.volumes, [[20, 30, 0], [15.3, 17.53, 0]])
            self.assertEqual(len(destination.history), 2)
        return

    def test_skip_zero_volumes(self) -> None:
        source = liquidhandling.Labware(
            "SourceLW", rows=3, columns=3, min_volume=10, max_volume=200, initial_volumes=200
        )
        destination = liquidhandling.Labware(
            "DestinationLW", rows=2, columns=3, min_volume=10, max_volume=200
        )
        with evotools.Worklist() as wl:
            wl.aspirate(source, ["A03", "B03", "C03"], [10, 0, 30.5])
            wl.dispense(destination, ["B01", "B02", "B03"], [10, 0, 30.5])
            self.assertEqual(
                wl,
                [
                    "A;SourceLW;;;7;;10.00;;;;",
                    "A;SourceLW;;;9;;30.50;;;;",
                    "D;DestinationLW;;;2;;10.00;;;;",
                    "D;DestinationLW;;;6;;30.50;;;;",
                ],
            )
            self.assertTrue(
                numpy.array_equal(
                    source.volumes,
                    [
                        [200, 200, 190],
                        [200, 200, 200],
                        [200, 200, 169.5],
                    ],
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    destination.volumes,
                    [
                        [0, 0, 0],
                        [10, 0, 30.5],
                    ],
                )
            )
            self.assertEqual(len(destination.history), 2)
        return

    def test_transfer_2d_volumes(self) -> None:
        A = liquidhandling.Labware("A", 2, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = liquidhandling.Labware("B", 2, 4, min_volume=50, max_volume=250)
        with evotools.Worklist() as wl:
            wl.transfer(
                A,
                A.wells[:, :2],
                B,
                B.wells[:, :2],
                volumes=numpy.array(
                    [
                        [20, 30],
                        [15.3, 17.53],
                    ]
                ),
            )
            self.assertEqual(
                wl,
                [
                    "A;A;;;1;;20.00;;;;",
                    "D;B;;;1;;20.00;;;;",
                    "W1;",
                    "A;A;;;2;;15.30;;;;",
                    "D;B;;;2;;15.30;;;;",
                    "W1;",
                    "A;A;;;3;;30.00;;;;",
                    "D;B;;;3;;30.00;;;;",
                    "W1;",
                    "A;A;;;4;;17.53;;;;",
                    "D;B;;;4;;17.53;;;;",
                    "W1;",
                ],
            )
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [180, 170, 200, 200],
                            [200 - 15.3, 200 - 17.53, 200, 200],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [20, 30, 0, 0],
                            [15.3, 17.53, 0, 0],
                        ]
                    ),
                )
            )
            self.assertEqual(len(A.history), 2)
            self.assertEqual(len(B.history), 2)
        return

    def test_transfer_2d_volumes_no_wash(self) -> None:
        A = liquidhandling.Labware("A", 2, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = liquidhandling.Labware("B", 2, 4, min_volume=50, max_volume=250)
        with evotools.Worklist() as wl:
            wl.transfer(
                A,
                A.wells[:, :2],
                B,
                B.wells[:, :2],
                volumes=numpy.array(
                    [
                        [20, 30],
                        [15.3, 17.53],
                    ]
                ),
                wash_scheme=None,
            )
            self.assertEqual(
                wl,
                [
                    "A;A;;;1;;20.00;;;;",
                    "D;B;;;1;;20.00;;;;",
                    "A;A;;;2;;15.30;;;;",
                    "D;B;;;2;;15.30;;;;",
                    "A;A;;;3;;30.00;;;;",
                    "D;B;;;3;;30.00;;;;",
                    "A;A;;;4;;17.53;;;;",
                    "D;B;;;4;;17.53;;;;",
                ],
            )
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [180, 170, 200, 200],
                            [200 - 15.3, 200 - 17.53, 200, 200],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [20, 30, 0, 0],
                            [15.3, 17.53, 0, 0],
                        ]
                    ),
                )
            )
            self.assertEqual(len(A.history), 2)
            self.assertEqual(len(B.history), 2)
        return

    def test_transfer_many_many(self) -> None:
        A = liquidhandling.Labware("A", 3, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = liquidhandling.Labware("B", 3, 4, min_volume=50, max_volume=250)
        wells = ["A01", "B01"]
        with evotools.Worklist() as worklist:
            worklist.transfer(A, wells, B, wells, 50, label="first transfer")
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [150, 200, 200, 200],
                            [150, 200, 200, 200],
                            [200, 200, 200, 200],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [50, 0, 0, 0],
                            [50, 0, 0, 0],
                            [0, 0, 0, 0],
                        ]
                    ),
                )
            )
            worklist.transfer(A, ["A03", "B04"], B, ["A04", "B04"], 50, label="second transfer")
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [150, 200, 150, 200],
                            [150, 200, 200, 150],
                            [200, 200, 200, 200],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [50, 0, 0, 50],
                            [50, 0, 0, 50],
                            [0, 0, 0, 0],
                        ]
                    ),
                )
            )
            self.assertEqual(
                worklist,
                [
                    "C;first transfer",
                    "A;A;;;1;;50.00;;;;",
                    "D;B;;;1;;50.00;;;;",
                    "W1;",
                    "A;A;;;2;;50.00;;;;",
                    "D;B;;;2;;50.00;;;;",
                    "W1;",
                    "C;second transfer",
                    "A;A;;;7;;50.00;;;;",
                    "D;B;;;10;;50.00;;;;",
                    "W1;",
                    "A;A;;;11;;50.00;;;;",
                    "D;B;;;11;;50.00;;;;",
                    "W1;",
                ],
            )
            self.assertEqual(len(A.history), 3)
            self.assertEqual(len(B.history), 3)
        return

    def test_transfer_many_many_2d(self) -> None:
        A = liquidhandling.Labware("A", 3, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = liquidhandling.Labware("B", 3, 4, min_volume=50, max_volume=250)
        wells = A.wells[:, :2]
        with evotools.Worklist() as worklist:
            worklist.transfer(A, wells, B, wells, 50)
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [150, 150, 200, 200],
                            [150, 150, 200, 200],
                            [150, 150, 200, 200],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [50, 50, 0, 0],
                            [50, 50, 0, 0],
                            [50, 50, 0, 0],
                        ]
                    ),
                )
            )
            self.assertEqual(
                worklist,
                [
                    # first transfer
                    "A;A;;;1;;50.00;;;;",
                    "D;B;;;1;;50.00;;;;",
                    "W1;",
                    "A;A;;;2;;50.00;;;;",
                    "D;B;;;2;;50.00;;;;",
                    "W1;",
                    "A;A;;;3;;50.00;;;;",
                    "D;B;;;3;;50.00;;;;",
                    "W1;",
                    "A;A;;;4;;50.00;;;;",
                    "D;B;;;4;;50.00;;;;",
                    "W1;",
                    "A;A;;;5;;50.00;;;;",
                    "D;B;;;5;;50.00;;;;",
                    "W1;",
                    "A;A;;;6;;50.00;;;;",
                    "D;B;;;6;;50.00;;;;",
                    "W1;",
                ],
            )
            self.assertEqual(len(A.history), 2)
            self.assertEqual(len(B.history), 2)
        return

    def test_transfer_one_many(self) -> None:
        A = liquidhandling.Labware("A", 3, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = liquidhandling.Labware("B", 3, 4, min_volume=50, max_volume=250)
        with evotools.Worklist() as worklist:
            worklist.transfer(A, "A01", B, ["B01", "B02", "B03"], 25)
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [125, 200, 200, 200],
                            [200, 200, 200, 200],
                            [200, 200, 200, 200],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [0, 0, 0, 0],
                            [25, 25, 25, 0],
                            [0, 0, 0, 0],
                        ]
                    ),
                )
            )
            worklist.transfer(A, ["A01"], B, ["B01", "B02", "B03"], 25)
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [50, 200, 200, 200],
                            [200, 200, 200, 200],
                            [200, 200, 200, 200],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [0, 0, 0, 0],
                            [50, 50, 50, 0],
                            [0, 0, 0, 0],
                        ]
                    ),
                )
            )
            self.assertEqual(
                worklist,
                [
                    # first transfer
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;5;;25.00;;;;",
                    "W1;",
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;8;;25.00;;;;",
                    "W1;",
                    # second transfer
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;5;;25.00;;;;",
                    "W1;",
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;8;;25.00;;;;",
                    "W1;",
                ],
            )
            self.assertEqual(len(A.history), 3)
            self.assertEqual(len(B.history), 3)
        return

    def test_transfer_many_one(self) -> None:
        A = liquidhandling.Labware("A", 3, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = liquidhandling.Labware("B", 3, 4, min_volume=50, max_volume=250)
        with evotools.Worklist() as worklist:
            worklist.transfer(A, ["A01", "A02", "A03"], B, "B01", 25)
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [175, 175, 175, 200],
                            [200, 200, 200, 200],
                            [200, 200, 200, 200],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [0, 0, 0, 0],
                            [75, 0, 0, 0],
                            [0, 0, 0, 0],
                        ]
                    ),
                )
            )
            self.assertEqual(
                worklist,
                [
                    # first transfer
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                    "A;A;;;4;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                    "A;A;;;7;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                ],
            )
            self.assertEqual(len(A.history), 2)
            self.assertEqual(len(B.history), 2)
        return

    def test_tip_selection(self) -> None:
        A = liquidhandling.Labware("A", 3, 4, min_volume=10, max_volume=250, initial_volumes=100)
        with evotools.Worklist() as wl:
            wl.aspirate(A, "A01", 10, tip=1)
            wl.aspirate(A, "A01", 10, tip=2)
            wl.aspirate(A, "A01", 10, tip=3)
            wl.aspirate(A, "A01", 10, tip=4)
            wl.aspirate(A, "A01", 10, tip=5)
            wl.aspirate(A, "A01", 10, tip=6)
            wl.aspirate(A, "A01", 10, tip=7)
            wl.aspirate(A, "A01", 10, tip=8)
            wl.dispense(A, "B01", 10, tip=evotools.Tip.T1)
            wl.dispense(A, "B02", 10, tip=evotools.Tip.T2)
            wl.dispense(A, "B03", 10, tip=evotools.Tip.T3)
            wl.dispense(A, "B04", 10, tip=evotools.Tip.T4)
            wl.dispense(A, "B04", 10, tip=evotools.Tip.T5)
            wl.dispense(A, "B04", 10, tip=evotools.Tip.T6)
            wl.dispense(A, "B04", 10, tip=evotools.Tip.T7)
            wl.dispense(A, "B04", 10, tip=evotools.Tip.T8)
            self.assertEqual(
                wl,
                [
                    "A;A;;;1;;10.00;;;1;",
                    "A;A;;;1;;10.00;;;2;",
                    "A;A;;;1;;10.00;;;4;",
                    "A;A;;;1;;10.00;;;8;",
                    "A;A;;;1;;10.00;;;16;",
                    "A;A;;;1;;10.00;;;32;",
                    "A;A;;;1;;10.00;;;64;",
                    "A;A;;;1;;10.00;;;128;",
                    "D;A;;;2;;10.00;;;1;",
                    "D;A;;;5;;10.00;;;2;",
                    "D;A;;;8;;10.00;;;4;",
                    "D;A;;;11;;10.00;;;8;",
                    "D;A;;;11;;10.00;;;16;",
                    "D;A;;;11;;10.00;;;32;",
                    "D;A;;;11;;10.00;;;64;",
                    "D;A;;;11;;10.00;;;128;",
                ],
            )
        return

    def test_history_condensation(self) -> None:
        A = liquidhandling.Labware("A", 3, 2, min_volume=300, max_volume=4600, initial_volumes=1500)
        B = liquidhandling.Labware("B", 3, 2, min_volume=300, max_volume=4600, initial_volumes=1500)

        with evotools.Worklist() as wl:
            wl.transfer(A, ["A01", "B01", "C02"], B, ["A01", "B02", "C01"], [900, 100, 900], label="transfer")

        self.assertEqual(len(A.history), 2)
        self.assertEqual(A.history[-1][0], "transfer")
        numpy.testing.assert_array_equal(
            A.history[-1][1],
            numpy.array(
                [
                    [1500 - 900, 1500],
                    [1500 - 100, 1500],
                    [1500, 1500 - 900],
                ]
            ),
        )

        self.assertEqual(len(B.history), 2)
        self.assertEqual(B.history[-1][0], "transfer")
        numpy.testing.assert_array_equal(
            B.history[-1][1],
            numpy.array(
                [
                    [1500 + 900, 1500],
                    [1500, 1500 + 100],
                    [1500 + 900, 1500],
                ]
            ),
        )
        return

    def test_history_condensation_within_labware(self) -> None:
        A = liquidhandling.Labware("A", 3, 2, min_volume=300, max_volume=4600, initial_volumes=1500)

        with evotools.Worklist() as wl:
            wl.transfer(A, ["A01", "B01", "C02"], A, ["A01", "B02", "C01"], [900, 100, 900], label="mix")

        self.assertEqual(len(A.history), 2)
        self.assertEqual(A.history[-1][0], "mix")
        numpy.testing.assert_array_equal(
            A.history[-1][1],
            numpy.array(
                [
                    [1500 - 900 + 900, 1500],
                    [1500 - 100, 1500 + 100],
                    [1500 + 900, 1500 - 900],
                ]
            ),
        )
        return


class TestTroughLabwareWorklist(unittest.TestCase):
    def test_aspirate(self) -> None:
        source = liquidhandling.Trough(
            "SourceLW", virtual_rows=3, columns=3, min_volume=10, max_volume=200, initial_volumes=200
        )
        with evotools.Worklist() as wl:
            wl.aspirate(source, ["A01", "A02", "C02"], 50)
            wl.aspirate(source, ["A01", "A02", "C02"], [1, 2, 3])
            self.assertEqual(
                wl,
                [
                    "A;SourceLW;;;1;;50.00;;;;",
                    "A;SourceLW;;;4;;50.00;;;;",
                    "A;SourceLW;;;6;;50.00;;;;",
                    "A;SourceLW;;;1;;1.00;;;;",
                    "A;SourceLW;;;4;;2.00;;;;",
                    "A;SourceLW;;;6;;3.00;;;;",
                ],
            )
            numpy.testing.assert_array_equal(source.volumes, [[149, 95, 200]])
            self.assertEqual(len(source.history), 3)
        return

    def test_dispense(self) -> None:
        destination = liquidhandling.Trough(
            "DestinationLW", virtual_rows=3, columns=3, min_volume=10, max_volume=200
        )
        with evotools.Worklist() as wl:
            wl.dispense(destination, ["A01", "A02", "A03", "B01"], 50)
            wl.dispense(destination, ["A01", "A02", "C02"], [1, 2, 3])
            self.assertEqual(
                wl,
                [
                    "D;DestinationLW;;;1;;50.00;;;;",
                    "D;DestinationLW;;;4;;50.00;;;;",
                    "D;DestinationLW;;;7;;50.00;;;;",
                    "D;DestinationLW;;;2;;50.00;;;;",
                    "D;DestinationLW;;;1;;1.00;;;;",
                    "D;DestinationLW;;;4;;2.00;;;;",
                    "D;DestinationLW;;;6;;3.00;;;;",
                ],
            )
            numpy.testing.assert_array_equal(destination.volumes, [[101, 55, 50]])
            self.assertEqual(len(destination.history), 3)
        return

    def test_transfer_many_many(self) -> None:
        A = liquidhandling.Trough("A", 3, 4, min_volume=50, max_volume=2500, initial_volumes=2000)
        B = liquidhandling.Labware("B", 3, 4, min_volume=50, max_volume=250)
        with evotools.Worklist() as worklist:
            worklist.transfer(A, ["A01", "B01"], B, ["A01", "B01"], 50)
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [1900, 2000, 2000, 2000],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [50, 0, 0, 0],
                            [50, 0, 0, 0],
                            [0, 0, 0, 0],
                        ]
                    ),
                )
            )
            worklist.transfer(A, ["A03", "B04"], B, ["A04", "B04"], [50, 75])
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [1900, 2000, 1950, 1925],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [50, 0, 0, 50],
                            [50, 0, 0, 75],
                            [0, 0, 0, 0],
                        ]
                    ),
                )
            )
            self.assertEqual(
                worklist,
                [
                    # first transfer
                    "A;A;;;1;;50.00;;;;",
                    "D;B;;;1;;50.00;;;;",
                    "W1;",
                    "A;A;;;2;;50.00;;;;",
                    "D;B;;;2;;50.00;;;;",
                    "W1;",
                    # second transfer
                    "A;A;;;7;;50.00;;;;",
                    "D;B;;;10;;50.00;;;;",
                    "W1;",
                    "A;A;;;11;;75.00;;;;",
                    "D;B;;;11;;75.00;;;;",
                    "W1;",
                ],
            )
            self.assertEqual(len(A.history), 3)
            self.assertEqual(len(B.history), 3)
        return

    def test_transfer_one_many(self) -> None:
        A = liquidhandling.Trough("A", 3, 4, min_volume=50, max_volume=2500, initial_volumes=2000)
        B = liquidhandling.Labware("B", 3, 4, min_volume=50, max_volume=250)
        with evotools.Worklist() as worklist:
            worklist.transfer(A, "A01", B, ["B01", "B02", "B03"], 25)
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [1925, 2000, 2000, 2000],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [0, 0, 0, 0],
                            [25, 25, 25, 0],
                            [0, 0, 0, 0],
                        ]
                    ),
                )
            )
            self.assertEqual(
                worklist,
                [
                    # first transfer
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;5;;25.00;;;;",
                    "W1;",
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;8;;25.00;;;;",
                    "W1;",
                ],
            )

            worklist.transfer(A, ["A01"], B, ["B01", "B02", "B03"], [25, 30, 35])
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [1835, 2000, 2000, 2000],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [0, 0, 0, 0],
                            [50, 55, 60, 0],
                            [0, 0, 0, 0],
                        ]
                    ),
                )
            )
            self.assertEqual(
                worklist,
                [
                    # first transfer
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;5;;25.00;;;;",
                    "W1;",
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;8;;25.00;;;;",
                    "W1;",
                    # second transfer
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                    "A;A;;;1;;30.00;;;;",
                    "D;B;;;5;;30.00;;;;",
                    "W1;",
                    "A;A;;;1;;35.00;;;;",
                    "D;B;;;8;;35.00;;;;",
                    "W1;",
                ],
            )
            self.assertEqual(len(A.history), 3)
            self.assertEqual(len(B.history), 3)
        return

    def test_transfer_many_one(self) -> None:
        A = liquidhandling.Trough(
            "A", 3, 4, min_volume=50, max_volume=2500, initial_volumes=[2000, 1500, 1000, 500]
        )
        B = liquidhandling.Labware("B", 3, 4, min_volume=10, max_volume=250, initial_volumes=100)
        with evotools.Worklist() as worklist:
            worklist.transfer(A, ["A01", "A02", "A03"], B, "B01", 25)
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [1975, 1475, 975, 500],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [100, 100, 100, 100],
                            [175, 100, 100, 100],
                            [100, 100, 100, 100],
                        ]
                    ),
                )
            )
            self.assertEqual(
                worklist,
                [
                    # first transfer
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                    "A;A;;;4;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                    "A;A;;;7;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                ],
            )

            worklist.transfer(B, B.wells[:, 2], A, A.wells[:, 3], [50, 60, 70])
            self.assertTrue(
                numpy.array_equal(
                    A.volumes,
                    numpy.array(
                        [
                            [1975, 1475, 975, 680],
                        ]
                    ),
                )
            )
            self.assertTrue(
                numpy.array_equal(
                    B.volumes,
                    numpy.array(
                        [
                            [100, 100, 50, 100],
                            [175, 100, 40, 100],
                            [100, 100, 30, 100],
                        ]
                    ),
                )
            )
            self.assertEqual(
                worklist,
                [
                    # first transfer
                    "A;A;;;1;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                    "A;A;;;4;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                    "A;A;;;7;;25.00;;;;",
                    "D;B;;;2;;25.00;;;;",
                    "W1;",
                    # second transfer
                    "A;B;;;7;;50.00;;;;",
                    "D;A;;;10;;50.00;;;;",
                    "W1;",
                    "A;B;;;8;;60.00;;;;",
                    "D;A;;;11;;60.00;;;;",
                    "W1;",
                    "A;B;;;9;;70.00;;;;",
                    "D;A;;;12;;70.00;;;;",
                    "W1;",
                ],
            )
            self.assertEqual(len(A.history), 3)
            self.assertEqual(len(B.history), 3)
        return


class TestLargeVolumeHandling(unittest.TestCase):
    def test_partition_volume_helper(self) -> None:
        self.assertEqual([], evotools._partition_volume(0, max_volume=950))
        self.assertEqual([550.3], evotools._partition_volume(550.3, max_volume=950))
        self.assertEqual([500, 500], evotools._partition_volume(1000, max_volume=950))
        self.assertEqual([500, 499], evotools._partition_volume(999, max_volume=950))
        self.assertEqual([667, 667, 666], evotools._partition_volume(2000, max_volume=950))
        return

    def test_worklist_constructor(self) -> None:
        with self.assertRaises(ValueError):
            with evotools.Worklist(max_volume=None) as wl:
                pass
        with evotools.Worklist(max_volume=800, auto_split=True) as wl:
            self.assertEqual(wl.max_volume, 800)
            self.assertEqual(wl.auto_split, True)
        with evotools.Worklist(max_volume=800, auto_split=False) as wl:
            self.assertEqual(wl.max_volume, 800)
            self.assertEqual(wl.auto_split, False)
        return

    def test_max_volume_checking(self) -> None:
        source = liquidhandling.Trough(
            "WaterTrough",
            virtual_rows=3,
            columns=3,
            min_volume=1000,
            max_volume=100 * 1000,
            initial_volumes=50 * 1000,
        )
        destination = liquidhandling.Trough(
            "WaterTrough",
            virtual_rows=3,
            columns=3,
            min_volume=1000,
            max_volume=100 * 1000,
            initial_volumes=50 * 1000,
        )
        with evotools.Worklist(max_volume=900, auto_split=False) as wl:
            with self.assertRaises(evotools.InvalidOperationError):
                wl.aspirate_well("WaterTrough", 1, 1000)
            with self.assertRaises(evotools.InvalidOperationError):
                wl.dispense_well("WaterTrough", 1, 1000)
            with self.assertRaises(evotools.InvalidOperationError):
                wl.aspirate(source, ["A01", "A02", "C02"], 1000)
            with self.assertRaises(evotools.InvalidOperationError):
                wl.dispense(source, ["A01", "A02", "C02"], 1000)
            with self.assertRaises(evotools.InvalidOperationError):
                wl.transfer(source, ["A01", "B01"], destination, ["A01", "B01"], 1000)

        source = liquidhandling.Trough(
            "WaterTrough",
            virtual_rows=3,
            columns=3,
            min_volume=1000,
            max_volume=100 * 1000,
            initial_volumes=50 * 1000,
        )
        destination = liquidhandling.Trough(
            "WaterTrough",
            virtual_rows=3,
            columns=3,
            min_volume=1000,
            max_volume=100 * 1000,
            initial_volumes=50 * 1000,
        )
        with evotools.Worklist(max_volume=1200) as wl:
            wl.aspirate_well("WaterTrough", 1, 1000)
            wl.dispense_well("WaterTrough", 1, 1000)
            wl.aspirate(source, ["A01", "A02", "C02"], 1000)
            wl.dispense(source, ["A01", "A02", "C02"], 1000)
            wl.transfer(source, ["A01", "B01"], destination, ["A01", "B01"], 1000)
        return

    def test_partition_by_columns_source(self) -> None:
        column_groups = evotools._partition_by_column(
            ["A01", "B01", "A03", "B03", "C02"],
            ["A01", "B01", "C01", "D01", "E01"],
            [2500, 3500, 1000, 500, 2000],
            partition_by="source",
        )
        self.assertEqual(len(column_groups), 3)
        self.assertEqual(
            column_groups[0],
            (
                ["A01", "B01"],
                ["A01", "B01"],
                [2500, 3500],
            ),
        )
        self.assertEqual(
            column_groups[1],
            (
                ["C02"],
                ["E01"],
                [2000],
            ),
        )
        self.assertEqual(
            column_groups[2],
            (
                ["A03", "B03"],
                ["C01", "D01"],
                [1000, 500],
            ),
        )
        return

    def test_partition_by_columns_destination(self) -> None:
        column_groups = evotools._partition_by_column(
            ["A01", "B01", "A03", "B03", "C02"],
            ["A01", "B01", "C02", "D01", "E02"],
            [2500, 3500, 1000, 500, 2000],
            partition_by="destination",
        )
        self.assertEqual(len(column_groups), 2)
        self.assertEqual(
            column_groups[0],
            (
                ["A01", "B01", "B03"],
                ["A01", "B01", "D01"],
                [2500, 3500, 500],
            ),
        )
        self.assertEqual(
            column_groups[1],
            (
                ["A03", "C02"],
                ["C02", "E02"],
                [1000, 2000],
            ),
        )
        return

    def test_partition_by_columns_sorting(self) -> None:
        # within every column, the wells are supposed to be sorted by row
        # The test source wells are partially sorted (col 1 is in the right order, col 3 in the reverse)
        # The result is expected to always be sorted by row, either in the source (first case) or destination:

        # by source
        column_groups = evotools._partition_by_column(
            ["A01", "B01", "B03", "A03", "C02"],
            ["B01", "A01", "C01", "D01", "E01"],
            [2500, 3500, 1000, 500, 2000],
            partition_by="source",
        )
        self.assertEqual(len(column_groups), 3)
        self.assertEqual(
            column_groups[0],
            (
                ["A01", "B01"],
                ["B01", "A01"],
                [2500, 3500],
            ),
        )
        self.assertEqual(
            column_groups[1],
            (
                ["C02"],
                ["E01"],
                [2000],
            ),
        )
        self.assertEqual(
            column_groups[2],
            (
                ["A03", "B03"],
                ["D01", "C01"],
                [500, 1000],
            ),
        )

        # by destination
        # (destination wells are across 3 columns; reverse order in col 1, forward order in col 3)
        column_groups = evotools._partition_by_column(
            ["A01", "B01", "B03", "A03", "C02"],
            ["B01", "A01", "C03", "D03", "E02"],
            [2500, 3500, 1000, 500, 2000],
            partition_by="destination",
        )
        self.assertEqual(len(column_groups), 3)
        self.assertEqual(
            column_groups[0],
            (
                ["B01", "A01"],
                ["A01", "B01"],
                [3500, 2500],
            ),
        )
        self.assertEqual(
            column_groups[1],
            (
                ["C02"],
                ["E02"],
                [2000],
            ),
        )
        self.assertEqual(
            column_groups[2],
            (
                ["B03", "A03"],
                ["C03", "D03"],
                [1000, 500],
            ),
        )
        return

    def test_single_split(self) -> None:
        src = liquidhandling.Labware("A", 3, 2, min_volume=1000, max_volume=25000, initial_volumes=12000)
        dst = liquidhandling.Labware("B", 3, 2, min_volume=1000, max_volume=25000)
        with evotools.Worklist(auto_split=True) as wl:
            wl.transfer(src, "A01", dst, "A01", 2000, label="Transfer more than 2x the max")
            self.assertEqual(
                wl,
                [
                    "C;Transfer more than 2x the max",
                    "A;A;;;1;;667.00;;;;",
                    "D;B;;;1;;667.00;;;;",
                    "W1;",
                    # no breaks when pipetting single wells
                    "A;A;;;1;;667.00;;;;",
                    "D;B;;;1;;667.00;;;;",
                    "W1;",
                    # no breaks when pipetting single wells
                    "A;A;;;1;;666.00;;;;",
                    "D;B;;;1;;666.00;;;;",
                    "W1;",
                    "B;",  # always break after partitioning
                ],
            )
        # Two extra steps were necessary because of LVH
        assert "Transfer more than 2x the max (2 LVH steps)" in src.report
        assert "Transfer more than 2x the max (2 LVH steps)" in dst.report
        numpy.testing.assert_array_equal(
            src.volumes,
            [
                [12000 - 2000, 12000],
                [12000, 12000],
                [12000, 12000],
            ],
        )
        numpy.testing.assert_array_equal(
            dst.volumes,
            [
                [2000, 0],
                [0, 0],
                [0, 0],
            ],
        )
        return

    def test_column_split(self) -> None:
        src = liquidhandling.Labware("A", 4, 2, min_volume=1000, max_volume=25000, initial_volumes=12000)
        dst = liquidhandling.Labware("B", 4, 2, min_volume=1000, max_volume=25000)
        with evotools.Worklist(auto_split=True) as wl:
            wl.transfer(
                src, ["A01", "B01", "D01", "C01"], dst, ["A01", "B01", "D01", "C01"], [1500, 250, 0, 1200]
            )
            self.assertEqual(
                wl,
                [
                    "A;A;;;1;;750.00;;;;",
                    "D;B;;;1;;750.00;;;;",
                    "W1;",
                    "A;A;;;2;;250.00;;;;",
                    "D;B;;;2;;250.00;;;;",
                    "W1;",
                    # D01 is ignored because the volume is 0
                    "A;A;;;3;;600.00;;;;",
                    "D;B;;;3;;600.00;;;;",
                    "W1;",
                    "B;",  # within-column break
                    "A;A;;;1;;750.00;;;;",
                    "D;B;;;1;;750.00;;;;",
                    "W1;",
                    "A;A;;;3;;600.00;;;;",
                    "D;B;;;3;;600.00;;;;",
                    "W1;",
                    "B;",  # tailing break after partitioning
                ],
            )
        numpy.testing.assert_array_equal(
            src.volumes,
            [
                [12000 - 1500, 12000],
                [12000 - 250, 12000],
                [12000 - 1200, 12000],
                [12000, 12000],
            ],
        )
        numpy.testing.assert_array_equal(
            dst.volumes,
            [
                [1500, 0],
                [250, 0],
                [1200, 0],
                [0, 0],
            ],
        )
        return

    def test_block_split(self) -> None:
        src = liquidhandling.Labware("A", 3, 2, min_volume=1000, max_volume=25000, initial_volumes=12000)
        dst = liquidhandling.Labware("B", 3, 2, min_volume=1000, max_volume=25000)
        with evotools.Worklist(auto_split=True) as wl:
            wl.transfer(
                # A01, B01, A02, B02
                src,
                src.wells[:2, :],
                dst,
                ["A01", "B01", "C01", "A02"],
                [1500, 250, 1200, 3000],
            )
            self.assertEqual(
                wl,
                [
                    "A;A;;;1;;750.00;;;;",
                    "D;B;;;1;;750.00;;;;",
                    "W1;",
                    "A;A;;;2;;250.00;;;;",
                    "D;B;;;2;;250.00;;;;",
                    "W1;",
                    "B;",  # within-column 1 break
                    "A;A;;;1;;750.00;;;;",
                    "D;B;;;1;;750.00;;;;",
                    "W1;",
                    "B;",  # between-column 1/2 break
                    "A;A;;;4;;600.00;;;;",
                    "D;B;;;3;;600.00;;;;",
                    "W1;",
                    "A;A;;;5;;750.00;;;;",
                    "D;B;;;4;;750.00;;;;",
                    "W1;",
                    "B;",  # within-column 2 break
                    "A;A;;;4;;600.00;;;;",
                    "D;B;;;3;;600.00;;;;",
                    "W1;",
                    "A;A;;;5;;750.00;;;;",
                    "D;B;;;4;;750.00;;;;",
                    "W1;",
                    "B;",  # within-column 2 break
                    "A;A;;;5;;750.00;;;;",
                    "D;B;;;4;;750.00;;;;",
                    "W1;",
                    # no break because only one well is accessed in this partition
                    "A;A;;;5;;750.00;;;;",
                    "D;B;;;4;;750.00;;;;",
                    "W1;",
                    "B;",  # tailing break after partitioning
                ],
            )

        # How the number of splits is calculated:
        # 1500 is split 2x → 1 extra
        # 250 is not split
        # 1200 is split 2x → 1 extra
        # 3000 is split 4x → 3 extra
        # Sum of extra steps: 5
        assert "5 LVH steps" in src.report
        assert "5 LVH steps" in dst.report
        numpy.testing.assert_array_equal(
            src.volumes,
            [
                [12000 - 1500, 12000 - 1200],
                [12000 - 250, 12000 - 3000],
                [12000, 12000],
            ],
        )
        numpy.testing.assert_array_equal(
            dst.volumes,
            [
                [1500, 3000],
                [250, 0],
                [1200, 0],
            ],
        )
        return


class TestReagentDistribution(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with evotools.Worklist() as wl:
            with self.assertRaises(ValueError):
                wl.reagent_distribution("S1", 1, 8, "D1", 1, 20, volume=50, direction="invalid")
            with self.assertRaises(ValueError):
                # one excluded well not in the destination range
                wl.reagent_distribution("S1", 1, 8, "D1", 1, 20, volume=50, exclude_wells=[18, 19, 23])
        with self.assertRaises(evotools.InvalidOperationError):
            evo_logger = logging.getLogger("evotools")
            with self.assertLogs(logger=evo_logger, level=logging.WARNING):
                with evotools.Worklist(max_volume=950) as wl:
                    # dispense more than diluter volume
                    wl.reagent_distribution("S1", 1, 8, "D1", 1, 20, volume=1200)
        return

    def test_default_parameterization(self) -> None:
        with evotools.Worklist() as wl:
            wl.reagent_distribution("S1", 1, 20, "D1", 2, 21, volume=50)
        self.assertEqual(wl[0], "R;S1;;;1;20;D1;;;2;21;50;;1;1;0")
        return

    def test_full_parameterization(self) -> None:
        with evotools.Worklist() as wl:
            wl.reagent_distribution(
                "S1",
                1,
                20,
                "D1",
                2,
                21,
                volume=50,
                diti_reuse=2,
                multi_disp=3,
                exclude_wells=[2, 4, 8],
                liquid_class="TestLC",
                direction="right_to_left",
                src_rack_type="MP3Pos",
                dst_rack_type="MP4Pos",
                src_rack_id="S1234",
                dst_rack_id="D1234",
            )
        self.assertEqual(wl[0], "R;S1;S1234;MP3Pos;1;20;D1;D1234;MP4Pos;2;21;50;TestLC;2;3;1;2;4;8")
        return

    def test_large_volume_multi_disp_adaption(self) -> None:
        with evotools.Worklist() as wl:
            wl.reagent_distribution(
                "S1",
                1,
                8,
                "D1",
                1,
                96,
                volume=400,
                multi_disp=6,
            )
        self.assertEqual(wl[0], "R;S1;;;1;8;D1;;;1;96;400;;1;2;0")
        return

    def test_oo_parameter_validation(self) -> None:
        with evotools.Worklist() as wl:
            src = liquidhandling.Labware("NotATrough", 6, 2, min_volume=20, max_volume=1000)
            dst = liquidhandling.Labware("48deep", 6, 8, min_volume=50, max_volume=4000)
            with self.assertRaises(ValueError):
                wl.distribute(src, 0, dst, dst.wells[:, :3], volume=50)
        with evotools.Worklist(max_volume=950) as wl:
            src = liquidhandling.Trough("Water", 8, 2, min_volume=20, max_volume=1000)
            dst = liquidhandling.Labware("48deep", 6, 8, min_volume=50, max_volume=4000)
            with self.assertRaises(evotools.InvalidOperationError):
                wl.distribute(src, 0, dst, dst.wells[:, :3], volume=1200)
        return

    def test_oo_example_1(self) -> None:
        src = liquidhandling.Trough(
            "T3",
            8,
            1,
            min_volume=20,
            max_volume=100 * 1000,
            initial_volumes=100 * 1000,
        )
        dst = liquidhandling.Labware("MTP-96-3", 8, 12, min_volume=20, max_volume=300)
        with evotools.Worklist() as wl:
            all_dst_wells = set(dst.wells.flatten("F"))
            skip_wells = set("C04,C07,E09,F06,B11".split(","))
            dst_wells = list(all_dst_wells.difference(skip_wells))
            wl.distribute(
                src,
                0,
                dst,
                dst_wells,
                volume=100,
                liquid_class="Water",
                diti_reuse=1,
                multi_disp=6,
                direction="left_to_right",
                src_rack_type="Trough 100ml",
                dst_rack_type="96 Well Microplate",
                label="Test Label",
            )
        self.assertEqual(wl[0], "C;Test Label")
        self.assertEqual(
            wl[1], "R;T3;;Trough 100ml;1;8;MTP-96-3;;96 Well Microplate;1;96;100;Water;1;6;0;27;46;51;69;82"
        )
        self.assertEqual(src.volumes[0, 0], 100 * 1000 - 91 * 100)
        dst_exp = numpy.ones_like(dst.volumes) * 100
        dst_exp[dst.indices["C04"]] = 0
        dst_exp[dst.indices["C07"]] = 0
        dst_exp[dst.indices["E09"]] = 0
        dst_exp[dst.indices["F06"]] = 0
        dst_exp[dst.indices["B11"]] = 0
        numpy.testing.assert_array_equal(dst.volumes, dst_exp)
        return

    def test_oo_example_2(self) -> None:
        src = liquidhandling.Trough(
            "T2",
            8,
            1,
            min_volume=20,
            max_volume=100 * 1000,
            initial_volumes=100 * 1000,
        )
        dst = liquidhandling.Labware("MTP-96-2", 8, 12, min_volume=20, max_volume=300)
        with evotools.Worklist() as wl:
            wl.distribute(
                src,
                0,
                dst,
                dst.wells,
                volume=100,
                liquid_class="Water",
                diti_reuse=2,
                multi_disp=5,
                direction="left_to_right",
                src_rack_type="Trough 100ml",
                dst_rack_type="96 Well Microplate",
                label="Test Label",
            )
        self.assertEqual(wl[0], "C;Test Label")
        self.assertEqual(wl[1], "R;T2;;Trough 100ml;1;8;MTP-96-2;;96 Well Microplate;1;96;100;Water;2;5;0")
        self.assertEqual(src.volumes[0, 0], 100 * 1000 - 96 * 100)
        numpy.testing.assert_array_equal(dst.volumes, numpy.ones_like(dst.volumes) * 100)
        return

    def test_oo_block_from_right(self) -> None:
        src = liquidhandling.Trough(
            "Water",
            8,
            1,
            min_volume=20,
            max_volume=100 * 1000,
            initial_volumes=100 * 1000,
        )
        dst = liquidhandling.Labware("96mtp", 8, 12, min_volume=20, max_volume=300)
        with evotools.Worklist() as wl:
            wl.distribute(
                src,
                0,
                dst,
                dst.wells[1:4, 2:7],
                volume=50,
                liquid_class="TestLC",
                diti_reuse=10,
                multi_disp=5,
                direction="right_to_left",
                label="Test Label",
            )
        self.assertEqual(wl[0], "C;Test Label")
        skip_pos = ";21;22;23;24;25;29;30;31;32;33;37;38;39;40;41;45;46;47;48;49"
        self.assertEqual(wl[1], f"R;Water;;;1;8;96mtp;;;18;52;50;TestLC;10;5;1{skip_pos}")
        self.assertEqual(src.volumes[0, 0], 100 * 1000 - 15 * 50)
        self.assertTrue(numpy.all(dst.volumes[1:4, 2:7] == 50))
        return


class TestCompositionTracking(unittest.TestCase):
    def test_get_initial_composition(self) -> None:
        wells2x3 = numpy.array(
            [
                ["A01", "A02", "A03"],
                ["B01", "B02", "B03"],
            ]
        )

        # Raise errors on invalid component well ids
        with pytest.raises(ValueError, match=r"Invalid component name keys: \{'G02'\}"):
            liquidhandling._get_initial_composition("eppis", wells2x3, dict(G02="beer"), numpy.zeros((2, 3)))

        # Raise errors on attempts to name empty wells
        with pytest.raises(ValueError, match=r"name 'beer' was specified for eppis.A02, but"):
            liquidhandling._get_initial_composition("eppis", wells2x3, dict(A02="beer"), numpy.zeros((2, 3)))

        # No components if all wells are empty
        result = liquidhandling._get_initial_composition("eppis", wells2x3, {}, numpy.zeros((2, 3)))
        assert result == {}

        # Default to labware name for one-well labwares
        result = liquidhandling._get_initial_composition("media", [["A01"]], {}, numpy.atleast_2d(100))
        assert "media" in result
        assert len(result) == 1

        # Assigning default component names to all wells
        result = liquidhandling._get_initial_composition("samples", wells2x3, {}, numpy.ones((2, 3)))
        assert isinstance(result, dict)
        # Non-empty wells default to unique component names
        assert "samples.A01" in result
        assert "samples.B03" in result
        # Only the well with the component has 100 % of it
        numpy.testing.assert_array_equal(
            result["samples.B02"],
            numpy.array(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                ]
            ),
        )

        # Mix of user-defined and default component names
        result = liquidhandling._get_initial_composition(
            "samples", wells2x3, {"B01": "water"}, numpy.ones((2, 3))
        )
        assert isinstance(result, dict)
        # Non-empty wells default to unique component names
        assert "samples.A01" in result
        assert "water" in result
        assert "samples.B03" in result
        return

    def test_get_trough_component_names(self) -> None:
        # The function requies the correct number of column names and initial volumes
        # and should raise informative errors otherwise.
        with pytest.raises(ValueError, match=r"column names \['A', 'B', 'C'\] don't match"):
            liquidhandling._get_trough_component_names("water", 2, ["A", "B", "C"], [20, 10])
        with pytest.raises(ValueError, match=r"initial volumes \[20, 10\] don't match"):
            liquidhandling._get_trough_component_names("water", 3, ["A", "B", "C"], [20, 10])
        with pytest.raises(ValueError, match=r"initial volumes \[\[20], \[10\]\] don't match"):
            liquidhandling._get_trough_component_names("water", 2, ["A", "B"], [[20], [10]])

        # It should also check that no names are given for empty columns
        with pytest.raises(ValueError, match="Empty columns must be unnamed"):
            liquidhandling._get_trough_component_names("water", 2, ["A", "B"], [20, 0])

        # It explicitly sets names of empty wells to None
        result = liquidhandling._get_trough_component_names("water", 1, [None], [0])
        assert result == {"A01": None}

        # And defaults to the labware name of single-column troughs
        result = liquidhandling._get_trough_component_names("water", 1, [None], [100])
        assert result == {"A01": "water"}

        # But includes the 1-based column number in the default name for non-empty wells
        result = liquidhandling._get_trough_component_names("stocks", 2, [None, None], [0, 50])
        assert result == {"A01": None, "A02": "stocks.column_02"}

        # User-provided names, default naming and empty-well all in one:
        result = liquidhandling._get_trough_component_names(
            "stocks", 4, ["acid", "base", None, None], [100, 100, 50, 0]
        )
        assert result == {"A01": "acid", "A02": "base", "A03": "stocks.column_03", "A04": None}
        return

    def test_combine_composition(self) -> None:
        A = dict(water=1)
        B = dict(water=0.5, glucose=0.5)
        expected = {"water": (1 * 10 + 0.5 * 15) / (10 + 15), "glucose": 0.5 * 15 / (10 + 15)}
        actual = liquidhandling._combine_composition(10, A, 15, B)
        self.assertDictEqual(actual, expected)
        return

    def test_combine_unknown_composition(self) -> None:
        A = dict(water=1)
        B = None
        expected = None
        actual = liquidhandling._combine_composition(10, A, 15, B)
        self.assertEqual(actual, expected)
        return

    def test_labware_init(self) -> None:
        minmax = dict(min_volume=0, max_volume=4000)
        # without initial volume, there's no composition tracking
        A = liquidhandling.Labware("glc", 6, 8, **minmax)
        self.assertIsInstance(A.composition, dict)
        self.assertEqual(len(A.composition), 0)
        self.assertDictEqual(A.get_well_composition("A01"), {})

        # Single-well Labware defaults to the labware name for components
        A = liquidhandling.Labware("x", 1, 1, **minmax, initial_volumes=300)
        assert set(A.composition) == {"x"}
        A = liquidhandling.Labware("x", 1, 1, virtual_rows=3, **minmax, initial_volumes=300)
        assert set(A.composition) == {"x"}

        # by setting an initial volume, the well-wise liquids take part in composition tracking
        A = liquidhandling.Labware("glc", 6, 8, **minmax, initial_volumes=100)
        self.assertIsInstance(A.composition, dict)
        self.assertEqual(len(A.composition), 48)
        self.assertIn("glc.A01", A.composition)
        self.assertIn("glc.F08", A.composition)
        self.assertDictEqual(A.get_well_composition("A01"), {"glc.A01": 1})

        # Only wells with initial volumes take part
        A = liquidhandling.Labware(
            "test", 1, 3, **minmax, initial_volumes=[10, 0, 0], component_names=dict(A01="water")
        )
        assert set(A.composition) == {"water"}
        return

    def test_get_well_composition(self) -> None:
        A = liquidhandling.Labware("glc", 6, 8, min_volume=0, max_volume=4000)
        A._composition = {
            "glc": 0.25 * numpy.ones_like(A.volumes),
            "water": 0.75 * numpy.ones_like(A.volumes),
        }
        self.assertDictEqual(
            A.get_well_composition("A01"),
            {
                "glc": 0.25,
                "water": 0.75,
            },
        )
        return

    def test_labware_add(self) -> None:
        A = liquidhandling.Labware(
            "water",
            6,
            8,
            min_volume=0,
            max_volume=4000,
            initial_volumes=10,
            component_names={
                "A01": "water",
                "B01": "water",
            },
        )
        water_comp = numpy.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        numpy.testing.assert_array_equal(A.composition["water"], water_comp)

        A.add(
            wells=["A01", "B01"],
            volumes=[10, 20],
            compositions=[
                dict(glc=0.5, water=0.5),
                dict(glc=1),
            ],
        )

        self.assertIn("water", A.composition)
        self.assertIn("glc", A.composition)
        self.assertDictEqual(A.get_well_composition("A01"), dict(water=0.75, glc=0.25))
        self.assertDictEqual(A.get_well_composition("B01"), dict(water=1 / 3, glc=2 / 3))
        return

    def test_dilution_series(self) -> None:
        A = liquidhandling.Labware("dilutions", 1, 3, min_volume=0, max_volume=100)
        # 100 % in 1st column
        A.add(wells="A01", volumes=100, compositions=[dict(glucose=1)])
        # 10x dilution to 2nd
        A.add(wells="A02", volumes=10, compositions=[A.get_well_composition("A01")])
        A.add(wells="A02", volumes=90, compositions=[dict(water=1)])
        # 4x dilution to 3rd
        A.add(wells="A03", volumes=2.5, compositions=[A.get_well_composition("A02")])
        A.add(wells="A03", volumes=7.5, compositions=[dict(water=1)])

        numpy.testing.assert_array_equal(A.volumes, [[100, 100, 10]])
        numpy.testing.assert_array_equal(A.composition["water"], [[0, 0.9, 0.975]])
        numpy.testing.assert_array_equal(A.composition["glucose"], [[1, 0.1, 0.025]])
        self.assertDictEqual(A.get_well_composition("A01"), dict(glucose=1))
        self.assertDictEqual(A.get_well_composition("A02"), dict(glucose=0.1, water=0.9))
        self.assertDictEqual(A.get_well_composition("A03"), dict(glucose=0.025, water=0.975))
        return

    def test_trough_init(self) -> None:
        minmax = dict(min_volume=0, max_volume=100_000)
        # Single-column troughs use the labware name for the composition
        W = liquidhandling.Trough("water", 2, 1, **minmax, initial_volumes=10000)
        assert set(W.composition) == {"water"}

        # Multi-column troughs get component names automatically
        A = liquidhandling.Trough(
            name="water", virtual_rows=2, columns=3, **minmax, initial_volumes=[0, 200, 200]
        )
        assert set(A.composition) == {"water.column_02", "water.column_03"}
        assert "water.column_01" not in A.composition
        assert "water.column_02" in A.composition
        assert "water.column_03" in A.composition

        # Components in troughs are named via a list, because the dict would
        # require well names and columns could lead to 0/1-based confusion.
        T = liquidhandling.Trough(
            "stocks",
            6,
            3,
            min_volume=0,
            max_volume=10_000,
            initial_volumes=[100, 200, 0],
            column_names=["rich", "complex", None],
        )
        assert set(T.composition.keys()) == {"rich", "complex"}

        # Naming just some of them works too
        A = liquidhandling.Trough(
            name="alice",
            virtual_rows=2,
            columns=3,
            **minmax,
            initial_volumes=[0, 200, 200],
            column_names=[None, "NaCl", None],
        )
        assert set(A.composition) == {"NaCl", "alice.column_03"}
        numpy.testing.assert_array_equal(A.composition["NaCl"], [[0, 1, 0]])

    def test_trough_composition(self) -> None:
        T = liquidhandling.Trough("media", 8, 1, min_volume=1000, max_volume=25000)
        T.add(wells=T.wells, volumes=100, compositions=[dict(glucose=1)] * 8)
        T.add(wells=T.wells, volumes=900, compositions=[dict(water=1)] * 8)
        numpy.testing.assert_array_equal(T.composition["water"], [[0.9]])
        numpy.testing.assert_array_equal(T.composition["glucose"], [[0.1]])
        self.assertDictEqual(T.get_well_composition("B01"), dict(water=0.9, glucose=0.1))
        return

    def test_worklist_dilution(self) -> None:
        W = liquidhandling.Trough("water", 4, 1, min_volume=0, max_volume=10000, initial_volumes=10000)
        G = liquidhandling.Trough("glucose", 4, 1, min_volume=0, max_volume=10000, initial_volumes=10000)
        D = liquidhandling.Labware("dilutions", 4, 2, min_volume=0, max_volume=10000)

        with evotools.Worklist() as wl:
            # 100 % in first column
            wl.transfer(G, G.wells, D, D.wells[:, 0], volumes=[1000, 800, 600, 550])
            wl.transfer(W, W.wells, D, D.wells[:, 0], volumes=[0, 200, 400, 450])
            numpy.testing.assert_array_equal(D.volumes[:, 0], [1000, 1000, 1000, 1000])
            numpy.testing.assert_array_equal(D.composition["glucose"][:, 0], [1, 0.8, 0.6, 0.55])
            numpy.testing.assert_array_equal(D.composition["water"][:, 0], [0, 0.2, 0.4, 0.45])
            # 10x dilution to the 2nd column
            wl.transfer(D, D.wells[:, 0], D, D.wells[:, 1], volumes=100)
            wl.transfer(W, W.wells, D, D.wells[:, 1], volumes=900)
            numpy.testing.assert_array_equal(D.volumes[:, 1], [1000, 1000, 1000, 1000])
            numpy.testing.assert_allclose(D.composition["glucose"][:, 1], [0.1, 0.08, 0.06, 0.055])
            numpy.testing.assert_allclose(D.composition["water"][:, 1], [0.9, 0.92, 0.94, 0.945])

        return

    def test_worklist_distribution(self) -> None:
        W = liquidhandling.Trough("water", 2, 1, min_volume=0, max_volume=10000, initial_volumes=10000)
        G = liquidhandling.Trough("glucose", 2, 1, min_volume=0, max_volume=10000, initial_volumes=10000)
        D = liquidhandling.Labware("dilutions", 2, 4, min_volume=0, max_volume=10000)

        with evotools.Worklist() as wl:
            # transfer some glucose
            wl.transfer(
                G,
                G.wells[:, [0] * 4],
                D,
                D.wells,
                volumes=numpy.array(
                    [
                        [100, 80, 60, 55],
                        [55, 60, 80, 100],
                    ]
                ),
            )
            # fill up to 100
            wl.transfer(W, W.wells[:, [0] * 4], D, D.wells, volumes=100 - D.volumes)
            numpy.testing.assert_allclose(D.volumes, 100)
            numpy.testing.assert_array_equal(D.composition["glucose"][0, :], [1, 0.8, 0.6, 0.55])
            numpy.testing.assert_array_equal(D.composition["glucose"][1, :], [0.55, 0.6, 0.8, 1])
            numpy.testing.assert_array_equal(D.composition["water"][0, :], [0, 0.2, 0.4, 0.45])
            numpy.testing.assert_array_equal(D.composition["water"][1, :], [0.45, 0.4, 0.2, 0])
            # dilute 2x with water
            wl.distribute(source=W, source_column=0, destination=D, destination_wells=D.wells, volume=100)
            numpy.testing.assert_allclose(D.volumes, 200)
            numpy.testing.assert_array_equal(D.composition["glucose"][0, :], [0.5, 0.4, 0.3, 0.275])
            numpy.testing.assert_array_equal(D.composition["glucose"][1, :], [0.275, 0.3, 0.4, 0.5])
            numpy.testing.assert_array_equal(D.composition["water"][0, :], [0.5, 0.6, 0.7, 0.725])
            numpy.testing.assert_array_equal(D.composition["water"][1, :], [0.725, 0.7, 0.6, 0.5])

        return

    def test_worklist_mix_no_composition_change(self) -> None:
        A = liquidhandling.Labware("solution", 2, 3, min_volume=0, max_volume=1000)
        A._composition["water"] = 0.25 * numpy.ones_like(A.volumes)
        A._composition["salt"] = 0.75 * numpy.ones_like(A.volumes)
        A._volumes = numpy.ones_like(A.volumes) * 500
        with evotools.Worklist() as wl:
            wl.transfer(A, A.wells, A, A.wells, volumes=300)
        # make sure that the composition of the liquid is not changed
        numpy.testing.assert_array_equal(A.composition["water"], 0.25 * numpy.ones_like(A.volumes))
        numpy.testing.assert_array_equal(A.composition["salt"], 0.75 * numpy.ones_like(A.volumes))
        return


class TestFunctions(unittest.TestCase):
    def test_automatic_partitioning(self) -> None:
        evo_logger = logging.getLogger("evotools")
        S = liquidhandling.Labware("S", 8, 2, min_volume=5000, max_volume=250 * 1000)
        D = liquidhandling.Labware("D", 8, 2, min_volume=5000, max_volume=250 * 1000)
        ST = liquidhandling.Trough("ST", 8, 2, min_volume=5000, max_volume=250 * 1000)
        DT = liquidhandling.Trough("DT", 8, 2, min_volume=5000, max_volume=250 * 1000)

        # Expected behaviors:
        # + always keep settings other than 'auto'
        # + warn user about inefficient configuration (when user selects to partition by the trough)

        # automatic
        self.assertEqual("source", evotools._optimize_partition_by(S, D, "auto", "No troughs at all"))
        self.assertEqual("source", evotools._optimize_partition_by(S, DT, "auto", "Trough destination"))
        self.assertEqual("destination", evotools._optimize_partition_by(ST, D, "auto", "Trough source"))
        self.assertEqual(
            "source", evotools._optimize_partition_by(ST, DT, "auto", "Trough source and destination")
        )

        # fixed to source
        self.assertEqual("source", evotools._optimize_partition_by(S, D, "source", "No troughs at all"))
        self.assertEqual("source", evotools._optimize_partition_by(S, DT, "source", "Trough destination"))
        with self.assertLogs(logger=evo_logger, level=logging.WARNING):
            self.assertEqual("source", evotools._optimize_partition_by(ST, D, "source", "Trough source"))
        self.assertEqual(
            "source", evotools._optimize_partition_by(ST, DT, "auto", "Trough source and destination")
        )

        # fixed to destination
        self.assertEqual(
            "destination", evotools._optimize_partition_by(S, D, "destination", "No troughs at all")
        )
        with self.assertLogs(logger=evo_logger, level=logging.WARNING):
            self.assertEqual(
                "destination", evotools._optimize_partition_by(S, DT, "destination", "Trough destination")
            )
        self.assertEqual(
            "destination", evotools._optimize_partition_by(ST, D, "destination", "Trough source")
        )
        self.assertEqual(
            "destination",
            evotools._optimize_partition_by(ST, DT, "destination", "Trough source and destination"),
        )
        return


class TestDilutionPlan(unittest.TestCase):
    def test_argchecking(self) -> None:
        with self.assertRaises(ValueError):
            robotools.DilutionPlan(
                xmin=0.001, xmax=30, R=8, C=12, stock=20, mode="log", vmax=1000, min_transfer=20
            )

        with self.assertRaises(ValueError):
            robotools.DilutionPlan(
                xmin=0.001, xmax=30, R=8, C=12, stock=30, mode="invalid", vmax=1000, min_transfer=20
            )

        with self.assertRaises(ValueError):
            robotools.DilutionPlan(
                xmin=0.001, xmax=30, R=6, C=4, stock=30, mode="linear", vmax=1000, min_transfer=20
            )

        with self.assertRaises(ValueError):
            robotools.DilutionPlan(
                xmin=0.001,
                xmax=30,
                R=6,
                C=4,
                stock=30,
                mode="linear",
                vmax=[1000, 1000, 1000],
                min_transfer=20,
            )

        return

    def test_repr(self) -> None:
        plan = robotools.DilutionPlan(
            xmin=0.001, xmax=30, R=8, C=12, stock=30, mode="log", vmax=1000, min_transfer=20
        )

        out = plan.__repr__()

        self.assertIsNotNone(out)
        self.assertIsInstance(out, str)
        return

    def test_linear_plan(self) -> None:
        plan = robotools.DilutionPlan(
            xmin=1, xmax=10, R=10, C=1, stock=20, mode="linear", vmax=1000, min_transfer=20
        )

        numpy.testing.assert_array_equal(plan.x, plan.ideal_x)
        self.assertEqual(plan.max_steps, 0)
        self.assertEqual(plan.v_stock, 2750)
        self.assertEqual(plan.instructions[0][0], 0)
        self.assertEqual(plan.instructions[0][1], 0)
        self.assertEqual(plan.instructions[0][2], "stock")
        numpy.testing.assert_array_equal(
            plan.instructions[0][3],
            [
                500,
                450,
                400,
                350,
                300,
                250,
                200,
                150,
                100,
                50,
            ],
        )
        return

    def test_log_plan(self) -> None:
        plan = robotools.DilutionPlan(
            xmin=0.01, xmax=10, R=4, C=3, stock=20, mode="log", vmax=1000, min_transfer=20
        )

        self.assertTrue(numpy.allclose(plan.x, plan.ideal_x, rtol=0.05))
        self.assertEqual(plan.max_steps, 2)
        self.assertEqual(plan.v_stock, 985)
        self.assertEqual(plan.instructions[0][0], 0)
        self.assertEqual(plan.instructions[0][1], 0)
        self.assertEqual(plan.instructions[0][2], "stock")
        numpy.testing.assert_array_equal(plan.instructions[0][3], [500, 267, 142, 76])
        numpy.testing.assert_array_equal(plan.instructions[1][3], [82, 82, 82, 82])
        numpy.testing.assert_array_equal(plan.instructions[2][3], [81, 81, 81, 81])
        return

    def test_vector_vmax(self) -> None:
        plan = robotools.DilutionPlan(
            xmin=0.01, xmax=10, R=4, C=3, stock=20, mode="log", vmax=[1000, 500, 1500], min_transfer=20
        )

        self.assertTrue(numpy.allclose(plan.x, plan.ideal_x, rtol=0.05))
        self.assertEqual(plan.max_steps, 2)
        self.assertEqual(plan.v_stock, 985)
        self.assertEqual(plan.instructions[0][0], 0)
        self.assertEqual(plan.instructions[0][1], 0)
        self.assertEqual(plan.instructions[0][2], "stock")
        numpy.testing.assert_array_equal(plan.instructions[0][3], [500, 267, 142, 76])
        numpy.testing.assert_array_equal(plan.instructions[1][3], [41, 41, 41, 41])
        numpy.testing.assert_array_equal(plan.instructions[2][3], [121, 121, 121, 121])
        return

    def test_to_worklist(self) -> None:
        # this test case tries to make it as hard as possible for the `to_worklist` method:
        # + vmax is different in every column
        # + stock has 2 rows, but dilution plan has 3
        # + diluent has 4 rows but dilution plan has 3
        # + diluent is not in the first column of a multi-column trough
        # + dilution plate is bigger than the plan
        # + destination plate is bigger than the plan
        stock_concentration = 20
        plan = robotools.DilutionPlan(
            xmin=0.01,
            xmax=10,
            R=3,
            C=4,
            stock=stock_concentration,
            mode="log",
            vmax=[1000, 1900, 980, 500],
            min_transfer=50,
        )
        stock = liquidhandling.Trough("Stock", 2, 1, min_volume=0, max_volume=10000, initial_volumes=10000)
        diluent = liquidhandling.Trough(
            "Diluent", 4, 2, min_volume=0, max_volume=20_000, initial_volumes=[0, 20_000]
        )
        dilution = liquidhandling.Labware("Dilution", 6, 8, min_volume=0, max_volume=2000)
        destination = liquidhandling.Labware("Destination", 7, 10, min_volume=0, max_volume=1000)
        with evotools.Worklist() as wl:
            plan.to_worklist(
                worklist=wl,
                stock=stock,
                stock_column=0,
                diluent=diluent,
                diluent_column=1,
                dilution_plate=dilution,
                destination_plate=destination,
                v_destination=200,
                mix_volume=0.75,
            )
        # assert the achieved concentrations in the destination
        numpy.testing.assert_array_almost_equal(
            plan.x, destination.composition["Stock"][: plan.R, : plan.C] * stock_concentration
        )
        assert "Mix column 0 with 75 % of its volume" in dilution.report
        assert "Mix column 1 with 50 % of its volume" in dilution.report
        return

    def test_to_worklist_hooks(self) -> None:
        stock_concentration = 123
        plan = robotools.DilutionPlan(
            xmin=1,
            xmax=123,
            R=3,
            C=4,
            stock=stock_concentration,
            mode="log",
            vmax=1000,
            min_transfer=50,
        )
        stock = liquidhandling.Trough("Stock", 2, 1, min_volume=0, max_volume=10000, initial_volumes=10000)
        diluent = liquidhandling.Trough(
            "Diluent", 4, 2, min_volume=0, max_volume=10000, initial_volumes=[0, 10000]
        )
        dilution = liquidhandling.Labware("Dilution", 3, 4, min_volume=0, max_volume=2000)

        # Multiple destinations; transferred to via a hook
        destinations = [
            robotools.Labware("DestinationOne", 3, 2, min_volume=0, max_volume=1000),
            robotools.Labware("DestinationTwo", 3, 2, min_volume=0, max_volume=1000),
        ]

        # Split the work across two worklists (also via the hook)
        wl_one = robotools.Worklist()
        wl_two = robotools.Worklist()

        def pre_mix(col, wl):
            wl.comment(f"Pre-mix on column {col}")
            if col == 1:
                return wl_two

        def post_mix(col, wl):
            wl.comment(f"Post-mix on column {col}")
            if col == 1:
                wl.transfer(
                    dilution,
                    dilution.wells[:, 0:2],
                    destinations[0],
                    destinations[0].wells[:, :],
                    volumes=100,
                )
            elif col == 3:
                wl.transfer(
                    dilution,
                    dilution.wells[:, 2:4],
                    destinations[1],
                    destinations[1].wells[:, :],
                    volumes=100,
                )

        plan.to_worklist(
            worklist=wl_one,
            stock=stock,
            stock_column=0,
            diluent=diluent,
            diluent_column=1,
            dilution_plate=dilution,
            pre_mix_hook=pre_mix,
            post_mix_hook=post_mix,
        )

        assert len(wl_two) > 0
        assert "C;Pre-mix on column 0" in wl_one
        assert "C;Pre-mix on column 1" in wl_one
        assert "C;Pre-mix on column 2" in wl_two
        assert "C;Pre-mix on column 3" in wl_two

        assert "C;Post-mix on column 0" in wl_one
        assert "C;Post-mix on column 1" in wl_two  # worklist switched in the mix hook!
        assert "C;Post-mix on column 2" in wl_two
        assert "C;Post-mix on column 3" in wl_two

        numpy.testing.assert_almost_equal(
            destinations[0].composition["Stock"] * stock_concentration, plan.x[:, [0, 1]]
        )
        numpy.testing.assert_almost_equal(
            destinations[1].composition["Stock"] * stock_concentration, plan.x[:, [2, 3]]
        )
        return


class TestUtils(unittest.TestCase):
    def test_get_trough_wells(self) -> None:
        with self.assertRaises(ValueError):
            robotools.get_trough_wells(n=-1, trough_wells=list("ABC"))
        with self.assertRaises(ValueError):
            robotools.get_trough_wells(n=3, trough_wells=[])
        with self.assertRaises(TypeError):
            robotools.get_trough_wells(n=0.5, trough_wells=list("ABC"))
        with self.assertRaises(TypeError):
            robotools.get_trough_wells(n=2, trough_wells="ABC")
        self.assertSequenceEqual(robotools.get_trough_wells(n=0, trough_wells=list("ABC")), list())
        self.assertSequenceEqual(robotools.get_trough_wells(n=1, trough_wells=list("ABC")), list("A"))
        self.assertSequenceEqual(robotools.get_trough_wells(n=3, trough_wells=list("ABC")), list("ABC"))
        self.assertSequenceEqual(robotools.get_trough_wells(n=4, trough_wells=list("ABC")), list("ABCA"))
        self.assertSequenceEqual(robotools.get_trough_wells(n=7, trough_wells=list("ABC")), list("ABCABCA"))
        return


class TestWellShifter(unittest.TestCase):
    def test_identity_transform(self) -> None:
        A = (6, 8)
        B = (8, 12)
        shifter = transform.WellShifter(A, B, shifted_A01="A01")

        original = ["A01", "C03", "D06", "F08"]
        expected = ["A01", "C03", "D06", "F08"]
        shifted = shifter.shift(original)
        numpy.testing.assert_array_equal(expected, shifter.shift(original))
        numpy.testing.assert_array_equal(shifter.unshift(shifted), original)
        return

    def test_center_shift(self) -> None:
        A = (6, 8)
        B = (8, 12)
        shifter = transform.WellShifter(A, B, shifted_A01="B03")

        original = ["A01", "C03", "D06", "F08"]
        expected = ["B03", "D05", "E08", "G10"]
        shifted = shifter.shift(original)
        numpy.testing.assert_array_equal(expected, shifter.shift(original))
        numpy.testing.assert_array_equal(shifter.unshift(shifted), original)
        return

    def test_boundcheck(self) -> None:
        A = (6, 8)
        B = (8, 12)

        with self.assertRaises(ValueError):
            transform.WellShifter(A, B, shifted_A01="E03")

        with self.assertRaises(ValueError):
            transform.WellShifter(A, B, shifted_A01="B06")
        return


class TestWellRotator(unittest.TestCase):
    def test_init(self) -> None:
        rotator = transform.WellRotator(original_shape=(7, 3))
        self.assertEqual(rotator.original_shape, (7, 3))
        self.assertEqual(rotator.rotated_shape, (3, 7))
        return

    def test_clockwise(self) -> None:
        A = (6, 8)
        rotator = transform.WellRotator(A)

        original = ["A01", "C03", "D06", "F08", "B04"]
        expected = ["A06", "C04", "F03", "H01", "D05"]
        rotated = rotator.rotate_cw(original)
        numpy.testing.assert_array_equal(expected, rotated)
        return

    def test_counterclockwise(self) -> None:
        A = (6, 8)
        rotator = transform.WellRotator(A)

        original = ["A01", "C03", "D06", "F08", "B04"]
        expected = ["H01", "F03", "C04", "A06", "E02"]
        rotated = rotator.rotate_ccw(original)
        numpy.testing.assert_array_equal(expected, rotated)
        return


if __name__ == "__main__":
    unittest.main()
