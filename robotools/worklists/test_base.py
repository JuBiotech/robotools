import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from robotools import EvoWorklist, FluentWorklist
from robotools.evotools.types import Tip
from robotools.liquidhandling.labware import Labware, Trough
from robotools.worklists import BaseWorklist
from robotools.worklists.exceptions import CompatibilityError, InvalidOperationError
from robotools.worklists.utils import (
    partition_by_column,
    partition_volume,
    prepare_aspirate_dispense_parameters,
)


class TestWorklist:
    def test_context(self) -> None:
        with BaseWorklist() as worklist:
            assert worklist is not None
        return

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(rack_label=None, position=1, volume=15)
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(rack_label=15, position=1, volume=15)
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(
                rack_label="thisisaveryverylongracklabelthatexceedsthemaximumlength", position=1, volume=15
            )
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(
                rack_label="rack label; with semicolon", position=1, volume=15
            )
        prepare_aspirate_dispense_parameters(rack_label="valid rack label", position=1, volume=15)
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=None, volume=15)
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position="3", volume=15)
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=-1, volume=15)
        prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=15)
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=None)
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume="nan")
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=float("nan"))
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=-15.4)
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume="bla")
        prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume="15")
        prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=20)
        prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=23.78)
        prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=np.array(23.4))

        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, liquid_class=None
            )
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, liquid_class="liquid;class"
            )
        prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, liquid_class="valid liquid class"
        )

        _, _, _, _, tip, _, _, _, _ = prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=4
        )
        assert tip == 8
        _, _, _, _, tip, _, _, _, _ = prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=Tip.T5
        )
        assert tip == 16
        _, _, _, _, tip, _, _, _, _ = prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=(Tip.T4, 4)
        )
        assert tip == 8
        _, _, _, _, tip, _, _, _, _ = prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=[Tip.T1, 4]
        )
        assert tip == 9
        _, _, _, _, tip, _, _, _, _ = prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=[1, 4]
        )
        assert tip == 9
        _, _, _, _, tip, _, _, _, _ = prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=[1, Tip.T4]
        )
        assert tip == 9
        _, _, _, _, tip, _, _, _, _ = prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, tip=Tip.Any
        )
        assert tip == ""

        with pytest.raises(ValueError, match="no Tip.Any elements are allowed"):
            prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, tip=(Tip.T1, Tip.Any)
            )
        with pytest.raises(ValueError, match="tip must be an int between 1 and 8, Tip or Iterable"):
            prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=15, tip=None)
        with pytest.raises(ValueError, match="it may only contain int or Tip values"):
            prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, tip=[1, 2.6]
            )
        with pytest.raises(ValueError, match="should be an int between 1 and 8 for _int_to_tip"):
            prepare_aspirate_dispense_parameters(rack_label="WaterTrough", position=1, volume=15, tip=12)

        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, rack_id=None
            )
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, rack_id="invalid;rack"
            )
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough",
                position=1,
                volume=15,
                rack_id="thisisaveryverylongrackthatexceedsthemaximumlength",
            )
        prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, rack_id="1235464"
        )
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, rack_type=None
            )
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, rack_type="invalid;rack type"
            )
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough",
                position=1,
                volume=15,
                rack_type="thisisaveryverylongracktypethatexceedsthemaximumlength",
            )
        prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, rack_type="valid rack type"
        )
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, forced_rack_type=None
            )
        with pytest.raises(ValueError):
            prepare_aspirate_dispense_parameters(
                rack_label="WaterTrough", position=1, volume=15, forced_rack_type="invalid;forced rack type"
            )
        prepare_aspirate_dispense_parameters(
            rack_label="WaterTrough", position=1, volume=15, forced_rack_type="valid forced rack type"
        )
        return

    def test_comment(self) -> None:
        with BaseWorklist() as wl:
            # empty and None comments should be ignored
            wl.comment("")
            wl.comment(None)
            # this will be the first actual comment
            wl.comment("This is a simple comment")
            with pytest.raises(ValueError):
                wl.comment("It must not contain ; semicolons")
            wl.comment(
                """
            But it may very well be
            a multiline comment
            """
            )
            exp = ["C;This is a simple comment", "C;But it may very well be", "C;a multiline comment"]
            assert wl == exp
        return

    def test_wash(self) -> None:
        with BaseWorklist() as wl:
            wl.wash()
            with pytest.raises(ValueError):
                wl.wash(scheme=15)
            with pytest.raises(ValueError):
                wl.wash(scheme="2")
            wl.wash(scheme=1)
            wl.wash(scheme=2)
            wl.wash(scheme=3)
            wl.wash(scheme=4)
            exp = [
                "W1;",
                "W1;",
                "W2;",
                "W3;",
                "W4;",
            ]
            assert wl == exp
        return

    def test_decontaminate(self) -> None:
        with BaseWorklist() as wl:
            wl.decontaminate()
            assert wl == ["WD;"]
        return

    def test_flush(self) -> None:
        with BaseWorklist() as wl:
            wl.flush()
            assert wl == ["F;"]
        return

    def test_commit(self) -> None:
        with BaseWorklist() as wl:
            wl.commit()
            assert wl == ["B;"]
        return

    def test_set_diti(self) -> None:
        with BaseWorklist() as wl:
            wl.set_diti(diti_index=1)
            with pytest.raises(InvalidOperationError):
                wl.set_diti(diti_index=2)
            wl.commit()
            wl.set_diti(diti_index=2)
            assert wl == [
                "S;1",
                "B;",
                "S;2",
            ]
        return

    def test_aspirate_single(self) -> None:
        with BaseWorklist() as wl:
            wl.aspirate_well("WaterTrough", 1, 200)
            assert wl[-1] == "A;WaterTrough;;;1;;200.00;;;;"
            wl.aspirate_well(
                "WaterTrough", 1, 200, rack_id="12345", rack_type="my_rack_id", tube_id="my_tube_id"
            )
            assert wl[-1] == "A;WaterTrough;12345;my_rack_id;1;my_tube_id;200.00;;;;"
            wl.aspirate_well(
                "WaterTrough", 1, 200, liquid_class="my_liquid_class", tip=8, forced_rack_type="forced_rack"
            )
            assert wl[-1] == "A;WaterTrough;;;1;;200.00;my_liquid_class;;128;forced_rack"
        return

    def test_dispense_single(self) -> None:
        with BaseWorklist() as wl:
            wl.dispense_well("WaterTrough", 1, 200)
            assert wl[-1] == "D;WaterTrough;;;1;;200.00;;;;"
            wl.dispense_well(
                "WaterTrough", 1, 200, rack_id="12345", rack_type="my_rack_id", tube_id="my_tube_id"
            )
            assert wl[-1] == "D;WaterTrough;12345;my_rack_id;1;my_tube_id;200.00;;;;"
            wl.dispense_well(
                "WaterTrough", 1, 200, liquid_class="my_liquid_class", tip=8, forced_rack_type="forced_rack"
            )
            assert wl[-1] == "D;WaterTrough;;;1;;200.00;my_liquid_class;;128;forced_rack"
        return

    def test_generic_transfer_raises_notimplemented(self) -> None:
        with pytest.raises(CompatibilityError, match="generic .*? type"):
            with BaseWorklist() as wl:
                wl.transfer(None, "A01", None, "A01", 100)
        return

    def test_accepts_path(self):
        fp = Path(tempfile.gettempdir(), os.urandom(24).hex() + ".gwl")
        try:
            with BaseWorklist(fp) as wl:
                wl.comment("Test")
            assert isinstance(wl._filepath, Path)
            assert fp.exists()
        finally:
            fp.unlink(missing_ok=True)
        return

    def test_save(self) -> None:
        tf = tempfile.mktemp() + ".gwl"
        error = None
        try:
            with BaseWorklist() as worklist:
                assert worklist.filepath is None
                worklist.flush()
                worklist.save(tf)
                assert os.path.exists(tf)
                # also check that the file can be overwritten if it exists already
                worklist.save(tf)
            assert os.path.exists(tf)
            with open(tf) as file:
                lines = file.readlines()
                assert lines == ["F;"]
        except Exception as ex:
            error = ex
        finally:
            os.remove(tf)
        assert os.path.exists(tf == False)
        if error:
            raise error
        return

    def test_autosave(self) -> None:
        tf = tempfile.mktemp() + ".gwl"
        error = None
        try:
            with BaseWorklist(tf) as worklist:
                assert isinstance(worklist.filepath, Path)
                worklist.flush()
            assert os.path.exists(tf)
            with open(tf) as file:
                lines = file.readlines()
                assert lines == ["F;"]
        except Exception as ex:
            error = ex
        finally:
            os.remove(tf)
        assert os.path.exists(tf == False)
        if error:
            raise error
        return

    def test_aspirate_dispense_distribute_require_specific_type(self):
        lw = Labware("A", 2, 3, min_volume=0, max_volume=1000, initial_volumes=500)
        tr = Trough("A", 1, 1, min_volume=0, max_volume=1000, initial_volumes=500)
        with BaseWorklist() as wl:
            with pytest.raises(TypeError, match="specific worklist type"):
                wl.aspirate(lw, "A01", 50)
            with pytest.raises(TypeError, match="specific worklist type"):
                wl.dispense(lw, "A01", 50)
            with pytest.raises(TypeError, match="specific worklist type"):
                wl.distribute(tr, 0, lw, ["A01"], volume=10)
        pass


@pytest.mark.parametrize("wl_cls", [EvoWorklist, FluentWorklist])
class TestStandardLabwareWorklist:
    def test_aspirate(self, wl_cls) -> None:
        source = Labware("SourceLW", rows=3, columns=3, min_volume=10, max_volume=200, initial_volumes=200)
        with wl_cls() as wl:
            wl.aspirate(source, ["A01", "A02", "C02"], 50, label=None)
            wl.aspirate(source, ["A03", "B03", "C03"], [10, 20, 30.5], label="second aspirate")
            assert wl == [
                "A;SourceLW;;;1;;50.00;;;;",
                "A;SourceLW;;;4;;50.00;;;;",
                "A;SourceLW;;;6;;50.00;;;;",
                "C;second aspirate",
                "A;SourceLW;;;7;;10.00;;;;",
                "A;SourceLW;;;8;;20.00;;;;",
                "A;SourceLW;;;9;;30.50;;;;",
            ]
            np.testing.assert_array_equal(
                source.volumes,
                [
                    [150, 150, 190],
                    [200, 200, 180],
                    [200, 150, 169.5],
                ],
            )
            assert len(source.history) == 3
        return

    def test_aspirate_2d_volumes(self, wl_cls) -> None:
        source = Labware("SourceLW", rows=2, columns=3, min_volume=10, max_volume=200, initial_volumes=200)
        with wl_cls() as wl:
            wl.aspirate(
                source,
                source.wells[:, :2],
                volumes=np.array(
                    [
                        [20, 30],
                        [15.3, 17.53],
                    ]
                ),
            )
            assert wl == [
                "A;SourceLW;;;1;;20.00;;;;",
                "A;SourceLW;;;2;;15.30;;;;",
                "A;SourceLW;;;3;;30.00;;;;",
                "A;SourceLW;;;4;;17.53;;;;",
            ]
            np.testing.assert_array_equal(source.volumes, [[180, 170, 200], [200 - 15.3, 200 - 17.53, 200]])
            assert len(source.history) == 2
        return

    def test_dispense(self, wl_cls) -> None:
        destination = Labware("DestinationLW", rows=2, columns=3, min_volume=10, max_volume=200)
        with wl_cls() as wl:
            wl.dispense(destination, ["A01", "A02", "A03"], 150, label=None)
            wl.dispense(destination, ["B01", "B02", "B03"], [10, 20, 30.5], label="second dispense")
            assert wl == [
                "D;DestinationLW;;;1;;150.00;;;;",
                "D;DestinationLW;;;3;;150.00;;;;",
                "D;DestinationLW;;;5;;150.00;;;;",
                "C;second dispense",
                "D;DestinationLW;;;2;;10.00;;;;",
                "D;DestinationLW;;;4;;20.00;;;;",
                "D;DestinationLW;;;6;;30.50;;;;",
            ]
            np.testing.assert_array_equal(
                destination.volumes,
                [
                    [150, 150, 150],
                    [10, 20, 30.5],
                ],
            )
            assert len(destination.history) == 3
        return

    def test_dispense_2d_volumes(self, wl_cls) -> None:
        destination = Labware("DestinationLW", rows=2, columns=3, min_volume=10, max_volume=200)
        with wl_cls() as wl:
            wl.dispense(
                destination,
                destination.wells[:, :2],
                volumes=np.array(
                    [
                        [20, 30],
                        [15.3, 17.53],
                    ]
                ),
            )
            assert wl == [
                "D;DestinationLW;;;1;;20.00;;;;",
                "D;DestinationLW;;;2;;15.30;;;;",
                "D;DestinationLW;;;3;;30.00;;;;",
                "D;DestinationLW;;;4;;17.53;;;;",
            ]
            np.testing.assert_array_equal(destination.volumes, [[20, 30, 0], [15.3, 17.53, 0]])
            assert len(destination.history) == 2
        return

    def test_skip_zero_volumes(self, wl_cls) -> None:
        source = Labware("SourceLW", rows=3, columns=3, min_volume=10, max_volume=200, initial_volumes=200)
        destination = Labware("DestinationLW", rows=2, columns=3, min_volume=10, max_volume=200)
        with wl_cls() as wl:
            wl.aspirate(source, ["A03", "B03", "C03"], [10, 0, 30.5])
            wl.dispense(destination, ["B01", "B02", "B03"], [10, 0, 30.5])
            assert wl == [
                "A;SourceLW;;;7;;10.00;;;;",
                "A;SourceLW;;;9;;30.50;;;;",
                "D;DestinationLW;;;2;;10.00;;;;",
                "D;DestinationLW;;;6;;30.50;;;;",
            ]
            np.testing.assert_array_equal(
                source.volumes,
                [
                    [200, 200, 190],
                    [200, 200, 200],
                    [200, 200, 169.5],
                ],
            )
            np.testing.assert_array_equal(
                destination.volumes,
                [
                    [0, 0, 0],
                    [10, 0, 30.5],
                ],
            )
            assert len(destination.history) == 2
        return

    def test_tip_selection(self, wl_cls) -> None:
        A = Labware("A", 3, 4, min_volume=10, max_volume=250, initial_volumes=100)
        with wl_cls() as wl:
            wl.aspirate(A, "A01", 10, tip=1)
            wl.aspirate(A, "A01", 10, tip=2)
            wl.aspirate(A, "A01", 10, tip=3)
            wl.aspirate(A, "A01", 10, tip=4)
            wl.aspirate(A, "A01", 10, tip=5)
            wl.aspirate(A, "A01", 10, tip=6)
            wl.aspirate(A, "A01", 10, tip=7)
            wl.aspirate(A, "A01", 10, tip=8)
            wl.dispense(A, "B01", 10, tip=Tip.T1)
            wl.dispense(A, "B02", 10, tip=Tip.T2)
            wl.dispense(A, "B03", 10, tip=Tip.T3)
            wl.dispense(A, "B04", 10, tip=Tip.T4)
            wl.dispense(A, "B04", 10, tip=Tip.T5)
            wl.dispense(A, "B04", 10, tip=Tip.T6)
            wl.dispense(A, "B04", 10, tip=Tip.T7)
            wl.dispense(A, "B04", 10, tip=Tip.T8)
            assert wl == [
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
            ]
        return

    def test_tip_mask(self, wl_cls) -> None:
        A = Labware("A", 3, 4, min_volume=10, max_volume=250)

        # Only allow three specific tips to be used...
        tips = [
            Tip.T1,  # 1 +
            Tip.T4,  # 8 +
            Tip.T7,  # 64
            # The sum of tips is = 73
        ]
        with wl_cls() as wl:
            wl.dispense(A, "A01", 10, tip=tips)
        assert wl[-1] == "D;A;;;1;;10.00;;;73;"
        pass


class TestLargeVolumeHandling:
    def testpartition_volume_helper(self) -> None:
        assert [] == partition_volume(0, max_volume=950)
        assert [550.3] == partition_volume(550.3, max_volume=950)
        assert [500 == 500], partition_volume(1000, max_volume=950)
        assert [500 == 499], partition_volume(999, max_volume=950)
        assert [667 == 667, 666], partition_volume(2000, max_volume=950)
        return

    def test_worklist_constructor(self) -> None:
        with pytest.raises(ValueError):
            with BaseWorklist(max_volume=None) as wl:
                pass
        with BaseWorklist(max_volume=800, auto_split=True) as wl:
            assert wl.max_volume == 800
            assert wl.auto_split == True
        with BaseWorklist(max_volume=800, auto_split=False) as wl:
            assert wl.max_volume == 800
            assert wl.auto_split == False
        return

    def test_max_volume_checking(self) -> None:
        source = Trough(
            "WaterTrough",
            virtual_rows=3,
            columns=3,
            min_volume=1000,
            max_volume=100 * 1000,
            initial_volumes=50 * 1000,
        )
        with EvoWorklist(max_volume=900, auto_split=False) as wl:
            with pytest.raises(InvalidOperationError):
                wl.aspirate_well("WaterTrough", 1, 1000)
            with pytest.raises(InvalidOperationError):
                wl.dispense_well("WaterTrough", 1, 1000)
            with pytest.raises(InvalidOperationError):
                wl.aspirate(source, ["A01", "A02", "C02"], 1000)
            with pytest.raises(InvalidOperationError):
                wl.dispense(source, ["A01", "A02", "C02"], 1000)

        source = Trough(
            "WaterTrough",
            virtual_rows=3,
            columns=3,
            min_volume=1000,
            max_volume=100 * 1000,
            initial_volumes=50 * 1000,
        )
        with EvoWorklist(max_volume=1200) as wl:
            wl.aspirate_well("WaterTrough", 1, 1000)
            wl.dispense_well("WaterTrough", 1, 1000)
            wl.aspirate(source, ["A01", "A02", "C02"], 1000)
            wl.dispense(source, ["A01", "A02", "C02"], 1000)
        return

    def testpartition_by_columns_source(self) -> None:
        column_groups = partition_by_column(
            ["A01", "B01", "A03", "B03", "C02"],
            ["A01", "B01", "C01", "D01", "E01"],
            [2500, 3500, 1000, 500, 2000],
            partition_by="source",
        )
        assert len(column_groups) == 3
        assert column_groups[0] == (
            ["A01", "B01"],
            ["A01", "B01"],
            [2500, 3500],
        )
        assert column_groups[1] == (
            ["C02"],
            ["E01"],
            [2000],
        )
        assert column_groups[2] == (
            ["A03", "B03"],
            ["C01", "D01"],
            [1000, 500],
        )
        return

    def testpartition_by_columns_destination(self) -> None:
        column_groups = partition_by_column(
            ["A01", "B01", "A03", "B03", "C02"],
            ["A01", "B01", "C02", "D01", "E02"],
            [2500, 3500, 1000, 500, 2000],
            partition_by="destination",
        )
        assert len(column_groups) == 2
        assert column_groups[0] == (
            ["A01", "B01", "B03"],
            ["A01", "B01", "D01"],
            [2500, 3500, 500],
        )
        assert column_groups[1] == (
            ["A03", "C02"],
            ["C02", "E02"],
            [1000, 2000],
        )
        return

    def testpartition_by_columns_sorting(self) -> None:
        # within every column, the wells are supposed to be sorted by row
        # The test source wells are partially sorted (col 1 is in the right order, col 3 in the reverse)
        # The result is expected to always be sorted by row, either in the source (first case) or destination:

        # by source
        column_groups = partition_by_column(
            ["A01", "B01", "B03", "A03", "C02"],
            ["B01", "A01", "C01", "D01", "E01"],
            [2500, 3500, 1000, 500, 2000],
            partition_by="source",
        )
        assert len(column_groups) == 3
        assert column_groups[0] == (
            ["A01", "B01"],
            ["B01", "A01"],
            [2500, 3500],
        )
        assert column_groups[1] == (
            ["C02"],
            ["E01"],
            [2000],
        )
        assert column_groups[2] == (
            ["A03", "B03"],
            ["D01", "C01"],
            [500, 1000],
        )

        # by destination
        # (destination wells are across 3 columns; reverse order in col 1, forward order in col 3)
        column_groups = partition_by_column(
            ["A01", "B01", "B03", "A03", "C02"],
            ["B01", "A01", "C03", "D03", "E02"],
            [2500, 3500, 1000, 500, 2000],
            partition_by="destination",
        )
        assert len(column_groups) == 3
        assert column_groups[0] == (
            ["B01", "A01"],
            ["A01", "B01"],
            [3500, 2500],
        )
        assert column_groups[1] == (
            ["C02"],
            ["E02"],
            [2000],
        )
        assert column_groups[2] == (
            ["B03", "A03"],
            ["C03", "D03"],
            [1000, 500],
        )
        return


class TestReagentDistribution:
    def test_parameter_validation(self, caplog) -> None:
        with BaseWorklist() as wl:
            with pytest.raises(ValueError):
                wl.reagent_distribution("S1", 1, 8, "D1", 1, 20, volume=50, direction="invalid")
            with pytest.raises(ValueError):
                # one excluded well not in the destination range
                wl.reagent_distribution("S1", 1, 8, "D1", 1, 20, volume=50, exclude_wells=[18, 19, 23])
        with pytest.raises(InvalidOperationError):
            with caplog.at_level(logging.WARNING, logger="robotools.evotools"):
                with BaseWorklist(max_volume=950) as wl:
                    # dispense more than diluter volume
                    wl.reagent_distribution("S1", 1, 8, "D1", 1, 20, volume=1200)
            assert "account for a large dispense" in caplog.records[0].message
        return

    def test_default_parameterization(self) -> None:
        with BaseWorklist() as wl:
            wl.reagent_distribution("S1", 1, 20, "D1", 2, 21, volume=50)
        assert wl[0] == "R;S1;;;1;20;D1;;;2;21;50;;1;1;0"
        return

    def test_full_parameterization(self) -> None:
        with BaseWorklist() as wl:
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
        assert wl[0] == "R;S1;S1234;MP3Pos;1;20;D1;D1234;MP4Pos;2;21;50;TestLC;2;3;1;2;4;8"
        return

    def test_large_volume_multi_disp_adaption(self) -> None:
        with BaseWorklist() as wl:
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
        assert wl[0] == "R;S1;;;1;8;D1;;;1;96;400;;1;2;0"
        return

    def test_oo_parameter_validation(self) -> None:
        with EvoWorklist() as wl:
            src = Labware("NotATrough", 6, 2, min_volume=20, max_volume=1000)
            dst = Labware("48deep", 6, 8, min_volume=50, max_volume=4000)
            with pytest.raises(ValueError):
                wl.distribute(src, 0, dst, dst.wells[:, :3], volume=50)
        with FluentWorklist(max_volume=950) as wl:
            src = Trough("Water", 8, 2, min_volume=20, max_volume=1000)
            dst = Labware("48deep", 6, 8, min_volume=50, max_volume=4000)
            with pytest.raises(InvalidOperationError):
                wl.distribute(src, 0, dst, dst.wells[:, :3], volume=1200)
        return

    def test_oo_example_1(self) -> None:
        src = Trough(
            "T3",
            8,
            1,
            min_volume=20,
            max_volume=100 * 1000,
            initial_volumes=100 * 1000,
        )
        dst = Labware("MTP-96-3", 8, 12, min_volume=20, max_volume=300)
        with FluentWorklist() as wl:
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
        assert wl[0] == "C;Test Label"
        assert (
            wl[1] == "R;T3;;Trough 100ml;1;8;MTP-96-3;;96 Well Microplate;1;96;100;Water;1;6;0;27;46;51;69;82"
        )
        assert src.volumes[0 == 0], 100 * 1000 - 91 * 100
        dst_exp = np.ones_like(dst.volumes) * 100
        dst_exp[dst.indices["C04"]] = 0
        dst_exp[dst.indices["C07"]] = 0
        dst_exp[dst.indices["E09"]] = 0
        dst_exp[dst.indices["F06"]] = 0
        dst_exp[dst.indices["B11"]] = 0
        np.testing.assert_array_equal(dst.volumes, dst_exp)
        return

    def test_oo_example_2(self) -> None:
        src = Trough(
            "T2",
            8,
            1,
            min_volume=20,
            max_volume=100 * 1000,
            initial_volumes=100 * 1000,
        )
        dst = Labware("MTP-96-2", 8, 12, min_volume=20, max_volume=300)
        with EvoWorklist() as wl:
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
        assert wl[0] == "C;Test Label"
        assert wl[1] == "R;T2;;Trough 100ml;1;8;MTP-96-2;;96 Well Microplate;1;96;100;Water;2;5;0"
        assert src.volumes[0 == 0], 100 * 1000 - 96 * 100
        np.testing.assert_array_equal(dst.volumes, np.ones_like(dst.volumes) * 100)
        return

    def test_oo_block_from_right(self) -> None:
        src = Trough(
            "Water",
            8,
            1,
            min_volume=20,
            max_volume=100 * 1000,
            initial_volumes=100 * 1000,
        )
        dst = Labware("96mtp", 8, 12, min_volume=20, max_volume=300)
        with EvoWorklist() as wl:
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
        assert wl[0] == "C;Test Label"
        skip_pos = ";21;22;23;24;25;29;30;31;32;33;37;38;39;40;41;45;46;47;48;49"
        assert wl[1] == f"R;Water;;;1;8;96mtp;;;18;52;50;TestLC;10;5;1{skip_pos}"
        assert src.volumes[0 == 0], 100 * 1000 - 15 * 50
        assert np.all(dst.volumes[1:4, 2:7] == 50)
        return
