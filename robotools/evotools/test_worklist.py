import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from robotools.evotools.exceptions import InvalidOperationError
from robotools.evotools.types import Labwares, Tip
from robotools.evotools.worklist import Worklist
from robotools.liquidhandling.labware import Labware, Trough
from robotools.worklists.utils import (
    optimize_partition_by,
    partition_by_column,
    partition_volume,
    prepare_aspirate_dispense_parameters,
)


class TestWorklist:
    def test_context(self) -> None:
        with Worklist() as worklist:
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
        with Worklist() as wl:
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
        with Worklist() as wl:
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
        with Worklist() as wl:
            wl.decontaminate()
            assert wl == ["WD;"]
        return

    def test_flush(self) -> None:
        with Worklist() as wl:
            wl.flush()
            assert wl == ["F;"]
        return

    def test_commit(self) -> None:
        with Worklist() as wl:
            wl.commit()
            assert wl == ["B;"]
        return

    def test_set_diti(self) -> None:
        with Worklist() as wl:
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
        with Worklist() as wl:
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
        with Worklist() as wl:
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

    def test_aspirate_systemliquid(self) -> None:
        with Worklist() as wl:
            wl.aspirate_well(Labwares.SystemLiquid.value, 1, 200)
            assert wl[-1] == "A;Systemliquid;;;1;;200.00;;;;"
        return

    def test_accepts_path(self):
        fp = Path(tempfile.gettempdir(), os.urandom(24).hex() + ".gwl")
        try:
            with Worklist(fp) as wl:
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
            with Worklist() as worklist:
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
            with Worklist(tf) as worklist:
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


class TestStandardLabwareWorklist:
    def test_aspirate(self) -> None:
        source = Labware("SourceLW", rows=3, columns=3, min_volume=10, max_volume=200, initial_volumes=200)
        with Worklist() as wl:
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

    def test_aspirate_2d_volumes(self) -> None:
        source = Labware("SourceLW", rows=2, columns=3, min_volume=10, max_volume=200, initial_volumes=200)
        with Worklist() as wl:
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

    def test_dispense(self) -> None:
        destination = Labware("DestinationLW", rows=2, columns=3, min_volume=10, max_volume=200)
        with Worklist() as wl:
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

    def test_dispense_2d_volumes(self) -> None:
        destination = Labware("DestinationLW", rows=2, columns=3, min_volume=10, max_volume=200)
        with Worklist() as wl:
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

    def test_skip_zero_volumes(self) -> None:
        source = Labware("SourceLW", rows=3, columns=3, min_volume=10, max_volume=200, initial_volumes=200)
        destination = Labware("DestinationLW", rows=2, columns=3, min_volume=10, max_volume=200)
        with Worklist() as wl:
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

    def test_transfer_2d_volumes(self) -> None:
        A = Labware("A", 2, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = Labware("B", 2, 4, min_volume=50, max_volume=250)
        with Worklist() as wl:
            wl.transfer(
                A,
                A.wells[:, :2],
                B,
                B.wells[:, :2],
                volumes=np.array(
                    [
                        [20, 30],
                        [15.3, 17.53],
                    ]
                ),
            )
            assert wl == [
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
            ]
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [180, 170, 200, 200],
                    [200 - 15.3, 200 - 17.53, 200, 200],
                ],
            )
            assert np.array_equal(
                B.volumes,
                np.array(
                    [
                        [20, 30, 0, 0],
                        [15.3, 17.53, 0, 0],
                    ]
                ),
            )
            assert len(A.history) == 2
            assert len(B.history) == 2
        return

    def test_transfer_2d_volumes_no_wash(self) -> None:
        A = Labware("A", 2, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = Labware("B", 2, 4, min_volume=50, max_volume=250)
        with Worklist() as wl:
            wl.transfer(
                A,
                A.wells[:, :2],
                B,
                B.wells[:, :2],
                volumes=np.array(
                    [
                        [20, 30],
                        [15.3, 17.53],
                    ]
                ),
                wash_scheme=None,
            )
            assert wl == [
                "A;A;;;1;;20.00;;;;",
                "D;B;;;1;;20.00;;;;",
                "A;A;;;2;;15.30;;;;",
                "D;B;;;2;;15.30;;;;",
                "A;A;;;3;;30.00;;;;",
                "D;B;;;3;;30.00;;;;",
                "A;A;;;4;;17.53;;;;",
                "D;B;;;4;;17.53;;;;",
            ]
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [180, 170, 200, 200],
                    [200 - 15.3, 200 - 17.53, 200, 200],
                ],
            )
            np.testing.assert_array_equal(
                B.volumes,
                [
                    [20, 30, 0, 0],
                    [15.3, 17.53, 0, 0],
                ],
            )
            assert len(A.history) == 2
            assert len(B.history) == 2
        return

    def test_transfer_many_many(self) -> None:
        A = Labware("A", 3, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = Labware("B", 3, 4, min_volume=50, max_volume=250)
        wells = ["A01", "B01"]
        with Worklist() as worklist:
            worklist.transfer(A, wells, B, wells, 50, label="first transfer")
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [150, 200, 200, 200],
                    [150, 200, 200, 200],
                    [200, 200, 200, 200],
                ],
            )
            np.testing.assert_array_equal(
                B.volumes,
                [
                    [50, 0, 0, 0],
                    [50, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            )
            worklist.transfer(A, ["A03", "B04"], B, ["A04", "B04"], 50, label="second transfer")
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [150, 200, 150, 200],
                    [150, 200, 200, 150],
                    [200, 200, 200, 200],
                ],
            )
            np.testing.assert_array_equal(
                B.volumes,
                [
                    [50, 0, 0, 50],
                    [50, 0, 0, 50],
                    [0, 0, 0, 0],
                ],
            )
            assert worklist == [
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
            ]
            assert len(A.history) == 3
            assert len(B.history) == 3
        return

    def test_transfer_many_many_2d(self) -> None:
        A = Labware("A", 3, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = Labware("B", 3, 4, min_volume=50, max_volume=250)
        wells = A.wells[:, :2]
        with Worklist() as worklist:
            worklist.transfer(A, wells, B, wells, 50)
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [150, 150, 200, 200],
                    [150, 150, 200, 200],
                    [150, 150, 200, 200],
                ],
            )
            np.testing.assert_array_equal(
                B.volumes,
                [
                    [50, 50, 0, 0],
                    [50, 50, 0, 0],
                    [50, 50, 0, 0],
                ],
            )
            assert worklist == [
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
            ]
            assert len(A.history) == 2
            assert len(B.history) == 2
        return

    def test_transfer_one_many(self) -> None:
        A = Labware("A", 3, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = Labware("B", 3, 4, min_volume=50, max_volume=250)
        with Worklist() as worklist:
            worklist.transfer(A, "A01", B, ["B01", "B02", "B03"], 25)
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [125, 200, 200, 200],
                    [200, 200, 200, 200],
                    [200, 200, 200, 200],
                ],
            )
            np.testing.assert_array_equal(
                B.volumes,
                [
                    [0, 0, 0, 0],
                    [25, 25, 25, 0],
                    [0, 0, 0, 0],
                ],
            )
            worklist.transfer(A, ["A01"], B, ["B01", "B02", "B03"], 25)
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [50, 200, 200, 200],
                    [200, 200, 200, 200],
                    [200, 200, 200, 200],
                ],
            )
            np.testing.assert_array_equal(
                B.volumes,
                [
                    [0, 0, 0, 0],
                    [50, 50, 50, 0],
                    [0, 0, 0, 0],
                ],
            )
            assert worklist == [
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
            ]
            assert len(A.history) == 3
            assert len(B.history) == 3
        return

    def test_transfer_many_one(self) -> None:
        A = Labware("A", 3, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = Labware("B", 3, 4, min_volume=50, max_volume=250)
        with Worklist() as worklist:
            worklist.transfer(A, ["A01", "A02", "A03"], B, "B01", 25)
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [175, 175, 175, 200],
                    [200, 200, 200, 200],
                    [200, 200, 200, 200],
                ],
            )
            np.testing.assert_array_equal(
                B.volumes,
                [
                    [0, 0, 0, 0],
                    [75, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            )
            assert worklist == [
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
            ]
            assert len(A.history) == 2
            assert len(B.history) == 2
        return

    def test_tip_selection(self) -> None:
        A = Labware("A", 3, 4, min_volume=10, max_volume=250, initial_volumes=100)
        with Worklist() as wl:
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

    def test_tip_mask(self) -> None:
        A = Labware("A", 3, 4, min_volume=10, max_volume=250)

        # Only allow three specific tips to be used...
        tips = [
            Tip.T1,  # 1 +
            Tip.T4,  # 8 +
            Tip.T7,  # 64
            # The sum of tips is = 73
        ]
        with Worklist() as wl:
            wl.dispense(A, "A01", 10, tip=tips)
        assert wl[-1] == "D;A;;;1;;10.00;;;73;"
        pass

    def test_history_condensation(self) -> None:
        A = Labware("A", 3, 2, min_volume=300, max_volume=4600, initial_volumes=1500)
        B = Labware("B", 3, 2, min_volume=300, max_volume=4600, initial_volumes=1500)

        with Worklist() as wl:
            wl.transfer(A, ["A01", "B01", "C02"], B, ["A01", "B02", "C01"], [900, 100, 900], label="transfer")

        assert len(A.history) == 2
        assert A.history[-1][0] == "transfer"
        np.testing.assert_array_equal(
            A.history[-1][1],
            [
                [1500 - 900, 1500],
                [1500 - 100, 1500],
                [1500, 1500 - 900],
            ],
        )

        assert len(B.history) == 2
        assert B.history[-1][0] == "transfer"
        np.testing.assert_array_equal(
            B.history[-1][1],
            [
                [1500 + 900, 1500],
                [1500, 1500 + 100],
                [1500 + 900, 1500],
            ],
        )
        return

    def test_history_condensation_within_labware(self) -> None:
        A = Labware("A", 3, 2, min_volume=300, max_volume=4600, initial_volumes=1500)

        with Worklist() as wl:
            wl.transfer(A, ["A01", "B01", "C02"], A, ["A01", "B02", "C01"], [900, 100, 900], label="mix")

        assert len(A.history) == 2
        assert A.history[-1][0] == "mix"
        np.testing.assert_array_equal(
            A.history[-1][1],
            [
                [1500 - 900 + 900, 1500],
                [1500 - 100, 1500 + 100],
                [1500 + 900, 1500 - 900],
            ],
        )
        return


class TestTroughLabwareWorklist:
    def test_aspirate(self) -> None:
        source = Trough(
            "SourceLW", virtual_rows=3, columns=3, min_volume=10, max_volume=200, initial_volumes=200
        )
        with Worklist() as wl:
            wl.aspirate(source, ["A01", "A02", "C02"], 50)
            wl.aspirate(source, ["A01", "A02", "C02"], [1, 2, 3])
            assert wl == [
                "A;SourceLW;;;1;;50.00;;;;",
                "A;SourceLW;;;4;;50.00;;;;",
                "A;SourceLW;;;6;;50.00;;;;",
                "A;SourceLW;;;1;;1.00;;;;",
                "A;SourceLW;;;4;;2.00;;;;",
                "A;SourceLW;;;6;;3.00;;;;",
            ]
            np.testing.assert_array_equal(source.volumes, [[149, 95, 200]])
            assert len(source.history) == 3
        return

    def test_dispense(self) -> None:
        destination = Trough("DestinationLW", virtual_rows=3, columns=3, min_volume=10, max_volume=200)
        with Worklist() as wl:
            wl.dispense(destination, ["A01", "A02", "A03", "B01"], 50)
            wl.dispense(destination, ["A01", "A02", "C02"], [1, 2, 3])
            assert wl == [
                "D;DestinationLW;;;1;;50.00;;;;",
                "D;DestinationLW;;;4;;50.00;;;;",
                "D;DestinationLW;;;7;;50.00;;;;",
                "D;DestinationLW;;;2;;50.00;;;;",
                "D;DestinationLW;;;1;;1.00;;;;",
                "D;DestinationLW;;;4;;2.00;;;;",
                "D;DestinationLW;;;6;;3.00;;;;",
            ]
            np.testing.assert_array_equal(destination.volumes, [[101, 55, 50]])
            assert len(destination.history) == 3
        return

    def test_transfer_many_many(self) -> None:
        A = Trough("A", 3, 4, min_volume=50, max_volume=2500, initial_volumes=2000)
        B = Labware("B", 3, 4, min_volume=50, max_volume=250)
        with Worklist() as worklist:
            worklist.transfer(A, ["A01", "B01"], B, ["A01", "B01"], 50)
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [1900, 2000, 2000, 2000],
                ],
            )
            np.testing.assert_array_equal(
                B.volumes,
                [
                    [50, 0, 0, 0],
                    [50, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            )
            worklist.transfer(A, ["A03", "B04"], B, ["A04", "B04"], [50, 75])
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [1900, 2000, 1950, 1925],
                ],
            )
            np.testing.assert_array_equal(
                B.volumes,
                [
                    [50, 0, 0, 50],
                    [50, 0, 0, 75],
                    [0, 0, 0, 0],
                ],
            )
            assert worklist == [
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
            ]
            assert len(A.history) == 3
            assert len(B.history) == 3
        return

    def test_transfer_one_many(self) -> None:
        A = Trough("A", 3, 4, min_volume=50, max_volume=2500, initial_volumes=2000)
        B = Labware("B", 3, 4, min_volume=50, max_volume=250)
        with Worklist() as worklist:
            worklist.transfer(A, "A01", B, ["B01", "B02", "B03"], 25)
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [1925, 2000, 2000, 2000],
                ],
            )
            np.testing.assert_array_equal(
                B.volumes,
                [
                    [0, 0, 0, 0],
                    [25, 25, 25, 0],
                    [0, 0, 0, 0],
                ],
            )
            assert worklist == [
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
            ]

            worklist.transfer(A, ["A01"], B, ["B01", "B02", "B03"], [25, 30, 35])
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [1835, 2000, 2000, 2000],
                ],
            )
            np.testing.assert_array_equal(
                B.volumes,
                [
                    [0, 0, 0, 0],
                    [50, 55, 60, 0],
                    [0, 0, 0, 0],
                ],
            )
            assert worklist == [
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
            ]
            assert len(A.history) == 3
            assert len(B.history) == 3
        return

    def test_transfer_many_one(self) -> None:
        A = Trough("A", 3, 4, min_volume=50, max_volume=2500, initial_volumes=[2000, 1500, 1000, 500])
        B = Labware("B", 3, 4, min_volume=10, max_volume=250, initial_volumes=100)
        with Worklist() as worklist:
            worklist.transfer(A, ["A01", "A02", "A03"], B, "B01", 25)
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [1975, 1475, 975, 500],
                ],
            )
            np.testing.assert_array_equal(
                B.volumes,
                [
                    [100, 100, 100, 100],
                    [175, 100, 100, 100],
                    [100, 100, 100, 100],
                ],
            )
            assert worklist == [
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
            ]

            worklist.transfer(B, B.wells[:, 2], A, A.wells[:, 3], [50, 60, 70])
            np.testing.assert_array_equal(
                A.volumes,
                [
                    [1975, 1475, 975, 680],
                ],
            )
            np.testing.assert_array_equal(
                B.volumes,
                [
                    [100, 100, 50, 100],
                    [175, 100, 40, 100],
                    [100, 100, 30, 100],
                ],
            )
            assert worklist == [
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
            ]
            assert len(A.history) == 3
            assert len(B.history) == 3
        return


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
            with Worklist(max_volume=None) as wl:
                pass
        with Worklist(max_volume=800, auto_split=True) as wl:
            assert wl.max_volume == 800
            assert wl.auto_split == True
        with Worklist(max_volume=800, auto_split=False) as wl:
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
        destination = Trough(
            "WaterTrough",
            virtual_rows=3,
            columns=3,
            min_volume=1000,
            max_volume=100 * 1000,
            initial_volumes=50 * 1000,
        )
        with Worklist(max_volume=900, auto_split=False) as wl:
            with pytest.raises(InvalidOperationError):
                wl.aspirate_well("WaterTrough", 1, 1000)
            with pytest.raises(InvalidOperationError):
                wl.dispense_well("WaterTrough", 1, 1000)
            with pytest.raises(InvalidOperationError):
                wl.aspirate(source, ["A01", "A02", "C02"], 1000)
            with pytest.raises(InvalidOperationError):
                wl.dispense(source, ["A01", "A02", "C02"], 1000)
            with pytest.raises(InvalidOperationError):
                wl.transfer(source, ["A01", "B01"], destination, ["A01", "B01"], 1000)

        source = Trough(
            "WaterTrough",
            virtual_rows=3,
            columns=3,
            min_volume=1000,
            max_volume=100 * 1000,
            initial_volumes=50 * 1000,
        )
        destination = Trough(
            "WaterTrough",
            virtual_rows=3,
            columns=3,
            min_volume=1000,
            max_volume=100 * 1000,
            initial_volumes=50 * 1000,
        )
        with Worklist(max_volume=1200) as wl:
            wl.aspirate_well("WaterTrough", 1, 1000)
            wl.dispense_well("WaterTrough", 1, 1000)
            wl.aspirate(source, ["A01", "A02", "C02"], 1000)
            wl.dispense(source, ["A01", "A02", "C02"], 1000)
            wl.transfer(source, ["A01", "B01"], destination, ["A01", "B01"], 1000)
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

    def test_single_split(self) -> None:
        src = Labware("A", 3, 2, min_volume=1000, max_volume=25000, initial_volumes=12000)
        dst = Labware("B", 3, 2, min_volume=1000, max_volume=25000)
        with Worklist(auto_split=True) as wl:
            wl.transfer(src, "A01", dst, "A01", 2000, label="Transfer more than 2x the max")
            assert wl == [
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
            ]
        # Two extra steps were necessary because of LVH
        assert "Transfer more than 2x the max (2 LVH steps)" in src.report
        assert "Transfer more than 2x the max (2 LVH steps)" in dst.report
        np.testing.assert_array_equal(
            src.volumes,
            [
                [12000 - 2000, 12000],
                [12000, 12000],
                [12000, 12000],
            ],
        )
        np.testing.assert_array_equal(
            dst.volumes,
            [
                [2000, 0],
                [0, 0],
                [0, 0],
            ],
        )
        return

    def test_column_split(self) -> None:
        src = Labware("A", 4, 2, min_volume=1000, max_volume=25000, initial_volumes=12000)
        dst = Labware("B", 4, 2, min_volume=1000, max_volume=25000)
        with Worklist(auto_split=True) as wl:
            wl.transfer(
                src, ["A01", "B01", "D01", "C01"], dst, ["A01", "B01", "D01", "C01"], [1500, 250, 0, 1200]
            )
            assert wl == [
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
            ]
        np.testing.assert_array_equal(
            src.volumes,
            [
                [12000 - 1500, 12000],
                [12000 - 250, 12000],
                [12000 - 1200, 12000],
                [12000, 12000],
            ],
        )
        np.testing.assert_array_equal(
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
        src = Labware("A", 3, 2, min_volume=1000, max_volume=25000, initial_volumes=12000)
        dst = Labware("B", 3, 2, min_volume=1000, max_volume=25000)
        with Worklist(auto_split=True) as wl:
            wl.transfer(
                # A01, B01, A02, B02
                src,
                src.wells[:2, :],
                dst,
                ["A01", "B01", "C01", "A02"],
                [1500, 250, 1200, 3000],
            )
            assert wl == [
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
            ]

        # How the number of splits is calculated:
        # 1500 is split 2x → 1 extra
        # 250 is not split
        # 1200 is split 2x → 1 extra
        # 3000 is split 4x → 3 extra
        # Sum of extra steps: 5
        assert "5 LVH steps" in src.report
        assert "5 LVH steps" in dst.report
        np.testing.assert_array_equal(
            src.volumes,
            [
                [12000 - 1500, 12000 - 1200],
                [12000 - 250, 12000 - 3000],
                [12000, 12000],
            ],
        )
        np.testing.assert_array_equal(
            dst.volumes,
            [
                [1500, 3000],
                [250, 0],
                [1200, 0],
            ],
        )
        return


class TestReagentDistribution:
    def test_parameter_validation(self, caplog) -> None:
        with Worklist() as wl:
            with pytest.raises(ValueError):
                wl.reagent_distribution("S1", 1, 8, "D1", 1, 20, volume=50, direction="invalid")
            with pytest.raises(ValueError):
                # one excluded well not in the destination range
                wl.reagent_distribution("S1", 1, 8, "D1", 1, 20, volume=50, exclude_wells=[18, 19, 23])
        with pytest.raises(InvalidOperationError):
            with caplog.at_level(logging.WARNING, logger="robotools.evotools"):
                with Worklist(max_volume=950) as wl:
                    # dispense more than diluter volume
                    wl.reagent_distribution("S1", 1, 8, "D1", 1, 20, volume=1200)
            assert "account for a large dispense" in caplog.records[0].message
        return

    def test_default_parameterization(self) -> None:
        with Worklist() as wl:
            wl.reagent_distribution("S1", 1, 20, "D1", 2, 21, volume=50)
        assert wl[0] == "R;S1;;;1;20;D1;;;2;21;50;;1;1;0"
        return

    def test_full_parameterization(self) -> None:
        with Worklist() as wl:
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
        with Worklist() as wl:
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
        with Worklist() as wl:
            src = Labware("NotATrough", 6, 2, min_volume=20, max_volume=1000)
            dst = Labware("48deep", 6, 8, min_volume=50, max_volume=4000)
            with pytest.raises(ValueError):
                wl.distribute(src, 0, dst, dst.wells[:, :3], volume=50)
        with Worklist(max_volume=950) as wl:
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
        with Worklist() as wl:
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
        with Worklist() as wl:
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
        with Worklist() as wl:
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


class TestFunctions:
    def test_automatic_partitioning(self, caplog) -> None:
        S = Labware("S", 8, 2, min_volume=5000, max_volume=250 * 1000)
        D = Labware("D", 8, 2, min_volume=5000, max_volume=250 * 1000)
        ST = Trough("ST", 8, 2, min_volume=5000, max_volume=250 * 1000)
        DT = Trough("DT", 8, 2, min_volume=5000, max_volume=250 * 1000)

        # Expected behaviors:
        # + always keep settings other than 'auto'
        # + warn user about inefficient configuration (when user selects to partition by the trough)

        # automatic
        assert "source" == optimize_partition_by(S, D, "auto", "No troughs at all")
        assert "source" == optimize_partition_by(S, DT, "auto", "Trough destination")
        assert "destination" == optimize_partition_by(ST, D, "auto", "Trough source")
        optimize_partition_by(ST, DT, "auto", "Trough source and destination") == "source"

        # fixed to source
        assert "source" == optimize_partition_by(S, D, "source", "No troughs at all")
        assert "source" == optimize_partition_by(S, DT, "source", "Trough destination")
        with caplog.at_level(logging.WARNING, logger="robotools.evotools"):
            assert "source" == optimize_partition_by(ST, D, "source", "Trough source")
        assert 'Consider using partition_by="destination"' in caplog.records[0].message
        optimize_partition_by(ST, DT, "auto", "Trough source and destination") == "source"

        # fixed to destination
        optimize_partition_by(S, D, "destination", "No troughs at all") == "destination"
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="robotools.evotools"):
            assert optimize_partition_by(S, DT, "destination", "Trough destination") == "destination"
        assert 'Consider using partition_by="source"' in caplog.records[0].message
        assert optimize_partition_by(ST, D, "destination", "Trough source") == "destination"
        assert optimize_partition_by(ST, DT, "destination", "Trough source and destination") == "destination"
        return
