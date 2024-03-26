import numpy as np
import pytest

from robotools.evotools import EvoWorklist, Labwares
from robotools.evotools.types import Tip
from robotools.liquidhandling.labware import Labware, Trough
from robotools.worklists.exceptions import InvalidOperationError


class TestEvoWorklist:
    def test_aspirate_systemliquid(self) -> None:
        with EvoWorklist() as wl:
            wl.aspirate_well(Labwares.SystemLiquid.value, 1, 200)
            assert wl[-1] == "A;Systemliquid;;;1;;200.00;;;;"
        return

    def test_transfer_volumechecks(self) -> None:
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
        with EvoWorklist(max_volume=900, auto_split=False) as wl:
            with pytest.raises(InvalidOperationError):
                wl.transfer(source, ["A01", "B01"], destination, ["A01", "B01"], 1000)

        with EvoWorklist(max_volume=1200) as wl:
            wl.transfer(source, ["A01", "B01"], destination, ["A01", "B01"], 1000)
        return

    def test_transfer_2d_volumes(self) -> None:
        A = Labware("A", 2, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = Labware("B", 2, 4, min_volume=50, max_volume=250)
        with EvoWorklist() as wl:
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
        with EvoWorklist() as wl:
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
        with EvoWorklist() as worklist:
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
        with EvoWorklist() as worklist:
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
        with EvoWorklist() as worklist:
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
        with EvoWorklist() as worklist:
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

    def test_transfer_many_many(self) -> None:
        A = Trough("A", 3, 4, min_volume=50, max_volume=2500, initial_volumes=2000)
        B = Labware("B", 3, 4, min_volume=50, max_volume=250)
        with EvoWorklist() as worklist:
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
        with EvoWorklist() as worklist:
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
        with EvoWorklist() as worklist:
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

    def test_history_condensation(self) -> None:
        A = Labware("A", 3, 2, min_volume=300, max_volume=4600, initial_volumes=1500)
        B = Labware("B", 3, 2, min_volume=300, max_volume=4600, initial_volumes=1500)

        with EvoWorklist() as wl:
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

        with EvoWorklist() as wl:
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
        with EvoWorklist() as wl:
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
        with EvoWorklist() as wl:
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


class TestEvoCommands:
    def test_evo_aspirate(self) -> None:
        lw = Labware("A", 4, 5, min_volume=10, max_volume=100)
        lw.add("A01", 50)
        with EvoWorklist() as wl:
            wl.evo_aspirate(
                lw,
                "A01",
                labware_position=(30, 2),
                tips=[Tip.T2],
                volumes=20,
                liquid_class="PowerSuck",
            )
        assert len(wl) == 1
        assert "B;Aspirate" in wl[0]
        assert lw.volumes[0, 0] == 30
        pass

    def test_evo_dispense(self) -> None:
        lw = Labware("A", 4, 5, min_volume=10, max_volume=100)
        with EvoWorklist() as wl:
            wl.evo_dispense(
                lw,
                "A01",
                labware_position=(30, 2),
                tips=[Tip.T2],
                volumes=50,
                liquid_class="PowerPee",
            )
        assert len(wl) == 1
        assert "B;Dispense" in wl[0]
        assert lw.volumes[0, 0] == 50
        pass

    def test_evo_wash(self) -> None:
        with EvoWorklist() as wl:
            wl.evo_wash(
                tips=[Tip.T1, Tip.T5],
                waste_location=(40, 2),
                cleaner_location=(40, 3),
            )
        assert len(wl) == 1
        assert "B;Wash" in wl[0]
        pass
