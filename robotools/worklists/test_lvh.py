import numpy as np
import pytest

from robotools.evotools.worklist import EvoWorklist
from robotools.fluenttools.worklist import FluentWorklist
from robotools.liquidhandling.labware import Labware


class TestLargeVolumeHandling:
    @pytest.mark.parametrize("cls", [EvoWorklist, FluentWorklist])
    def test_single_split(self, cls) -> None:
        src = Labware("A", 3, 2, min_volume=1000, max_volume=25000, initial_volumes=12000)
        dst = Labware("B", 3, 2, min_volume=1000, max_volume=25000)
        with cls(auto_split=True) as wl:
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

    @pytest.mark.parametrize("cls", [EvoWorklist, FluentWorklist])
    def test_column_split(self, cls) -> None:
        src = Labware("A", 4, 2, min_volume=1000, max_volume=25000, initial_volumes=12000)
        dst = Labware("B", 4, 2, min_volume=1000, max_volume=25000)
        with cls(auto_split=True) as wl:
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

    @pytest.mark.parametrize("cls", [EvoWorklist, FluentWorklist])
    def test_block_split(self, cls) -> None:
        src = Labware("A", 3, 2, min_volume=1000, max_volume=25000, initial_volumes=12000)
        dst = Labware("B", 3, 2, min_volume=1000, max_volume=25000)
        with cls(auto_split=True) as wl:
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
