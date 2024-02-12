import numpy as np
import pytest

from robotools.evotools import EvoWorklist
from robotools.liquidhandling import Labware, Trough
from robotools.utils import DilutionPlan, get_trough_wells


class TestDilutionPlan:
    def test_argchecking(self) -> None:
        with pytest.raises(ValueError):
            DilutionPlan(xmin=0.001, xmax=30, R=8, C=12, stock=20, mode="log", vmax=1000, min_transfer=20)

        with pytest.raises(ValueError):
            DilutionPlan(xmin=0.001, xmax=30, R=8, C=12, stock=30, mode="invalid", vmax=1000, min_transfer=20)

        with pytest.raises(ValueError):
            DilutionPlan(xmin=0.001, xmax=30, R=6, C=4, stock=30, mode="linear", vmax=1000, min_transfer=20)

        with pytest.raises(ValueError):
            DilutionPlan(
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
        plan = DilutionPlan(xmin=0.001, xmax=30, R=8, C=12, stock=30, mode="log", vmax=1000, min_transfer=20)

        out = plan.__repr__()

        assert out is not None
        assert isinstance(out, str)
        return

    def test_issue_48(self):
        """Columns are named 1-based, therefore the "from column" must be too."""
        plan = DilutionPlan(xmin=0.3, xmax=30, stock=30, R=1, C=3, mode="log", vmax=100, min_transfer=10)
        np.testing.assert_allclose(plan.x, [[30, 3, 0.3]])
        # Reformat instructions for easier equals comparison
        instructions = [(col, dsteps, pfrom, list(tvols)) for col, dsteps, pfrom, tvols in plan.instructions]
        assert instructions == [
            (0, 0, "stock", [100.0]),
            (1, 0, "stock", [10.0]),  # The previous column is equal to the stock!
            (2, 1, 1, [10.0]),
        ]
        plan_s = str(plan)
        lines = plan_s.split("\n")
        assert "Prepare column 1" in lines[1]
        assert "Prepare column 2" in lines[2]
        assert "Prepare column 3" in lines[3]
        assert "from stock" in lines[1]
        assert "from stock" in lines[2]
        assert "from column 2" in lines[3]
        pass

    def test_linear_plan(self) -> None:
        plan = DilutionPlan(xmin=1, xmax=10, R=10, C=1, stock=20, mode="linear", vmax=1000, min_transfer=20)

        np.testing.assert_array_equal(plan.x, plan.ideal_x)
        assert plan.max_steps == 0
        assert plan.v_stock == 2750
        assert plan.instructions[0][0] == 0
        assert plan.instructions[0][1] == 0
        assert plan.instructions[0][2] == "stock"
        np.testing.assert_array_equal(
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
        plan = DilutionPlan(xmin=0.01, xmax=10, R=4, C=3, stock=20, mode="log", vmax=1000, min_transfer=20)

        assert np.allclose(plan.x, plan.ideal_x, rtol=0.05)
        assert plan.max_steps == 2
        assert plan.v_stock == 985
        assert plan.instructions[0][0] == 0
        assert plan.instructions[0][1] == 0
        assert plan.instructions[0][2] == "stock"
        np.testing.assert_array_equal(plan.instructions[0][3], [500, 267, 142, 76])
        np.testing.assert_array_equal(plan.instructions[1][3], [82, 82, 82, 82])
        np.testing.assert_array_equal(plan.instructions[2][3], [81, 81, 81, 81])
        return

    def test_vector_vmax(self) -> None:
        plan = DilutionPlan(
            xmin=0.01, xmax=10, R=4, C=3, stock=20, mode="log", vmax=[1000, 500, 1500], min_transfer=20
        )

        assert np.allclose(plan.x, plan.ideal_x, rtol=0.05)
        assert plan.max_steps == 2
        assert plan.v_stock == 985
        assert plan.instructions[0][0] == 0
        assert plan.instructions[0][1] == 0
        assert plan.instructions[0][2] == "stock"
        np.testing.assert_array_equal(plan.instructions[0][3], [500, 267, 142, 76])
        np.testing.assert_array_equal(plan.instructions[1][3], [41, 41, 41, 41])
        np.testing.assert_array_equal(plan.instructions[2][3], [121, 121, 121, 121])
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
        plan = DilutionPlan(
            xmin=0.01,
            xmax=10,
            R=3,
            C=4,
            stock=stock_concentration,
            mode="log",
            vmax=[1000, 1900, 980, 500],
            min_transfer=50,
        )
        stock = Trough("Stock", 2, 1, min_volume=0, max_volume=10000, initial_volumes=10000)
        diluent = Trough("Diluent", 4, 2, min_volume=0, max_volume=20_000, initial_volumes=[0, 20_000])
        dilution = Labware("Dilution", 6, 8, min_volume=0, max_volume=2000)
        destination = Labware("Destination", 7, 10, min_volume=0, max_volume=1000)
        with EvoWorklist() as wl:
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
        np.testing.assert_array_almost_equal(
            plan.x, destination.composition["Stock"][: plan.R, : plan.C] * stock_concentration
        )
        assert "Mix column 0 with 75 % of its volume" in dilution.report
        assert "Mix column 1 with 50 % of its volume" in dilution.report
        return

    def test_to_worklist_hooks(self) -> None:
        stock_concentration = 123
        plan = DilutionPlan(
            xmin=1,
            xmax=123,
            R=3,
            C=4,
            stock=stock_concentration,
            mode="log",
            vmax=1000,
            min_transfer=50,
        )
        stock = Trough("Stock", 2, 1, min_volume=0, max_volume=10000, initial_volumes=10000)
        diluent = Trough("Diluent", 4, 2, min_volume=0, max_volume=10000, initial_volumes=[0, 10000])
        dilution = Labware("Dilution", 3, 4, min_volume=0, max_volume=2000)

        # Multiple destinations; transferred to via a hook
        destinations = [
            Labware("DestinationOne", 3, 2, min_volume=0, max_volume=1000),
            Labware("DestinationTwo", 3, 2, min_volume=0, max_volume=1000),
        ]

        # Split the work across two worklists (also via the hook)
        wl_one = EvoWorklist()
        wl_two = EvoWorklist()

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

        np.testing.assert_almost_equal(
            destinations[0].composition["Stock"] * stock_concentration, plan.x[:, [0, 1]]
        )
        np.testing.assert_almost_equal(
            destinations[1].composition["Stock"] * stock_concentration, plan.x[:, [2, 3]]
        )
        return


class TestUtils:
    def test_get_trough_wells(self) -> None:
        with pytest.raises(ValueError):
            get_trough_wells(n=-1, trough_wells=list("ABC"))
        with pytest.raises(ValueError):
            get_trough_wells(n=3, trough_wells=[])
        with pytest.raises(TypeError):
            get_trough_wells(n=0.5, trough_wells=list("ABC"))
        assert get_trough_wells(n=2, trough_wells="ABC") == ["ABC", "ABC"]
        np.testing.assert_array_equal(get_trough_wells(n=0, trough_wells=list("ABC")), list())
        np.testing.assert_array_equal(get_trough_wells(n=1, trough_wells=list("ABC")), list("A"))
        np.testing.assert_array_equal(get_trough_wells(n=3, trough_wells=list("ABC")), list("ABC"))
        np.testing.assert_array_equal(get_trough_wells(n=4, trough_wells=list("ABC")), list("ABCA"))
        np.testing.assert_array_equal(get_trough_wells(n=7, trough_wells=list("ABC")), list("ABCABCA"))
        return
