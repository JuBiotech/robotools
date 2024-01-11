import numpy as np
import pytest

from robotools.evotools.worklist import EvoWorklist
from robotools.liquidhandling.composition import (
    combine_composition,
    get_initial_composition,
    get_trough_component_names,
)
from robotools.liquidhandling.labware import Labware, Trough


class TestCompositionTracking:
    def test_get_initial_composition(self) -> None:
        wells2x3 = np.array(
            [
                ["A01", "A02", "A03"],
                ["B01", "B02", "B03"],
            ]
        )

        # Raise errors on invalid component well ids
        with pytest.raises(ValueError, match=r"Invalid component name keys: \{'G02'\}"):
            get_initial_composition("eppis", wells2x3, dict(G02="beer"), np.zeros((2, 3)))

        # Raise errors on attempts to name empty wells
        with pytest.raises(ValueError, match=r"name 'beer' was specified for eppis.A02, but"):
            get_initial_composition("eppis", wells2x3, dict(A02="beer"), np.zeros((2, 3)))

        # No components if all wells are empty
        result = get_initial_composition("eppis", wells2x3, {}, np.zeros((2, 3)))
        assert result == {}

        # Default to labware name for one-well labwares
        result = get_initial_composition("media", [["A01"]], {}, np.atleast_2d(100))
        assert "media" in result
        assert len(result) == 1

        # Assigning default component names to all wells
        result = get_initial_composition("samples", wells2x3, {}, np.ones((2, 3)))
        assert isinstance(result, dict)
        # Non-empty wells default to unique component names
        assert "samples.A01" in result
        assert "samples.B03" in result
        # Only the well with the component has 100 % of it
        np.testing.assert_array_equal(
            result["samples.B02"],
            np.array(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                ]
            ),
        )

        # Mix of user-defined and default component names
        result = get_initial_composition("samples", wells2x3, {"B01": "water"}, np.ones((2, 3)))
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
            get_trough_component_names("water", 2, ["A", "B", "C"], [20, 10])
        with pytest.raises(ValueError, match=r"initial volumes \[20, 10\] don't match"):
            get_trough_component_names("water", 3, ["A", "B", "C"], [20, 10])
        with pytest.raises(ValueError, match=r"initial volumes \[\[20], \[10\]\] don't match"):
            get_trough_component_names("water", 2, ["A", "B"], [[20], [10]])

        # It should also check that no names are given for empty columns
        with pytest.raises(ValueError, match="Empty columns must be unnamed"):
            get_trough_component_names("water", 2, ["A", "B"], [20, 0])

        # It explicitly sets names of empty wells to None
        result = get_trough_component_names("water", 1, [None], [0])
        assert result == {"A01": None}

        # And defaults to the labware name of single-column troughs
        result = get_trough_component_names("water", 1, [None], [100])
        assert result == {"A01": "water"}

        # But includes the 1-based column number in the default name for non-empty wells
        result = get_trough_component_names("stocks", 2, [None, None], [0, 50])
        assert result == {"A01": None, "A02": "stocks.column_02"}

        # User-provided names, default naming and empty-well all in one:
        result = get_trough_component_names("stocks", 4, ["acid", "base", None, None], [100, 100, 50, 0])
        assert result == {"A01": "acid", "A02": "base", "A03": "stocks.column_03", "A04": None}
        return

    def test_combine_composition(self) -> None:
        A = dict(water=1)
        B = dict(water=0.5, glucose=0.5)
        expected = {"water": (1 * 10 + 0.5 * 15) / (10 + 15), "glucose": 0.5 * 15 / (10 + 15)}
        actual = combine_composition(10, A, 15, B)
        assert actual == expected
        return

    def test_combine_unknown_composition(self) -> None:
        A = dict(water=1)
        B = None
        expected = None
        actual = combine_composition(10, A, 15, B)
        assert actual == expected
        return

    def test_labware_init(self) -> None:
        minmax = dict(min_volume=0, max_volume=4000)
        # without initial volume, there's no composition tracking
        A = Labware("glc", 6, 8, **minmax)
        assert isinstance(A.composition, dict)
        assert len(A.composition) == 0
        assert A.get_well_composition("A01") == {}

        # Single-well Labware defaults to the labware name for components
        A = Labware("x", 1, 1, **minmax, initial_volumes=300)
        assert set(A.composition) == {"x"}
        with pytest.warns(UserWarning, match="Trough class"):
            A = Labware("x", 1, 1, virtual_rows=3, **minmax, initial_volumes=300)
        assert set(A.composition) == {"x"}

        # by setting an initial volume, the well-wise liquids take part in composition tracking
        A = Labware("glc", 6, 8, **minmax, initial_volumes=100)
        assert isinstance(A.composition, dict)
        assert len(A.composition) == 48
        assert "glc.A01" in A.composition
        assert "glc.F08" in A.composition
        assert A.get_well_composition("A01") == {"glc.A01": 1}

        # Only wells with initial volumes take part
        A = Labware("test", 1, 3, **minmax, initial_volumes=[10, 0, 0], component_names=dict(A01="water"))
        assert set(A.composition) == {"water"}
        return

    def test_get_well_composition(self) -> None:
        A = Labware("glc", 6, 8, min_volume=0, max_volume=4000)
        A._composition = {
            "glc": 0.25 * np.ones_like(A.volumes),
            "water": 0.75 * np.ones_like(A.volumes),
        }
        expected = {
            "glc": 0.25,
            "water": 0.75,
        }
        assert A.get_well_composition("A01") == expected
        return

    def test_labware_add(self) -> None:
        A = Labware(
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
        water_comp = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        np.testing.assert_array_equal(A.composition["water"], water_comp)

        A.add(
            wells=["A01", "B01"],
            volumes=[10, 20],
            compositions=[
                dict(glc=0.5, water=0.5),
                dict(glc=1),
            ],
        )

        assert "water" in A.composition
        assert "glc" in A.composition
        assert A.get_well_composition("A01") == dict(water=0.75, glc=0.25)
        assert A.get_well_composition("B01") == dict(water=1 / 3, glc=2 / 3)
        return

    def test_dilution_series(self) -> None:
        A = Labware("dilutions", 1, 3, min_volume=0, max_volume=100)
        # 100 % in 1st column
        A.add(wells="A01", volumes=100, compositions=[dict(glucose=1)])
        # 10x dilution to 2nd
        A.add(wells="A02", volumes=10, compositions=[A.get_well_composition("A01")])
        A.add(wells="A02", volumes=90, compositions=[dict(water=1)])
        # 4x dilution to 3rd
        A.add(wells="A03", volumes=2.5, compositions=[A.get_well_composition("A02")])
        A.add(wells="A03", volumes=7.5, compositions=[dict(water=1)])

        np.testing.assert_array_equal(A.volumes, [[100, 100, 10]])
        np.testing.assert_array_equal(A.composition["water"], [[0, 0.9, 0.975]])
        np.testing.assert_array_equal(A.composition["glucose"], [[1, 0.1, 0.025]])
        assert A.get_well_composition("A01") == dict(glucose=1)
        assert A.get_well_composition("A02") == dict(glucose=0.1, water=0.9)
        assert A.get_well_composition("A03") == dict(glucose=0.025, water=0.975)
        return

    def test_trough_init(self) -> None:
        minmax = dict(min_volume=0, max_volume=100_000)
        # Single-column troughs use the labware name for the composition
        W = Trough("water", 2, 1, **minmax, initial_volumes=10000)
        assert set(W.composition) == {"water"}

        # Multi-column troughs get component names automatically
        A = Trough(name="water", virtual_rows=2, columns=3, **minmax, initial_volumes=[0, 200, 200])
        assert set(A.composition) == {"water.column_02", "water.column_03"}
        assert "water.column_01" not in A.composition
        assert "water.column_02" in A.composition
        assert "water.column_03" in A.composition

        # Components in troughs are named via a list, because the dict would
        # require well names and columns could lead to 0/1-based confusion.
        T = Trough(
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
        A = Trough(
            name="alice",
            virtual_rows=2,
            columns=3,
            **minmax,
            initial_volumes=[0, 200, 200],
            column_names=[None, "NaCl", None],
        )
        assert set(A.composition) == {"NaCl", "alice.column_03"}
        np.testing.assert_array_equal(A.composition["NaCl"], [[0, 1, 0]])

    def test_trough_composition(self) -> None:
        T = Trough("media", 8, 1, min_volume=1000, max_volume=25000)
        T.add(wells=T.wells, volumes=100, compositions=[dict(glucose=1)] * 8)
        T.add(wells=T.wells, volumes=900, compositions=[dict(water=1)] * 8)
        np.testing.assert_array_equal(T.composition["water"], [[0.9]])
        np.testing.assert_array_equal(T.composition["glucose"], [[0.1]])
        assert T.get_well_composition("B01") == dict(water=0.9, glucose=0.1)
        return

    def test_worklist_dilution(self) -> None:
        W = Trough("water", 4, 1, min_volume=0, max_volume=10000, initial_volumes=10000)
        G = Trough("glucose", 4, 1, min_volume=0, max_volume=10000, initial_volumes=10000)
        D = Labware("dilutions", 4, 2, min_volume=0, max_volume=10000)

        with EvoWorklist() as wl:
            # 100 % in first column
            wl.transfer(G, G.wells, D, D.wells[:, 0], volumes=[1000, 800, 600, 550])
            wl.transfer(W, W.wells, D, D.wells[:, 0], volumes=[0, 200, 400, 450])
            np.testing.assert_array_equal(D.volumes[:, 0], [1000, 1000, 1000, 1000])
            np.testing.assert_array_equal(D.composition["glucose"][:, 0], [1, 0.8, 0.6, 0.55])
            np.testing.assert_array_equal(D.composition["water"][:, 0], [0, 0.2, 0.4, 0.45])
            # 10x dilution to the 2nd column
            wl.transfer(D, D.wells[:, 0], D, D.wells[:, 1], volumes=100)
            wl.transfer(W, W.wells, D, D.wells[:, 1], volumes=900)
            np.testing.assert_array_equal(D.volumes[:, 1], [1000, 1000, 1000, 1000])
            np.testing.assert_allclose(D.composition["glucose"][:, 1], [0.1, 0.08, 0.06, 0.055])
            np.testing.assert_allclose(D.composition["water"][:, 1], [0.9, 0.92, 0.94, 0.945])

        return

    def test_worklist_distribution(self) -> None:
        W = Trough("water", 2, 1, min_volume=0, max_volume=10000, initial_volumes=10000)
        G = Trough("glucose", 2, 1, min_volume=0, max_volume=10000, initial_volumes=10000)
        D = Labware("dilutions", 2, 4, min_volume=0, max_volume=10000)

        with EvoWorklist() as wl:
            # transfer some glucose
            wl.transfer(
                G,
                G.wells[:, [0] * 4],
                D,
                D.wells,
                volumes=np.array(
                    [
                        [100, 80, 60, 55],
                        [55, 60, 80, 100],
                    ]
                ),
            )
            # fill up to 100
            wl.transfer(W, W.wells[:, [0] * 4], D, D.wells, volumes=100 - D.volumes)
            np.testing.assert_allclose(D.volumes, 100)
            np.testing.assert_array_equal(D.composition["glucose"][0, :], [1, 0.8, 0.6, 0.55])
            np.testing.assert_array_equal(D.composition["glucose"][1, :], [0.55, 0.6, 0.8, 1])
            np.testing.assert_array_equal(D.composition["water"][0, :], [0, 0.2, 0.4, 0.45])
            np.testing.assert_array_equal(D.composition["water"][1, :], [0.45, 0.4, 0.2, 0])
            # dilute 2x with water
            wl.distribute(source=W, source_column=0, destination=D, destination_wells=D.wells, volume=100)
            np.testing.assert_allclose(D.volumes, 200)
            np.testing.assert_array_equal(D.composition["glucose"][0, :], [0.5, 0.4, 0.3, 0.275])
            np.testing.assert_array_equal(D.composition["glucose"][1, :], [0.275, 0.3, 0.4, 0.5])
            np.testing.assert_array_equal(D.composition["water"][0, :], [0.5, 0.6, 0.7, 0.725])
            np.testing.assert_array_equal(D.composition["water"][1, :], [0.725, 0.7, 0.6, 0.5])

        return

    def test_worklist_mix_no_composition_change(self) -> None:
        A = Labware("solution", 2, 3, min_volume=0, max_volume=1000)
        A._composition["water"] = 0.25 * np.ones_like(A.volumes)
        A._composition["salt"] = 0.75 * np.ones_like(A.volumes)
        A._volumes = np.ones_like(A.volumes) * 500
        with EvoWorklist() as wl:
            wl.transfer(A, A.wells, A, A.wells, volumes=300)
        # make sure that the composition of the liquid is not changed
        np.testing.assert_array_equal(A.composition["water"], 0.25 * np.ones_like(A.volumes))
        np.testing.assert_array_equal(A.composition["salt"], 0.75 * np.ones_like(A.volumes))
        return
