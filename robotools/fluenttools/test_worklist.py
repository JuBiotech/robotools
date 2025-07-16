import pytest

from robotools.fluenttools.worklist import FluentWorklist
from robotools.liquidhandling.exceptions import (
    VolumeUnderflowError,
    VolumeUnderflowWarning,
)
from robotools.liquidhandling.labware import Labware


class TestFluentWorklist:
    def test_transfer(self):
        A = Labware("A", 3, 4, min_volume=10, max_volume=200)
        A.add("A01", 100)
        with FluentWorklist() as wl:
            wl.transfer(
                A,
                "A01",
                A,
                "B01",
                50,
            )
        assert len(wl) == 3
        a, d, w = wl
        assert a.startswith("A;")
        assert d.startswith("D;")
        assert w == "W1;"
        assert A.volumes[0, 0] == 50
        pass

    def test_input_checks(self):
        A = Labware("A", 3, 4, min_volume=10, max_volume=200, initial_volumes=150)
        with FluentWorklist() as wl:
            with pytest.raises(ValueError, match="must be equal"):
                wl.transfer(A, ["A01", "B01", "C01"], A, ["A01", "B01"], 20)
            with pytest.raises(ValueError, match="must be equal"):
                wl.transfer(A, ["A01", "B01"], A, ["A01", "B01", "C01"], 20)
            with pytest.raises(ValueError, match="must be equal"):
                wl.transfer(A, ["A01", "B01"], A, "A01", [30, 40, 25])
        pass

    def test_transfer_flush(self):
        A = Labware("A", 3, 4, min_volume=10, max_volume=200, initial_volumes=150)
        with FluentWorklist() as wl:
            wl.transfer(A, "A01", A, "B01", 20, wash_scheme="flush")
        assert len(wl) == 3
        assert wl[-1] == "F;"
        pass

    def test_transfer_on_underflow(self):
        A = Labware("A", 3, 2, min_volume=100, max_volume=2000, initial_volumes=500)
        with FluentWorklist() as wl:
            wl.transfer(A, "A01", A, "A02", 600, on_underflow="debug")
            assert A.volumes[0, 0] == 100
            assert A.volumes[0, 1] == 900  # only the aspiratable volume is transferred
            with pytest.warns(VolumeUnderflowWarning, match="500.0 - 600.0 < 100"):
                wl.transfer(A, "B01", A, "B02", 600, on_underflow="warn")
                assert A.volumes[1, 0] == 100
                assert A.volumes[1, 1] == 900
            with pytest.raises(VolumeUnderflowError, match="500.0 - 600.0 < 100"):
                wl.transfer(A, "C01", A, "C02", 600, on_underflow="raise")
