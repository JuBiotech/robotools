import pytest

from robotools import Labware, Trough
from robotools.evotools import utils


def test_get_well_position():
    plate = Labware("plate", 3, 4, min_volume=0, max_volume=50)
    assert utils.get_well_position(plate, "A01") == 1
    assert utils.get_well_position(plate, "B01") == 2
    assert utils.get_well_position(plate, "B04") == 11

    trough = Trough("trough", 2, 3, min_volume=0, max_volume=50)
    assert utils.get_well_position(trough, "A01") == 1
    assert utils.get_well_position(trough, "B01") == 2
    assert utils.get_well_position(trough, "A02") == 3
    assert utils.get_well_position(trough, "A03") == 5

    with pytest.raises(ValueError, match="not an alphanumeric well ID"):
        utils.get_well_position(trough, "A-3")

    # Currently not implemented at the Labware level:
    # megaplate = Labware("mplate", 50, 3, min_volume=0, max_volume=50)
    # assert utils.get_well_position(megaplate, "AA2") == 51
    pass
