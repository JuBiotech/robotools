import warnings

import numpy as np
import pytest

from robotools import Labware, Trough


def assert_labware_equal(original: Labware, restored: Labware):
    assert isinstance(restored, Labware)
    assert len(restored.history) == len(original.history)
    # compare history entries
    for (olabel, ovols), (rlabel, rvols) in zip(original.history, restored.history, strict=True):
        assert rlabel == olabel
        np.testing.assert_array_equal(ovols, rvols)
    # compare composition of all wells
    for w in original.wells.flatten():
        assert original.get_well_composition(w) == restored.get_well_composition(w)
    return


def test_labware_to_dict_from_dict():
    # create a labware and change it's state
    mtp = Labware(
        "emtepe",
        2,
        3,
        min_volume=30,
        max_volume=240,
        initial_volumes=50,
    )
    mtp.add("B02", 50, label="add toxic stuff", compositions=[{"toxin": 0.5, "water": 0.5}])

    # encode and recreate
    mtp_dict = mtp.to_dict()
    # the dict must not contain numpy arrays
    # that's why we can equals compare it here.
    assert mtp_dict == {
        "name": "emtepe",
        "rows": 2,
        "columns": 3,
        "min_volume": 30,
        "max_volume": 240,
        "volumes": [[50.0, 50.0, 50.0], [50.0, 100.0, 50.0]],
        "history": [[[50.0, 50.0, 50.0], [50.0, 50.0, 50.0]], [[50.0, 50.0, 50.0], [50.0, 100.0, 50.0]]],
        "labels": ["initial", "add toxic stuff"],
        "composition": {
            "emtepe.A01": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "emtepe.A02": [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            "emtepe.A03": [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            "emtepe.B01": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "emtepe.B02": [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
            "emtepe.B03": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            "toxin": [[0.0, 0.0, 0.0], [0.0, 0.25, 0.0]],
            "water": [[0.0, 0.0, 0.0], [0.0, 0.25, 0.0]],
        },
        "is_trough": False,
        "virtual_rows": None,
    }

    restored = Labware.from_dict(mtp_dict)

    assert_labware_equal(mtp, restored)
    assert isinstance(mtp, Labware)
    pass


def test_trough_to_dict_from_dict():
    # create a labware and change it's state
    orig = Trough(
        "buffers",
        8,
        2,
        min_volume=30_000,
        max_volume=200_000,
        initial_volumes=[100_000, 50_000],
        column_names=["left", "right"],
    )
    orig.add("A01", 50, label="add toxic stuff", compositions=[{"toxin": 0.5, "water": 0.5}])

    # encode and recreate
    assert orig.is_trough
    data = orig.to_dict()
    assert data["is_trough"] is True
    restored = Trough.from_dict(data)
    assert isinstance(restored, Trough)

    assert_labware_equal(orig, restored)
    pass


def test_labware_from_dict_can_return_troughs():
    # create a labware and change it's state
    orig = Trough(
        "buffers",
        8,
        2,
        min_volume=30_000,
        max_volume=200_000,
        initial_volumes=[100_000, 50_000],
        column_names=["left", "right"],
    )
    orig.add("A01", 50, label="add toxic stuff", compositions=[{"toxin": 0.5, "water": 0.5}])

    # encode and recreate
    assert orig.is_trough
    data = orig.to_dict()
    assert data["is_trough"] is True
    restored = Labware.from_dict(data)
    assert isinstance(restored, Trough)

    assert_labware_equal(orig, restored)
    pass


def test_from_dict_warns_about_downcasting_trough():
    class MyLabware(Labware):
        pass

    class MyTrough(MyLabware, Trough):
        pass

    # create a labware and change it's state
    orig = MyTrough(
        "buffers",
        8,
        2,
        min_volume=30_000,
        max_volume=200_000,
    )
    orig.add("A01", 50, label="add toxic stuff", compositions=[{"toxin": 0.5, "water": 0.5}])

    # encode and recreate
    assert orig.is_trough
    data = orig.to_dict()
    assert data["is_trough"] is True

    # robotools doesn't know about MyTrough, so we'll get a Trough with a warning
    with pytest.warns(UserWarning, match="downcasting"):
        restored = MyLabware.from_dict(data)
    # instantiation worked, but the type was downcasted
    assert isinstance(restored, Trough)
    # we can avoid downcasting by passing the trough type explicitly
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored2 = Labware.from_dict(data, cls_trough=MyTrough)
    assert isinstance(restored2, MyTrough)
    # or by calling through the custom type
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored3 = MyTrough.from_dict(data)
    assert isinstance(restored3, MyTrough)
    pass
