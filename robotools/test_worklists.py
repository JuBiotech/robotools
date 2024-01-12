import warnings

import pytest

from robotools import (
    BaseWorklist,
    CompatibilityError,
    EvoWorklist,
    FluentWorklist,
    Worklist,
)


def test_worklist_inheritance():
    assert issubclass(BaseWorklist, list)
    assert issubclass(EvoWorklist, BaseWorklist)
    assert issubclass(FluentWorklist, BaseWorklist)
    assert issubclass(Worklist, EvoWorklist)
    pass


def test_worklist_deprecation():
    with pytest.warns(DeprecationWarning, match="please switch to"):
        Worklist()
    pass


def test_recommended_instantiation():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BaseWorklist()
        EvoWorklist()
        FluentWorklist()
    pass


def test_base_worklist_cant_transfer():
    with BaseWorklist() as wl:
        with pytest.raises(CompatibilityError, match="specific, but this object"):
            wl.transfer(None, "A01", None, "B01", 100)
    pass
