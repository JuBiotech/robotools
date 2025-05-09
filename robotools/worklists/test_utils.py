import logging

from robotools.liquidhandling.labware import Labware, Trough
from robotools.worklists.utils import (
    non_repetitive_argsort,
    optimize_partition_by,
    partition_by_column,
)


def test_non_repetitive_argsort():
    wells = "A01,C01,B01,B01,A01,B01".split(",")
    result = non_repetitive_argsort(wells)
    assert isinstance(result, list)
    assert all(isinstance(r, int) for r in result)
    assert len(result) == len(wells)
    assert result == [0, 2, 1, 4, 3, 5]
    pass


def test_automatic_partitioning(caplog) -> None:
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
    assert "source" == optimize_partition_by(ST, D, "auto", "Trough source")
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


def test_partition_by_column_does_not_repeat_wells():
    wells = "C01,B01,A01,B01,A01,C01,D02,C02".split(",")
    cgroups = partition_by_column(
        sources=wells,
        destinations=["A01"] * 8,
        volumes=[1, 2, 3, 4, 5, 6, 7, 8],
        partition_by="source",
    )
    assert len(cgroups) == 2
    assert cgroups[0][0] == "A01,B01,C01,A01,B01,C01".split(",")
    assert cgroups[0][1] == ["A01"] * 6
    assert cgroups[0][2] == [3, 2, 1, 5, 4, 6]
    assert cgroups[1][0] == "C02,D02".split(",")
    assert cgroups[1][1] == ["A01"] * 2
    assert cgroups[1][2] == [8, 7]
    pass
