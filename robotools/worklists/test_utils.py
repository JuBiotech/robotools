import logging

from robotools.liquidhandling.labware import Labware, Trough
from robotools.worklists.utils import optimize_partition_by


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
