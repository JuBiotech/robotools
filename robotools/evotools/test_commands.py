import numpy as np
import pytest

from robotools.evotools.commands import (
    evo_aspirate,
    evo_dispense,
    evo_get_selection,
    evo_wash,
    prepare_evo_aspirate_dispense_parameters,
    prepare_evo_wash_parameters,
)
from robotools.evotools.types import Tip


def test_evo_get_selection():
    with pytest.raises(ValueError, match="from more than one column"):
        evo_get_selection(
            rows=2,
            cols=3,
            selected=np.array(
                [
                    [True, False, False],
                    [False, True, False],
                ]
            ),
        )
    selection = evo_get_selection(
        rows=2,
        cols=3,
        selected=np.array(
            [
                [True, False, False],
                [True, False, False],
            ]
        ),
    )
    assert selection == "03023"
    pass


class TestPrepareEvoAspirateDispenseParameters:
    def test_prepare_evo_aspirate_dispense_parameters(self):
        # test wells argument checks
        with pytest.raises(ValueError, match="Invalid wells:"):
            prepare_evo_aspirate_dispense_parameters(
                wells="A01",
                labware_position=(38, 2),
                volume=15,
                liquid_class="Water_DispZmax-1_AspZmax-1",
                tips=[1, 2],
                arm=0,
            )
        # test labware_position argument checks
        with pytest.raises(ValueError, match="second number in labware_position"):
            prepare_evo_aspirate_dispense_parameters(
                wells=["A01", "B01"],
                labware_position=(38, 0),
                volume=15,
                liquid_class="Water_DispZmax-1_AspZmax-1",
                tips=[1, 2],
                arm=0,
            )
        with pytest.raises(ValueError, match="first number in labware_position"):
            prepare_evo_aspirate_dispense_parameters(
                wells=["A01", "B01"],
                labware_position=("a", 2),
                volume=15,
                liquid_class="Water_DispZmax-1_AspZmax-1",
                tips=[1, 2],
                arm=0,
            )
        # test liquid_class argument checks
        with pytest.raises(ValueError, match="Invalid liquid_class:"):
            prepare_evo_aspirate_dispense_parameters(
                wells=["A01", "B01"],
                labware_position=(38, 2),
                volume=15,
                liquid_class=["Water_DispZmax-1_AspZmax-1"],
                tips=[1, 2],
                arm=0,
            )
        with pytest.raises(ValueError, match="Invalid liquid_class:"):
            prepare_evo_aspirate_dispense_parameters(
                wells=["A01", "B01"],
                labware_position=(38, 2),
                volume=15,
                liquid_class="Water;DispZmax-1;AspZmax-1",
                tips=[1, 2],
                arm=0,
            )
        # test tips argument checks
        with pytest.raises(ValueError, match="Invalid type of tips:"):
            prepare_evo_aspirate_dispense_parameters(
                wells=["A01", "B01"],
                labware_position=(38, 2),
                volume=15,
                liquid_class="Water_DispZmax-1_AspZmax-1",
                tips=[1, "2"],
                arm=0,
            )
        _, _, _, _, tips = prepare_evo_aspirate_dispense_parameters(
            wells=["A01", "B01"],
            labware_position=(38, 2),
            volume=15,
            liquid_class="Water_DispZmax-1_AspZmax-1",
            tips=[1, 2],
            arm=0,
        )
        if not all(isinstance(n, Tip) for n in tips):
            raise TypeError(
                f"Even after completing the prepare_evo_aspirate_dispense_parameters method, not all tips are type Tip."
            )
        # test volume argument checks
        with pytest.raises(ValueError, match="Invalid volume:"):
            prepare_evo_aspirate_dispense_parameters(
                wells=["A01", "B01"],
                labware_position=(38, 2),
                volume="volume",
                liquid_class="Water_DispZmax-1_AspZmax-1",
                tips=[1, 2],
                arm=0,
            )
        with pytest.raises(ValueError, match="Invalid volume:"):
            prepare_evo_aspirate_dispense_parameters(
                wells=["A01", "B01"],
                labware_position=(38, 2),
                volume=-10,
                liquid_class="Water_DispZmax-1_AspZmax-1",
                tips=[1, 2],
                arm=0,
            )
        with pytest.raises(ValueError, match="Invalid volume:"):
            prepare_evo_aspirate_dispense_parameters(
                wells=["A01", "B01"],
                labware_position=(38, 2),
                volume=7158279,
                liquid_class="Water_DispZmax-1_AspZmax-1",
                tips=[1, 2],
                arm=0,
            )

        # test complete prepare_evo_aspirate_dispense_parameters() command
        actual = prepare_evo_aspirate_dispense_parameters(
            wells=["E01", "F01", "G01"],
            labware_position=(38, 3),
            volume=750,
            liquid_class="Water_DispZmax_AspZmax",
            tips=[5, 6, 7],
            arm=0,
        )
        expected = (
            ["E01", "F01", "G01"],
            (38, 2),
            [750.0, 750.0, 750.0],
            "Water_DispZmax_AspZmax",
            [Tip.T5, Tip.T6, Tip.T7],
        )
        assert actual == expected


class TestEvoAspirate:
    def test_evo_aspirate1(self) -> None:
        cmd = evo_aspirate(
            n_rows=8,
            n_columns=12,
            wells=["E01", "F01", "G01"],
            labware_position=(38, 3),
            tips=[5, 6, 7],
            volume=750,
            liquid_class="Water_DispZmax_AspZmax",
            max_volume=950,
        )
        exp = 'B;Aspirate(112,"Water_DispZmax_AspZmax",0,0,0,0,"750.0","750.0","750.0",0,0,0,0,0,38,2,1,"0C08\xa00000000000000",0,0);'
        assert cmd == exp
        return

    def test_evo_aspirate2(self) -> None:
        cmd = evo_aspirate(
            n_rows=8,
            n_columns=12,
            wells=["E01", "F01", "G01"],
            labware_position=(38, 3),
            tips=[5, 6, 7],
            volume=[750, 730, 710],
            liquid_class="Water_DispZmax_AspZmax",
            max_volume=950,
        )
        exp = 'B;Aspirate(112,"Water_DispZmax_AspZmax",0,0,0,0,"750","730","710",0,0,0,0,0,38,2,1,"0C08\xa00000000000000",0,0);'
        assert cmd == exp
        return


class TestEvoDispense:
    def test_evo_dispense1(self) -> None:
        cmd = evo_dispense(
            n_rows=8,
            n_columns=12,
            wells=["E01", "F01", "G01"],
            labware_position=(38, 3),
            tips=[5, 6, 7],
            volume=750,
            liquid_class="Water_DispZmax_AspZmax",
            max_volume=950,
        )
        exp = 'B;Dispense(112,"Water_DispZmax_AspZmax",0,0,0,0,"750.0","750.0","750.0",0,0,0,0,0,38,2,1,"0C08\xa00000000000000",0,0);'
        assert cmd == exp
        return

    def test_evo_dispense2(self) -> None:
        cmd = evo_dispense(
            n_rows=8,
            n_columns=12,
            wells=["E01", "F01", "G01"],
            labware_position=(38, 3),
            tips=[5, 6, 7],
            volume=[750, 730, 710],
            liquid_class="Water_DispZmax_AspZmax",
            max_volume=950,
        )
        exp = 'B;Dispense(112,"Water_DispZmax_AspZmax",0,0,0,0,"750","730","710",0,0,0,0,0,38,2,1,"0C08\xa00000000000000",0,0);'
        assert cmd == exp
        return


class TestEvoWash:
    def test_prepare_evo_wash_parameters_checking(self):
        # test tips argument checks
        tips, _, _, _, _, _, _, _, _, _, _, _, _ = prepare_evo_wash_parameters(
            tips=[1, 2],
            waste_location=(52, 2),
            cleaner_location=(52, 1),
        )
        if not all(isinstance(n, Tip) for n in tips):
            raise TypeError(
                f"Even after completing the prepare_evo_aspirate_dispense_parameters method, not all tips are type Tip."
            )

        # test waste_location argument checks
        with pytest.raises(ValueError, match="Grid \\(first number in waste_location tuple\\)"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(68, 2),
                cleaner_location=(52, 1),
            )
        with pytest.raises(ValueError, match="Grid \\(first number in waste_location tuple\\)"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(0, 2),
                cleaner_location=(52, 1),
            )
        with pytest.raises(ValueError, match="Grid \\(first number in waste_location tuple\\)"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(1.7, 2),
                cleaner_location=(52, 1),
            )
        with pytest.raises(ValueError, match="Site \\(second number in waste_location tuple\\)"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 0),
                cleaner_location=(52, 1),
            )
        with pytest.raises(ValueError, match="Site \\(second number in waste_location tuple\\)"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 129),
                cleaner_location=(52, 1),
            )
        with pytest.raises(ValueError, match="Site \\(second number in waste_location tuple\\)"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 1.7),
                cleaner_location=(52, 1),
            )

        # test cleaner_location argument checks
        with pytest.raises(ValueError, match="Grid \\(first number in cleaner_location tuple\\)"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 1),
                cleaner_location=(68, 1),
            )
        with pytest.raises(ValueError, match="Grid \\(first number in cleaner_location tuple\\)"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 1),
                cleaner_location=(0, 1),
            )
        with pytest.raises(ValueError, match="Grid \\(first number in cleaner_location tuple\\)"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 1),
                cleaner_location=(1.7, 1),
            )
        with pytest.raises(ValueError, match="Site \\(second number in cleaner_location tuple\\)"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 1),
                cleaner_location=(52, 0),
            )
        with pytest.raises(ValueError, match="Site \\(second number in cleaner_location tuple\\)"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 1),
                cleaner_location=(52, 129),
            )
        with pytest.raises(ValueError, match="Site \\(second number in cleaner_location tuple\\)"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 1),
                cleaner_location=(52, 1.7),
            )

        # test arm argument check
        with pytest.raises(ValueError, match="Parameter arm"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                arm=2,
            )

        # test waste_vol argument check
        with pytest.raises(ValueError, match="waste_vol has to be a float"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                waste_vol=-1.0,
            )
        with pytest.raises(ValueError, match="waste_vol has to be a float"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                waste_vol=101.0,
            )
        with pytest.raises(ValueError, match="waste_vol has to be a float"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                waste_vol=1,
            )

        # test waste_delay argument check
        with pytest.raises(ValueError, match="waste_delay has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                waste_delay=-1,
            )
        with pytest.raises(ValueError, match="waste_delay has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                waste_delay=1001,
            )
        with pytest.raises(ValueError, match="waste_delay has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                waste_delay=10.0,
            )

        # test cleaner_vol argument check
        with pytest.raises(ValueError, match="cleaner_vol has to be a float"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                cleaner_vol=-1.0,
            )
        with pytest.raises(ValueError, match="cleaner_vol has to be a float"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                cleaner_vol=101.0,
            )
        with pytest.raises(ValueError, match="cleaner_vol has to be a float"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                cleaner_vol=1,
            )

        # test cleaner_delay argument check
        with pytest.raises(ValueError, match="cleaner_delay has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                cleaner_delay=-1,
            )
        with pytest.raises(ValueError, match="cleaner_delay has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                cleaner_delay=1001,
            )
        with pytest.raises(ValueError, match="cleaner_delay has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                cleaner_delay=10.0,
            )

        # test airgap argument check
        with pytest.raises(ValueError, match="airgap has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                airgap=-1,
            )
        with pytest.raises(ValueError, match="airgap has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                airgap=101,
            )
        with pytest.raises(ValueError, match="airgap has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                airgap=10.0,
            )

        # test airgap_speed argument check
        with pytest.raises(ValueError, match="airgap_speed has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                airgap_speed=0,
            )
        with pytest.raises(ValueError, match="airgap_speed has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                airgap_speed=1001,
            )
        with pytest.raises(ValueError, match="airgap_speed has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                airgap_speed=10.0,
            )

        # test retract_speed argument check
        with pytest.raises(ValueError, match="retract_speed has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                retract_speed=0,
            )
        with pytest.raises(ValueError, match="retract_speed has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                retract_speed=101,
            )
        with pytest.raises(ValueError, match="retract_speed has to be an int"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                retract_speed=10.0,
            )

        # test fastwash argument check
        with pytest.raises(ValueError, match="Parameter fastwash"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                fastwash=2,
            )
        with pytest.raises(ValueError, match="Parameter fastwash"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                fastwash=1.0,
            )

        # test low_volume argument check
        with pytest.raises(ValueError, match="Parameter low_volume"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                low_volume=2,
            )
        with pytest.raises(ValueError, match="Parameter low_volume"):
            prepare_evo_wash_parameters(
                tips=[1, 2],
                waste_location=(52, 2),
                cleaner_location=(52, 1),
                low_volume=1.0,
            )

        # test complete prepare_evo_wash_parameters() command
        actual = prepare_evo_wash_parameters(
            tips=[1, 2, 3, 4, 5, 6, 7, 8],
            waste_location=(52, 2),
            cleaner_location=(52, 1),
        )
        expected = (
            [
                Tip.T1,
                Tip.T2,
                Tip.T3,
                Tip.T4,
                Tip.T5,
                Tip.T6,
                Tip.T7,
                Tip.T8,
            ],
            (52, 1),
            (52, 0),
            0,
            3.0,
            500,
            4.0,
            500,
            10,
            70,
            30,
            1,
            0,
        )
        assert actual == expected
        return

    def test_evo_wash(self) -> None:
        cmd = evo_wash(
            tips=[1, 2, 3, 4, 5, 6, 7, 8],
            waste_location=(52, 2),
            cleaner_location=(52, 1),
        )
        assert cmd == 'B;Wash(255,52,1,52,0,"3.0",500,"4.0",500,10,70,30,1,0,1000,0);'
        return
