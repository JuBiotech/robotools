"""This module implements functions to create advanced worklist commands."""
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from robotools.evotools.types import Tip, int_to_tip
from robotools.evotools.utils import to_hex
from robotools.worklists.exceptions import InvalidOperationError

from .. import transform

__all__ = (
    "evo_make_selection_array",
    "evo_get_selection",
    "evo_aspirate",
    "evo_dispense",
    "evo_wash",
)

MAX_DILUTOR_VOLUME = 950
""""Maximum dilutor volume in µL"""


def evo_make_selection_array(rows: int, columns: int, wells: Union[Iterable[str], np.ndarray]) -> np.ndarray:
    """Translate well IDs to a numpy array with 1s (selected) and 0s (not selected).

    Parameters
    ----------
    rows : int
        Number of rows of target labware object
    cols : int
        Number of columns of target labware object
    wells : Union[Iterable[str], np.ndarray]
        Selected wells by well IDs as strings (e.g. ["A01", "B01"])

    Returns
    -------
    selection_array : np.ndarray
        Numpy array in labware dimensions with selected wells as 1 and others as 0
    """
    # create array with a shape beffiting the labware dimensions
    selection_array = np.zeros((rows, columns))
    # get a dictionary with the "coordinates" of well IDs (A01, B01 etc) as tuples
    well_index_dict = transform.make_well_index_dict(rows, columns)
    # insert 1s for all selected wells
    for well in np.asarray(wells).flatten():
        selection_array[well_index_dict[well]] = 1
    return selection_array


def evo_get_selection(rows: int, cols: int, selected: np.ndarray) -> str:
    """Function to generate the code string for the well selection of pipetting actions in EvoWare scripts (.esc).

    Adapted from the C++ function detailed in the EvoWare manual to Python by Martin Beyß (except the test at the end).

    Parameters
    ----------
    rows : int
        Number of rows of target labware object
    cols : int
        Number of columns of target labware object
    selected : np.ndarray
        Numpy array in labware dimensions with selected wells as 1 and others as 0 (from evo_make_selection_array)

    Returns
    -------
    selection : str
        Code string for well selection of pipetting actions in EvoWare scripts (.esc)
    """
    # apply bit mask with 7 bits, adapted from function detailed in EvoWare manual
    selection = f"0{to_hex(cols)}{rows:02d}"
    bit_counter = 0
    bit_mask = 0
    for x in range(cols):
        for y in range(rows):
            if selected[y, x] == 1:
                bit_mask |= 1 << bit_counter
            bit_counter += 1
            if bit_counter > 6:
                selection += chr(bit_mask + 48)
                bit_counter = 0
                bit_mask = 0
    if bit_counter > 0:
        selection += chr(bit_mask + 48)

    # check if wells from more than one column are selected and raise Exception if so
    check = 0
    for column in selected.transpose():
        if sum(column) >= 1:
            check += 1
    if check >= 2:
        raise ValueError(
            "Wells from more than one column are selected.\nSelect only wells from one column per pipetting action."
        )

    return selection


def prepare_evo_aspirate_dispense_parameters(
    wells: Union[str, Sequence[str], np.ndarray],
    *,
    labware_position: Tuple[int, int],
    volume: Union[float, Sequence[float], int],
    liquid_class: str,
    tips: Union[Sequence[Tip], Sequence[int]],
    arm: int,
    max_volume: Optional[Union[int, float]] = None,
) -> Tuple[List[str], Tuple[int, int], List[float], str, List[Tip]]:
    # wells, labware_position, volume, liquid_class, tecan_tips
    """Validates and prepares aspirate/dispense parameters.

    Parameters
    ----------
    wells : list of str
        List with target well ID(s)
    labware_position : tuple
        Grid position of the target labware on the robotic deck and site position on its carrier, e.g. labware on grid 38, site 2 -> (38,2).
        NOTE: The site numbering starts at 1.
    volume : int, float or list
        Volume in microliters (will be rounded to 2 decimal places); if several tips are used, these tips may aspirate individual volumes -> use list in these cases
    liquid_class : str, optional
        Overwrites the liquid class for this step (max 32 characters)
    tips : list of int
        Tip(s) that will be selected (out of tips 1-8)
    arm : int
        Which LiHa to use, if more than one is available
    max_volume : int, optional
        Maximum allowed volume

    Returns
    -------
    wells : list of str
        List with target well ID(s)
    labware_position : tuple
        Grid position of the target labware on the robotic deck and site position on its carrier, e.g. labware on grid 38, site 2 -> (38,2).
        NOTE: The returned site number starts at 1.
    volume : list
        Volume in microliters (will be rounded to 2 decimal places); if several tips are used, these tips may aspirate individual volumes -> use list in these cases
    liquid_class : str, optional
        Overwrites the liquid class for this step (max 32 characters)
    tips : list of int
        Tip(s) that will be selected (out of tips 1-8)
    """
    if wells is None:
        raise ValueError("Missing required parameter: wells")
    if not isinstance(wells, (str, list, tuple, np.ndarray)):
        raise ValueError(f"Invalid wells: {wells}")
    wells_list = list(np.atleast_1d(wells).flatten("F"))
    if not len(wells_list) == len(tips):
        raise ValueError(f"Invalid wells: wells and tips need to have the same length.")
    if labware_position is None:
        raise ValueError("Missing required parameter: position")
    grid, site = labware_position
    if not isinstance(grid, int) or not 1 <= grid <= 67:
        raise ValueError("Grid (first number in labware_position tuple) has to be an int from 1 - 67.")
    if not isinstance(site, int) or not 1 <= site <= 128:
        raise ValueError("Site (second number in labware_position tuple) has to be an int from 1 - 128.")
    labware_position = (grid, site - 1)

    if volume is None:
        raise ValueError("Missing required parameter: volume")
    if isinstance(volume, list):
        for vol in volume:
            try:
                vol = float(vol)
            except:
                raise ValueError(f"Invalid volume: {vol}")
            if vol < 0 or vol > 7158278 or np.isnan(vol):
                raise ValueError(f"Invalid volume: {vol}")
            if max_volume is not None and vol > max_volume:
                raise InvalidOperationError(f"Invalid volume: volume of {vol} exceeds max_volume.")
        if not len(volume) == len(tips) == len(wells_list):
            raise Exception(
                f"Invalid volume: Tips, wells, and volume lists have different lengths ({len(tips)}, {len(wells_list)} and {len(volume)}, respectively)."
            )
    elif isinstance(volume, (float, int)):
        # test volume like in the list section
        if volume < 0 or volume > 7158278 or np.isnan(volume):
            raise ValueError(f"Invalid volume: {volume}")
        if max_volume is not None and volume > max_volume:
            raise InvalidOperationError(f"Invalid volume: volume of {volume} exceeds max_volume.")
        # convert volume to list and multiply list to reach identical length as wells
        volume = [float(volume)] * len(wells_list)
    else:
        raise ValueError(f"Invalid volume: {volume}")

    # apply rounding and corrections for the right string formatting
    volume_list: List[float] = np.round(volume, decimals=2).tolist()

    if liquid_class is None:
        raise ValueError(f"Missing required parameter: liquid_class")
    if not isinstance(liquid_class, str) or ";" in liquid_class:
        raise ValueError(f"Invalid liquid_class: {liquid_class}")

    if tips is None:
        raise ValueError(f"Missing required parameter: tips")
    for tip in tips:
        if not isinstance(tip, (int, Tip)):
            raise ValueError(f"Invalid type of tips: {type(tip)}. Has to be int or Tip.")
    tecan_tips = []
    for tip in tips:
        if isinstance(tip, int) and not isinstance(tip, Tip):
            # User-specified integers from 1-8 need to be converted to Tecan logic
            tip = int_to_tip(tip)
        tecan_tips.append(tip)

    if arm is None:
        raise ValueError("Missing required paramter: arm")
    if not arm == 0 and not arm == 1:
        raise ValueError("Parameter arm has to be 0 (LiHa 1) or 1 (LiHa 2).")

    return wells_list, labware_position, volume_list, liquid_class, tecan_tips


def evo_aspirate(
    *,
    n_rows: int,
    n_columns: int,
    wells: Union[str, Sequence[str]],
    labware_position: Tuple[int, int],
    volume: Union[float, Sequence[float], int],
    liquid_class: str,
    tips: Union[Sequence[Tip], Sequence[int]],
    arm: int = 0,
    max_volume: Optional[Union[int, float]] = np.nan,
) -> str:
    """Command for aspirating with the EvoWARE aspirate command WITHOUT digital volume tracking.

    As many wells in one column may be selected as your liquid handling arm has pipettes.
    This method generates the full command (as can be observed when opening a .esc file with an editor) and calls upon other functions to create the code string
    specifying the target wells.

    Parameters
    ----------
    n_rows
        Number of rows in the labware.
    n_columns
        Number of columns in the labware.
    wells : list of str
        List with target well ID(s)
    labware_position : tuple
        Grid position of the target labware on the robotic deck and site position on its carrier, e.g. labware on grid 38, site 2 -> (38,2).
        NOTE: The site numbering starts at 1.
    volume : int, float or list
        Volume in microliters (will be rounded to 2 decimal places); if several tips are used, these tips may aspirate individual volumes -> use list in these cases
    liquid_class : str, optional
        Overwrites the liquid class for this step (max 32 characters)
    tips : list
        Tip(s) that will be selected; use either a list with integers from 1 - 8 or with tip.T1 - tip.T8
    arm : int
        Which LiHa to use, if more than one is available
    max_volume
        Maximum allowed dilutor volume.
    """
    # update max_volume (if no value was given) according to the maximum dilutor volume stated at the top
    if np.isnan(max_volume):
        max_volume = MAX_DILUTOR_VOLUME

    # perform consistency checks
    (wells, labware_position, volume, liquid_class, tips,) = prepare_evo_aspirate_dispense_parameters(
        wells=wells,
        labware_position=labware_position,
        volume=volume,
        liquid_class=liquid_class,
        tips=tips,
        arm=arm,
        max_volume=max_volume,
    )

    # calculate tip_selection based on tips argument (tips are converted to evotools.Tip in _prepare_evo_aspirate_dispense_parameters)
    tip_selection = 0
    for tip in tips:
        tip_selection += tip.value

    # prepare volume section (volume is converted to list in _prepare_evo_aspirate_dispense_parameters)
    tip_volumes = ""
    for tipv in [1, 2, 4, 8, 16, 32, 64, 128]:
        if tipv in [tecantip.value for tecantip in tips]:
            tip_volumes += f'"{volume[0]}",'
            volume.pop(0)
        else:
            tip_volumes += "0,"

    # convert selection from list of well ids to numpy array with same dimensions as target labware (1: well is selected, 0: well is not selected)
    selected = evo_make_selection_array(n_rows, n_columns, wells)
    # create code string containing information about target well(s)
    code_string = evo_get_selection(n_rows, n_columns, selected)
    return f'B;Aspirate({tip_selection},"{liquid_class}",{tip_volumes}0,0,0,0,{labware_position[0]},{labware_position[1]},1,"{code_string}",0,{arm});'


def evo_dispense(
    *,
    n_rows: int,
    n_columns: int,
    wells: Union[str, Sequence[str]],
    labware_position: Tuple[int, int],
    volume: Union[float, Sequence[float], int],
    liquid_class: str,
    tips: Union[Sequence[Tip], Sequence[int]],
    arm: int = 0,
    max_volume: Optional[Union[int, float]] = np.nan,
) -> str:
    """Command for dispensing using the EvoWARE dispense command WITHOUT digital volume tracking.

    As many wells in one column may be selected as your liquid handling arm has pipettes.
    This method generates the full command (as can be observed when opening a .esc file with an editor) and calls upon other functions to create the code string
    specifying the target wells.

    Parameters
    ----------
    n_rows
        Number of rows in the labware.
    n_columns
        Number of columns in the labware.
    wells : list of str
        List with target well ID(s)
    labware_position : tuple
        Grid position of the target labware on the robotic deck and site position on its carrier, e.g. labware on grid 38, site 2 -> (38,2).
        NOTE: The site numbering starts at 1.
    volume : int, float or list
        Volume in microliters (will be rounded to 2 decimal places); if several tips are used, these tips may aspirate individual volumes -> use list in these cases
    liquid_class : str, optional
        Overwrites the liquid class for this step (max 32 characters)
    tips : list
        Tip(s) that will be selected; use either a list with integers from 1 - 8 or with tip.T1 - tip.T8
    arm : int
        Which LiHa to use, if more than one is available
    max_volume
        Maximum allowed dilutor volume.
    """
    # update max_volume (if no value was given) according to the maximum dilutor volume stated at the top
    if np.isnan(max_volume):
        max_volume = MAX_DILUTOR_VOLUME

    # perform consistency checks
    (wells, labware_position, volume, liquid_class, tips,) = prepare_evo_aspirate_dispense_parameters(
        wells=wells,
        labware_position=labware_position,
        volume=volume,
        liquid_class=liquid_class,
        tips=tips,
        arm=arm,
        max_volume=max_volume,
    )

    # calculate tip_selection based on tips argument (tips are converted to evotools.Tip in _prepare_evo_aspirate_dispense_parameters)
    tip_selection = 0
    for tip in tips:
        tip_selection += tip.value

    # prepare volume section (volume is converted to list in _prepare_evo_aspirate_dispense_parameters)
    tip_volumes = ""
    for tipv in [1, 2, 4, 8, 16, 32, 64, 128]:
        if tipv in [tecantip.value for tecantip in tips]:
            tip_volumes += f'"{volume[0]}",'
            volume.pop(0)
        else:
            tip_volumes += "0,"

    # convert selection from list of well ids to numpy array with same dimensions as target labware (1: well is selected, 0: well is not selected)
    selected = evo_make_selection_array(n_rows, n_columns, wells)
    # create code string containing information about target well(s)
    code_string = evo_get_selection(n_rows, n_columns, selected)
    return f'B;Dispense({tip_selection},"{liquid_class}",{tip_volumes}0,0,0,0,{labware_position[0]},{labware_position[1]},1,"{code_string}",0,{arm});'


def prepare_evo_wash_parameters(
    *,
    tips: Union[List[Tip], List[int]],
    waste_location: Tuple[int, int],
    cleaner_location: Tuple[int, int],
    arm: int = 0,
    waste_vol: float = 3.0,
    waste_delay: int = 500,
    cleaner_vol: float = 4.0,
    cleaner_delay: int = 500,
    airgap: int = 10,
    airgap_speed: int = 70,
    retract_speed: int = 30,
    fastwash: int = 1,
    low_volume: int = 0,
) -> Tuple[List[Tip], Tuple[int, int], Tuple[int, int], int, float, int, float, int, int, int, int, int, int]:
    """Validates and prepares aspirate/dispense parameters.

    Parameters
    ----------
    tips : list
        Tip(s) that will be selected; use either a list with integers from 1 - 8 or with tip.T1 - tip.T8
    waste_location : tuple
        Tuple with grid position (1-67) and site number (1-128) of waste as integers
    cleaner_location : tuple
        Tuple with grid position (1-67) and site number (1-128) of cleaner as integers
    arm : int
        number of the LiHa performing the action: 0 = LiHa 1, 1 = LiHa 2
    waste_vol: float
        Volume in waste in mL (0-100)
    waste_delay : int
        Delay before closing valves in waste in ms (0-1000)
    cleaner_vol: float
        Volume in cleaner in mL (0-100)
    cleaner_delay : int
        Delay before closing valves in cleaner in ms (0-1000)
    airgap : int
        Volume of airgap in µL which is aspirated after washing the tips (system trailing airgap) (0-100)
    airgap_speed : int
        Speed of airgap aspiration in µL/s (1-1000)
    retract_speed : int
        Retract speed in mm/s (1-100)
    fastwash : int
        Use fast-wash module = 1, don't use it = 0
    low_volume : int
        Use pinch valves = 1, don't use them = 0

    Returns
    -------
    tips : list
        Tip(s) that will be selected; have been converted to tip.T1 - tip.T8 here if they weren't originally formatted that way
    waste_location : tuple
        Tuple with grid position (1-67) and site number (0-127) of waste as integers
    cleaner_location : tuple
        Tuple with grid position (1-67) and site number (0-127) of cleaner as integers
    arm : int
        number of the LiHa performing the action: 0 = LiHa 1, 1 = LiHa 2
    waste_vol: float
        Volume in waste in mL (0-100)
    waste_delay : int
        Delay before closing valves in waste in ms (0-1000)
    cleaner_vol: float
        Volume in cleaner in mL (0-100)
    cleaner_delay : int
        Delay before closing valves in cleaner in ms (0-1000)
    airgap : int
        Volume of airgap in µL which is aspirated after washing the tips (system trailing airgap) (0-100)
    airgap_speed : int
        Speed of airgap aspiration in µL/s (1-1000)
    retract_speed : int
        Retract speed in mm/s (1-100)
    fastwash : int
        Use fast-wash module = 1, don't use it = 0
    low_volume : int
        Use pinch valves = 1, don't use them = 0
    """
    if tips is None:
        raise ValueError("Missing required parameter: tips")

    tecan_tips: List[Tip] = []
    for tip in tips:
        if isinstance(tip, int) and not isinstance(tip, Tip):
            # User-specified integers from 1-8 need to be converted to Tecan logic
            tip = int_to_tip(tip)
        tecan_tips.append(tip)

    if waste_location is None:
        raise ValueError("Missing required parameter: waste_location")
    grid, site = waste_location
    if not isinstance(grid, int) or not 1 <= grid <= 67:
        raise ValueError("Grid (first number in waste_location tuple) has to be an int from 1 - 67.")
    if not isinstance(site, int) or not 1 <= site <= 128:
        raise ValueError("Site (second number in waste_location tuple) has to be an int from 1 - 128.")
    waste_location = (grid, site - 1)

    if cleaner_location is None:
        raise ValueError("Missing required parameter: cleaner_location")
    grid, site = cleaner_location
    if not isinstance(grid, int) or not 1 <= grid <= 67:
        raise ValueError("Grid (first number in cleaner_location tuple) has to be an int from 1 - 67.")
    if not isinstance(site, int) or not 1 <= site <= 128:
        raise ValueError("Site (second number in cleaner_location tuple) has to be an int from 1 - 128.")
    cleaner_location = (grid, site - 1)

    if arm is None:
        raise ValueError("Missing required paramter: arm")
    if not arm == 0 and not arm == 1:
        raise ValueError("Parameter arm has to be 0 (LiHa 1) or 1 (LiHa 2).")

    if waste_vol is None:
        raise ValueError("Missing required parameter: waste_vol")
    if not isinstance(waste_vol, float) or not 0 <= waste_vol <= 100:
        raise ValueError("waste_vol has to be a float from 0 - 100.")
    # round waste_vol to the first decimal (pre-requisite for Tecan's wash command)
    waste_vol = np.round(waste_vol, 1)

    if waste_delay is None:
        raise ValueError("Missing required parameter: waste_delay")
    if not isinstance(waste_delay, int) or not 0 <= waste_delay <= 1000:
        raise ValueError("waste_delay has to be an int from 0 - 1000.")

    if cleaner_vol is None:
        raise ValueError("Missing required parameter: cleaner_vol")
    if not isinstance(cleaner_vol, float) or not 0 <= cleaner_vol <= 100:
        raise ValueError("cleaner_vol has to be a float from 0 - 100.")
    # round cleaner_vol to the first decimal (pre-requisite for Tecan's wash command)
    cleaner_vol = np.round(cleaner_vol, 1)

    if cleaner_delay is None:
        raise ValueError("Missing required parameter: cleaner_delay")
    if not isinstance(cleaner_delay, int) or not 0 <= cleaner_delay <= 1000:
        raise ValueError("cleaner_delay has to be an int from 0 - 1000.")

    if airgap is None:
        raise ValueError("Missing required parameter: airgap")
    if not isinstance(airgap, int) or not 0 <= airgap <= 100:
        raise ValueError("airgap has to be an int from 0 - 100.")

    if airgap_speed is None:
        raise ValueError("Missing required parameter: airgap_speed")
    if not isinstance(airgap_speed, int) or not 1 <= airgap_speed <= 1000:
        raise ValueError("airgap_speed has to be an int from 1 - 1000.")

    if retract_speed is None:
        raise ValueError("Missing required parameter: retract_speed")
    if not isinstance(retract_speed, int) or not 1 <= retract_speed <= 100:
        raise ValueError("retract_speed has to be an int from 1 - 100.")

    if fastwash is None:
        raise ValueError("Missing required paramter: fastwash")
    if not isinstance(fastwash, int):
        raise ValueError("Parameter fastwash is not int.")
    if not fastwash == 0 and not fastwash == 1:
        raise ValueError("Parameter fastwash has to be 0 (no fast-wash) or 1 (use fast-wash).")

    if low_volume is None:
        raise ValueError("Missing required paramter: low_volume")
    if not isinstance(low_volume, int):
        raise ValueError("Parameter low_volume is not int.")
    if not low_volume == 0 and not low_volume == 1:
        raise ValueError("Parameter low_volume has to be 0 (no fast-wash) or 1 (use fast-wash).")

    return (
        tecan_tips,
        waste_location,
        cleaner_location,
        arm,
        waste_vol,
        waste_delay,
        cleaner_vol,
        cleaner_delay,
        airgap,
        airgap_speed,
        retract_speed,
        fastwash,
        low_volume,
    )


def evo_wash(
    *,
    tips: Union[List[Tip], List[int]],
    waste_location: Tuple[int, int],
    cleaner_location: Tuple[int, int],
    arm: int = 0,
    waste_vol: float = 3.0,
    waste_delay: int = 500,
    cleaner_vol: float = 4.0,
    cleaner_delay: int = 500,
    airgap: int = 10,
    airgap_speed: int = 70,
    retract_speed: int = 30,
    fastwash: int = 1,
    low_volume: int = 0,
) -> str:
    """Command for aspirating with the EvoWARE aspirate command. As many wells in one column may be selected as your liquid handling arm has pipettes.
    This method generates the full command (as can be observed when opening a .esc file with an editor) and calls upon other functions to create the code string
    specifying the target wells.

    Parameters
    ----------
    tips : list
        Tip(s) that will be selected; use either a list with integers from 1 - 8 or with tip.T1 - tip.T8
    waste_location : tuple
        Tuple with grid position (1-67) and site number (1-128) of waste as integers
    cleaner_location : tuple
        Tuple with grid position (1-67) and site number (1-128) of cleaner as integers
    arm : int
        number of the LiHa performing the action: 0 = LiHa 1, 1 = LiHa 2
    waste_vol: float
        Volume in waste in mL (0-100)
    waste_delay : int
        Delay before closing valves in waste in ms (0-1000)
    cleaner_vol: float
        Volume in cleaner in mL (0-100)
    cleaner_delay : int
        Delay before closing valves in cleaner in ms (0-1000)
    airgap : int
        Volume of airgap in µL which is aspirated after washing the tips (system trailing airgap) (0-100)
    airgap_speed : int
        Speed of airgap aspiration in µL/s (1-1000)
    retract_speed : int
        Retract speed in mm/s (1-100)
    fastwash : int
        Use fast-wash module = 1, don't use it = 0
    low_volume : int
        Use pinch valves = 1, don't use them = 0
    """
    # perform consistency checks
    (
        tips,
        waste_location,
        cleaner_location,
        arm,
        waste_vol,
        waste_delay,
        cleaner_vol,
        cleaner_delay,
        airgap,
        airgap_speed,
        retract_speed,
        fastwash,
        low_volume,
    ) = prepare_evo_wash_parameters(
        tips=tips,
        waste_location=waste_location,
        cleaner_location=cleaner_location,
        arm=arm,
        waste_vol=waste_vol,
        waste_delay=waste_delay,
        cleaner_vol=cleaner_vol,
        cleaner_delay=cleaner_delay,
        airgap=airgap,
        airgap_speed=airgap_speed,
        retract_speed=retract_speed,
        fastwash=fastwash,
        low_volume=low_volume,
    )
    # calculate tip_selection based on tips argument
    tip_selection = 0
    for tip in tips:
        tip_selection += tip.value
    return f'B;Wash({tip_selection},{waste_location[0]},{waste_location[1]},{cleaner_location[0]},{cleaner_location[1]},"{waste_vol}",{waste_delay},"{cleaner_vol}",{cleaner_delay},{airgap},{airgap_speed},{retract_speed},{fastwash},{low_volume},1000,{arm});'
