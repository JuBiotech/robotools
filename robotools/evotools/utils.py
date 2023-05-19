"""Generic utility functions."""


def to_hex(dec: int):
    """Method from stackoverflow to convert decimal to hex.
    Link: https://stackoverflow.com/questions/5796238/python-convert-decimal-to-hex
    Solution posted by user "Chunghee Kim" on 21.11.2020.
    """
    digits = "0123456789ABCDEF"
    x = dec % 16
    rest = dec // 16
    if rest == 0:
        return digits[x]
    return to_hex(rest) + digits[x]
