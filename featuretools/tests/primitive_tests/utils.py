def get_number_from_offset(offset):
    """Extract the numeric element of a potential offset string.

    Args:
        offset (int, str): If offset is an integer, that value is returned. If offset is a string,
            it's assumed to be an offset string of the format nD where n is a single digit integer.

    Note: This helper utility should only be used with offset strings that only have one numeric character.
        Only the first character will be returned, so if an offset string 24H is used, it will incorrectly
        return the integer 2. Additionally, any of the offset timespans (H for hourly, D for daily, etc.)
        can be used here; however, care should be taken by the user to remember what that timespan is when
        writing tests, as comparing 7 from 7D to 1 from 1W may not behave as expected.
    """
    if isinstance(offset, str):
        return int(offset[0])
    else:
        return offset
