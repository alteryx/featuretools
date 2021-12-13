def get_number_of_days(offset):
    if isinstance(offset, str):
        return int(offset[0])
    else:
        return offset
