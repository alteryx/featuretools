def convert_to_nth(integer):
    string_nth = str(integer)
    end_int = integer % 10
    if end_int == 1 and integer % 100 != 11:
        return str(integer) + 'st'
    elif end_int == 2:
        return str(string_nth) + 'nd'
    elif end_int == 3:
        return str(string_nth) + 'rd'
    else:
        return str(string_nth) + 'th'
