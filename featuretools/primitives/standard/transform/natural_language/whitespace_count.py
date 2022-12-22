from featuretools.primitives.standard.transform.natural_language.count_string import (
    CountString,
)


class WhitespaceCount(CountString):
    """Calculates number of whitespaces in a string.

    Description:
        Given a list of strings, determine the whitespaces in each string
        If a string is missing, return `NaN`

    Examples:
        >>> x = ['', 'hi im ethan', 'multiple    spaces']
        >>> upper_case_count = WhitespaceCount()
        >>> upper_case_count(x).tolist()
        [0.0, 2.0, 4.0]
    """

    name = "whitespace_count"
    default_value = 0

    def __init__(self):
        super().__init__(string=" ")
