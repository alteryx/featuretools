# -*- coding: utf-8 -*-
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable, NaturalLanguage

from featuretools.primitives.standard.transform.natural_language.count_string import (
    CountString,
)


class UpperCaseCount(CountString):
    """Calculates the number of upper case letters in text.

    Description:
        Given a list of strings, determine the number of characters in each string
        that are capitalized. Counts every letter individually, not just every
        word that contains capitalized letters.

        If a string is missing, return `NaN`

    Examples:
        >>> x = ['This IS a string.', 'This is a string', 'aaa']
        >>> upper_case_count = UpperCaseCount()
        >>> upper_case_count(x).tolist()
        [3.0, 1.0, 0.0]
    """

    name = "upper_case_count"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    default_value = 0

    def __init__(self):
        pattern = r"([A-Z])"
        super().__init__(string=pattern, is_regex=True, ignore_case=False)
