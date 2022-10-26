# -*- coding: utf-8 -*-

import re
import string

from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable, NaturalLanguage

from featuretools.primitives.standard.transform.natural_language.count_string import (
    CountString,
)


class PunctuationCount(CountString):
    """Determines number of punctuation characters in a string.

    Description:
        Given list of strings, determine the number of punctuation
        characters in each string. Looks for any of the following:

        !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~

        If a string is missing, return `NaN`.

    Examples:
        >>> x = ['This is a test file.', 'This is second line', 'third line: $1,000']
        >>> punctuation_count = PunctuationCount()
        >>> punctuation_count(x).tolist()
        [1.0, 0.0, 3.0]
    """

    name = "punctuation_count"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    default_value = 0

    def __init__(self):
        pattern = "(%s)" % "|".join([re.escape(x) for x in string.punctuation])
        super().__init__(string=pattern, is_regex=True, ignore_case=False)
