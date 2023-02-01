import re
from string import punctuation

import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable, NaturalLanguage

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.standard.transform.natural_language.constants import (
    DELIMITERS,
)


class UpperCaseWordCount(TransformPrimitive):
    """Determines the number of words in a string that are entirely capitalized.

    Description:
        Given list of strings, determine the number of words in each string
        that are entirely capitalized.

        If a string is missing, return `NaN`.

    Examples:
        >>> x = ['This IS a string.', 'This is a string', 'AAA']
        >>> upper_case_word_count = UpperCaseWordCount()
        >>> upper_case_word_count(x).tolist()
        [1, 0, 1]
    """

    name = "upper_case_word_count"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    default_value = 0

    def get_function(self):
        def upper_case_word_count(x):
            def _count_upper_case_words(elem):
                if pd.isna(elem):
                    return pd.NA
                return sum(
                    1
                    for word in re.split(DELIMITERS, elem)
                    if word.strip(punctuation) and word.upper() == word
                )

            return x.apply(_count_upper_case_words)

        return upper_case_word_count
