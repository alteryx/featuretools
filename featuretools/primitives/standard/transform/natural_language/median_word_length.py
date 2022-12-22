from numpy import median
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double, NaturalLanguage

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.standard.transform.natural_language.constants import (
    DELIMITERS,
)


class MedianWordLength(TransformPrimitive):
    """Determines the median word length.

    Description:
        Given list of strings, determine the median
        word length in each string. A word is defined as
        a series of any characters not separated by a delimiter.
        If a string is empty or `NaN`, return `NaN`.

    Args:
        delimiters_regex (str): Delimiters as a regex string for splitting text into words.
            Defaults to whitespace characters.

    Examples:
        >>> x = ['This is a test file', 'This is second line', 'third line $1,000', None]
        >>> median_word_length = MedianWordLength()
        >>> median_word_length(x).tolist()
        [4.0, 4.0, 5.0, nan]
    """

    name = "median_word_length"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    default_value = 0

    def __init__(self, delimiters_regex=DELIMITERS):
        self.delimiters_regex = delimiters_regex

    def get_function(self):
        def get_median(words):
            if isinstance(words, list):
                return median([len(word) for word in words if len(word) != 0])

        def median_word_length(x):
            words = x.str.split(self.delimiters_regex)
            return words.apply(get_median)

        return median_word_length
