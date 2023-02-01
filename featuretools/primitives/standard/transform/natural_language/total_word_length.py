from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable, NaturalLanguage

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.standard.transform.natural_language.constants import (
    PUNCTUATION_AND_WHITESPACE,
)


class TotalWordLength(TransformPrimitive):
    """Determines the total word length.

    Description:
        Given list of strings, determine the total
        word length in each string. A word is defined as
        a series of any characters not separated by a delimiter.
        If a string is empty or `NaN`, return `NaN`.

    Args:
        delimiters_regex (str): Delimiters as a regex string for splitting text into words.
            Defaults to whitespace characters.

    Examples:
        >>> x = ['This is a test file', 'This is second line', 'third line $1,000', None]
        >>> total_word_length = TotalWordLength()
        >>> total_word_length(x).tolist()
        [15.0, 16.0, 13.0, nan]
    """

    name = "total_word_length"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})

    default_value = 0

    def __init__(self, do_not_count=PUNCTUATION_AND_WHITESPACE):
        self.do_not_count = do_not_count

    def get_function(self):
        def total_word_length(x):
            return x.str.len() - x.str.count(self.do_not_count)

        return total_word_length
