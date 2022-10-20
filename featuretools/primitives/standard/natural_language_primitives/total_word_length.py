from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double, NaturalLanguage

from featuretools.primitives.base import TransformPrimitive


class TotalWordLength(TransformPrimitive):
    """Determines the total word length.

    Description:
        Given list of strings, determine the total
        word length in each string. A word is defined as
        a series of any characters not separated by a delimiter.
        If a string is empty or `NaN`, return `NaN`.

    Args:
        delimiters_regex (str): Delimiters as a regex string for splitting text into words.
            The default delimiters include "- [].,!?;\\n".

    Examples:
        >>> x = ['This is a test file', 'This is second line', 'third line $1,000', None]
        >>> total_word_length = TotalWordLength()
        >>> total_word_length(x).tolist()
        [15.0, 16.0, 14.0, nan]
    """

    name = "total_word_length"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    default_value = 0

    def __init__(self, delimiters_regex=r"[- \[\].,!\?;\n]"):
        self.delimiters_regex = delimiters_regex

    def get_function(self):
        def total_word_length(x):
            delimiters = x.str.count(self.delimiters_regex)
            return x.str.len() - delimiters

        return total_word_length
