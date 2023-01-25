import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double, NaturalLanguage

from featuretools.primitives.base import TransformPrimitive


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
        [1.0, 0.0, 1.0]
    """

    name = "upper_case_word_count"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    default_value = 0

    def get_function(self):
        pattern = r"(\w[A-Z0-9]+\b)"

        def upper_case_word_count(x):
            x = x.reset_index(drop=True)
            counts = x.str.extractall(pattern).groupby(level=0).count()[0]
            counts = counts.reindex_like(x).fillna(0)
            counts[x.isnull()] = np.nan
            return counts.astype("float64")

        return upper_case_word_count
