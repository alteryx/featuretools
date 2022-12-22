from string import punctuation
from typing import Iterable

import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable, NaturalLanguage

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.standard.transform.natural_language.constants import (
    DELIMITERS,
)


class NumberOfUniqueWords(TransformPrimitive):
    """Determines the number of unique words in a string.

    Description:
        Determines the number of unique words in a given string. Includes options for
        case-insensitive behavior.

    Args:
        case_insensitive (bool, optional): Specify case_insensitivity when searching for unique words.
        For example, setting this to True would mean "WORD word" would be treated as having
        one unique word. Defaults to False.

    Examples:
        >>> x = ['Word word Word', 'This is a SENTENCE.', 'green red green']
        >>> number_of_unique_words = NumberOfUniqueWords()
        >>> number_of_unique_words(x).tolist()
        [2, 4, 2]

        >>> x = ['word WoRD WORD worD', 'dog dog dog', 'catt CAT caT']
        >>> number_of_unique_words = NumberOfUniqueWords(case_insensitive=True)
        >>> number_of_unique_words(x).tolist()
        [1, 1, 2]
    """

    name = "number_of_unique_words"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})

    default_value = 0

    def __init__(self, case_insensitive=False):
        self.case_insensitive = case_insensitive

    def get_function(self):
        def _unique_word_helper(text):
            if not isinstance(text, Iterable):
                return pd.NA
            unique = set()
            for t in text:
                punct_less = t.strip(punctuation)
                if len(punct_less) > 0:
                    unique.add(punct_less)
            return len(unique)

        def num_unique_words(array):
            if self.case_insensitive:
                array = array.str.lower()
            array = array.str.split(f"{DELIMITERS}")
            return array.apply(_unique_word_helper)

        return num_unique_words
