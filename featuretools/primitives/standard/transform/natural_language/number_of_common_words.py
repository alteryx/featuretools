from string import punctuation
from typing import Iterable

import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable, NaturalLanguage

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.standard.transform.natural_language.constants import (
    DELIMITERS,
    common_words_1000,
)


class NumberOfCommonWords(TransformPrimitive):
    """Determines the number of common words in a string.

    Description:
        Given string, determine the number of words that appear in a supplied word set.
        The word set defaults to nlp_primitives.constants.common_words_1000. The string
        is case insensitive. The word bank should consist of only lower case strings. If a string is
        missing, return `NaN`.

    Args:
        word_set (set, optional): The set of words to look for in the string. These
            words should all be lower case strings.
        delimiters_regex (str, optional): The regular expression used to determine
            what separates words. Defaults to whitespace characters.

    Examples:
        >>> x = ['Hey! This is some natural language', 'bacon, cheesburger, AND, fries', 'I! Am. A; duck?']
        >>> number_of_common_words = NumberOfCommonWords(word_set={'and', 'some', 'am', 'a', 'the', 'is', 'i'})
        >>> number_of_common_words(x).tolist()
        [2, 1, 3]

        >>> x = ['Hey! This is. some. natural language']
        >>> number_of_common_words = NumberOfCommonWords(word_set={'hey', 'is', 'some'}, delimiters_regex="[ .]")
        >>> number_of_common_words(x).tolist()
        [3]
    """

    name = "number_of_common_words"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})

    default_value = 0

    def __init__(
        self,
        word_set=common_words_1000,
        delimiters_regex=DELIMITERS,
    ):
        self.delimiters_regex = delimiters_regex
        self.word_set = word_set

    def get_function(self):
        def get_num_in_word_bank(words):
            if not isinstance(words, Iterable):
                return pd.NA
            num_common_words = 0
            for w in words:
                if (
                    w.lower().strip(punctuation) in self.word_set
                ):  # assumes word_set is all lowercase
                    num_common_words += 1
            return num_common_words

        def num_common_words(x):
            words = x.str.split(self.delimiters_regex)
            return words.apply(get_num_in_word_bank)

        return num_common_words
