# -*- coding: utf-8 -*-
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable, NaturalLanguage

from featuretools.primitives.standard.transform.natural_language.count_string import (
    CountString,
)


class TitleWordCount(CountString):
    """Determines the number of title words in a string.

    Description:
        Given list of strings, determine the number of title words
        in each string. A title word is defined as any word starting
        with a capital letter. Words at the start of a sentence will
        be counted.

        If a string is missing, return `NaN`.

    Examples:
        >>> x = ['My favorite movie is Jaws.', 'this is a string', 'AAA']
        >>> title_word_count = TitleWordCount()
        >>> title_word_count(x).tolist()
        [2.0, 0.0, 1.0]
    """

    name = "title_word_count"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    default_value = 0

    def __init__(self):
        pattern = r"([A-Z][^\s]*)"
        super().__init__(string=pattern, is_regex=True, ignore_case=False)
