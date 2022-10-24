# -*- coding: utf-8 -*-

import re

import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double, NaturalLanguage

from featuretools.primitives.base import TransformPrimitive

PUNCTUATION = re.escape("!,.:;?")
END_OF_SENTENCE_PUNCT_RE = re.compile(
    rf"[{PUNCTUATION}]+$|[{PUNCTUATION}]+ |[{PUNCTUATION}]+\n",
)


def _mean_characters_per_word(value):
    if pd.isna(value):
        return np.nan

    # replace end-of-sentence punctuation with space
    value = END_OF_SENTENCE_PUNCT_RE.sub(" ", value)
    words = value.split()
    character_count = [len(x) for x in words]

    return np.mean(character_count) if len(character_count) else 0


class MeanCharactersPerWord(TransformPrimitive):
    """Determines the mean number of characters per word.

    Description:
        Given list of strings, determine the mean number of
        characters per word in each string. A word is defined as
        a series of any characters not separated by white space.
        Punctuation is removed before counting. If a string
        is empty or `NaN`, return `NaN`.

    Examples:
        >>> x = ['This is a test file', 'This is second line', 'third line $1,000']
        >>> mean_characters_per_word = MeanCharactersPerWord()
        >>> mean_characters_per_word(x).tolist()
        [3.0, 4.0, 5.0]
    """

    name = "mean_characters_per_word"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    default_value = 0

    def get_function(self):
        def mean_characters_per_word(series):
            return series.apply(_mean_characters_per_word)

        return mean_characters_per_word
