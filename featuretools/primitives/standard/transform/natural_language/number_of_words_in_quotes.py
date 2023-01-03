import re
from string import punctuation

import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable, NaturalLanguage

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.standard.transform.natural_language.constants import (
    DELIMITERS,
)


class NumberOfWordsInQuotes(TransformPrimitive):
    """Determines the number of words in quotes in a string.

    Description:
        Given a list of strings, determine the number of words in quotes
        in each string.

        This implementation handles Unicode characters.

        If a string is missing, return `NaN`.

    Args:
        quote_type (str, optional): Specifies what type of quotation marks to match.
        Argument "single" matches on only single quotes (' ').
        Argument "double" matches words between double quotes (" ").
        Argument "both" matches words between either type of quotes.
        Defaults to "both".

    Examples:
         >>> x = ['"python" java prolog "Diffie-Hellman" "4.99"', "Reach me at 'user@email.com'", "'Here's an interesting example!'"]
        >>> number_of_words_in_quotes = NumberOfWordsInQuotes()
        >>> number_of_words_in_quotes(x).tolist()
        [3, 1, 4]
    """

    name = "number_of_words_in_quotes"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    default_value = 0

    def __init__(self, quote_type="both"):
        if quote_type not in ["both", "single", "double"]:
            raise ValueError(
                f"{quote_type} is not a valid quote_type. Specify 'both', 'single', or 'double'",
            )
        self.quote_type = quote_type
        IN_DOUBLE_QUOTES = r'((^|\W)"(.)*?"(?!\w))'
        IN_SINGLE_QUOTES = r"((^|\W)'(.)*?'(?!\w))"
        if quote_type == "double":
            self.regex = IN_DOUBLE_QUOTES
        elif quote_type == "single":
            self.regex = IN_SINGLE_QUOTES
        else:
            self.regex = f"({IN_SINGLE_QUOTES}|{IN_DOUBLE_QUOTES})"

    def get_function(self):
        def count_words_in_quotes(text):
            if pd.isnull(text):
                return pd.NA
            matches = re.findall(self.regex, text, re.DOTALL)
            count = 0
            for match in matches:
                matched_phrase = match[0]
                words = re.split(f"{DELIMITERS}", matched_phrase)
                for word in words:
                    if len(word.strip(punctuation + " ")):
                        count += 1
            return count

        def num_words_in_quotes(array):
            return array.apply(count_words_in_quotes).astype("Int64")

        return num_words_in_quotes
