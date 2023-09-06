import re

import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable, NaturalLanguage

from featuretools.primitives.base import TransformPrimitive


class CountString(TransformPrimitive):
    """Determines how many times a given string shows up in a text field.

    Args:
        string (str): The string to determine the count of. Defaults to
            the word "the".
        ignore_case (bool): Determines if case of the string should be
            considered or not. Defaults to true.
        ignore_non_alphanumeric (bool): Determines if non-alphanumeric
            characters should be used in the search. Defaults to False.
        is_regex (bool): Defines if the string argument is a regex or not.
            Defaults to False.
        match_whole_words_only (bool): Determines if whole words should be
            matched or not. For example searching for word `the` against
            `then, the, there` should only return `the` if this argument
            was True. Defaults to False.
    Examples:
        >>> count_string = CountString(string="the")
        >>> count_string(["The problem was difficult.",
        ...               "He was there.",
        ...               "The girl went to the store."]).tolist()
        [1.0, 1.0, 2.0]
        >>> # Match case of string
        >>> count_string_ignore_case = CountString(string="the", ignore_case=False)
        >>> count_string_ignore_case(["The problem was difficult.",
        ...                           "He was there.",
        ...                           "The girl went to the store."]).tolist()
        [0.0, 1.0, 1.0]
        >>> # Ignore non-alphanumeric characters in the search
        >>> count_string_ignore_non_alphanumeric = CountString(string="the",
        ...                                                    ignore_non_alphanumeric=True)
        >>> count_string_ignore_non_alphanumeric(["Th*/e problem was difficult.",
        ...                                       "He was there.",
        ...                                       "The girl went to the store."]).tolist()
        [1.0, 1.0, 2.0]
        >>> # Specify the string as a regex
        >>> count_string_is_regex = CountString(string="t.e", is_regex=True)
        >>> count_string_is_regex(["The problem was difficult.",
        ...                        "He was there.",
        ...                        "The girl went to the store."]).tolist()
        [1.0, 1.0, 2.0]
        >>> # Match whole words only
        >>> count_string_match_whole_words_only = CountString(string="the",
        ...                                                   match_whole_words_only=True)
        >>> count_string_match_whole_words_only(["The problem was difficult.",
        ...                                      "He was there.",
        ...                                      "The girl went to the store."]).tolist()
        [1.0, 0.0, 2.0]
    """

    name = "count_string"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})

    def __init__(
        self,
        string="the",
        ignore_case=True,
        ignore_non_alphanumeric=False,
        is_regex=False,
        match_whole_words_only=False,
    ):
        self.string = string
        self.ignore_case = ignore_case
        self.ignore_non_alphanumeric = ignore_non_alphanumeric
        self.match_whole_words_only = match_whole_words_only
        self.is_regex = is_regex

        # we don't want to strip non alphanumeric characters from the pattern
        # ie h.ll. should match "hello" so we can't strip the dots to make hll
        if not is_regex:
            self.pattern = re.escape(self.process_text(string))
        else:
            self.pattern = string
            if ignore_case:
                self.pattern = self.pattern.lower()

        # \b\b.*\b\b is the same as \b.*\b so we don't have to check if
        # the pattern is given to us as regex and if it already has leading
        # and trailing \b's
        if match_whole_words_only:
            self.pattern = "\\b" + self.pattern + "\\b"

    def process_text(self, text):
        if self.ignore_non_alphanumeric:
            text = re.sub("[^0-9a-zA-Z ]+", "", text)
        if self.ignore_case:
            text = text.lower()
        return text

    def get_function(self):
        def count_string(words):
            if not isinstance(words, str):
                return np.nan
            words = self.process_text(words)
            return len(re.findall(self.pattern, words))

        return np.vectorize(count_string, otypes=[float])
