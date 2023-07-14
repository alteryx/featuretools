import numpy as np
import pandas as pd

from featuretools.primitives import NumberOfCommonWords
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestNumberOfCommonWords(PrimitiveTestBase):
    primitive = NumberOfCommonWords
    test_word_bank = {"and", "a", "is"}

    def test_delimiter_override(self):
        x = pd.Series(
            [
                "This is a test file.",
                "This,is,second,line, and?",
                "and;subsequent;lines...",
            ],
        )

        expected = pd.Series([2, 2, 1])
        actual = self.primitive(
            word_set=self.test_word_bank,
            delimiters_regex="[ ,;]",
        ).get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_multiline(self):
        x = pd.Series(
            [
                "This is a test file.",
                "This is second line\nthird line $1000;\nand subsequent lines",
            ],
        )

        expected = pd.Series([2, 2])
        actual = self.primitive(self.test_word_bank).get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_null(self):
        x = pd.Series([np.nan, pd.NA, None, "This is a test file."])

        actual = self.primitive(self.test_word_bank).get_function()(x)
        expected = pd.Series([pd.NA, pd.NA, pd.NA, 2])
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_case_insensitive(self):
        x = pd.Series(["Is", "a", "AND"])

        actual = self.primitive(self.test_word_bank).get_function()(x)
        expected = pd.Series([1, 1, 1])
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)
