import numpy as np
import pandas as pd
import pytest

from featuretools.primitives import NumberOfWordsInQuotes
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestNumberOfWordsInQuotes(PrimitiveTestBase):
    primitive = NumberOfWordsInQuotes

    def test_regular_double_quotes_input(self):
        x = pd.Series(
            [
                'Yes "    "',
                '"Hello this is a test"',
                '"Yes" "   "',
                "",
                '"Python, java prolog"',
                '"Python, java prolog" three words here "binary search algorithm"',
                '"Diffie-Hellman key exchange"',
                '"user@email.com"',
                '"https://alteryx.com"',
                '"100,000"',
                '"This Borderlands game here"" is the perfect conclusion to the ""Borderlands 3"" line, which focuses on the fans ""favorite character and gives the players the opportunity to close for a long time some very important questions about\'s character and the memorable scenery with which the players interact.',
            ],
        )
        expected = pd.Series([0, 5, 1, 0, 3, 6, 3, 1, 1, 1, 6], dtype="Int64")
        actual = self.primitive("double").get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_captures_regular_single_quotes(self):
        x = pd.Series(
            [
                "'Hello this is a test'",
                "'Python, Java Prolog'",
                "'Python, Java Prolog' three words here 'three words here'",
                "'Diffie-Hellman key exchange'",
                "'user@email.com'",
                "'https://alteryx.com'",
                "'there's where's here's' word 'word'",
                "'100,000'",
            ],
        )
        expected = pd.Series([5, 3, 6, 3, 1, 1, 4, 1], dtype="Int64")
        actual = self.primitive("single").get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_captures_both_single_and_double_quotes(self):
        x = pd.Series(
            [
                "'test test test test' three words here \"test test test!\"",
            ],
        )
        expected = pd.Series([7], dtype="Int64")
        actual = self.primitive().get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_unicode_input(self):
        x = pd.Series(
            [
                '"Ángel"',
                '"Ángel" word word',
            ],
        )
        expected = pd.Series([1, 1], dtype="Int64")
        actual = self.primitive().get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_multiline(self):
        x = pd.Series(
            [
                "'Yes\n, this is me'",
            ],
        )
        expected = pd.Series([4], dtype="Int64")
        actual = self.primitive().get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_raises_error_invalid_args(self):
        error_msg = (
            "NULL is not a valid quote_type. Specify 'both', 'single', or 'double'"
        )
        with pytest.raises(
            ValueError,
            match=error_msg,
        ):
            self.primitive(quote_type="NULL")

    def test_null(self):
        x = pd.Series([np.nan, pd.NA, None, '"test"'])
        actual = self.primitive().get_function()(x)
        expected = pd.Series([pd.NA, pd.NA, pd.NA, 1.0], dtype="Int64")
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)
