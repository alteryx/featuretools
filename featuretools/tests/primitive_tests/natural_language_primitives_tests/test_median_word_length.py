import numpy as np
import pandas as pd

from featuretools.primitives import MedianWordLength
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestMedianWordLength(PrimitiveTestBase):
    primitive = MedianWordLength

    def test_delimiter_override(self):
        x = pd.Series(
            ["This is a test file.", "This,is,second,line?", "and;subsequent;lines..."],
        )

        expected = pd.Series([4.0, 4.5, 8.0])
        actual = self.primitive("[ ,;]").get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_multiline(self):
        x = pd.Series(
            [
                "This is a test file.",
                "This is second line\nthird line $1000;\nand subsequent lines",
            ],
        )

        expected = pd.Series([4.0, 4.5])
        actual = self.primitive().get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_null(self):
        x = pd.Series([np.nan, pd.NA, None, "This is a test file."])

        actual = self.primitive().get_function()(x)
        expected = pd.Series([np.nan, np.nan, np.nan, 4.0])
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)
