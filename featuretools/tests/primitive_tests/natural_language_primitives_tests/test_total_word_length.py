import numpy as np
import pandas as pd

from featuretools.primitives import TotalWordLength
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestTotalWordLength(PrimitiveTestBase):
    primitive = TotalWordLength

    def test_delimiter_override(self):
        x = pd.Series(
            ["This is a test file.", "This,is,second,line?", "and;subsequent;lines..."],
        )

        expected = pd.Series([16, 17, 21])
        actual = self.primitive("[ ,;]").get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_multiline(self):
        x = pd.Series(
            [
                "This is a test file.",
                "This is second line\nthird line $1000;\nand subsequent lines",
            ],
        )

        expected = pd.Series([15, 47])
        actual = self.primitive().get_function()(x)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_null(self):
        x = pd.Series([np.nan, pd.NA, None, "This is a test file."])

        expected = pd.Series([np.nan, np.nan, np.nan, 15])
        actual = self.primitive().get_function()(x).astype(float)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)
