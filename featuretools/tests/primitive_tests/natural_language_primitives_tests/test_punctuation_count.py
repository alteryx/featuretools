import numpy as np
import pandas as pd

from featuretools.primitives import PunctuationCount
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestPunctuationCount(PrimitiveTestBase):
    primitive = PunctuationCount

    def test_punctuation(self):
        x = pd.Series(
            [
                "This is a test file.",
                "This, is second line?",
                "third/line $1,000;",
                "and--subsequen't lines...",
                "*and, more..",
            ],
        )
        primitive_func = self.primitive().get_function()
        answers = [1.0, 2.0, 4.0, 6.0, 4.0]
        np.testing.assert_array_equal(primitive_func(x), answers)

    def test_multiline(self):
        x = pd.Series(
            [
                "This is a test file.",
                "This is second line\nthird line $1000;\nand subsequent lines",
            ],
        )
        primitive_func = self.primitive().get_function()
        answers = [1.0, 2.0]
        np.testing.assert_array_equal(primitive_func(x), answers)

    def test_nan(self):
        x = pd.Series([np.nan, "", "This is a test file."])
        primitive_func = self.primitive().get_function()
        answers = [np.nan, 0.0, 1.0]
        np.testing.assert_array_equal(primitive_func(x), answers)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)
