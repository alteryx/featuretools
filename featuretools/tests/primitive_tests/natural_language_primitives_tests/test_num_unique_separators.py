import numpy as np
import pandas as pd

from featuretools.primitives import NumUniqueSeparators
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestNumUniqueSeparators(PrimitiveTestBase):
    primitive = NumUniqueSeparators

    def test_punctuation(self):
        x = pd.Series(
            [
                "This: is a test file",
                "This, is second line?",
                "third/line $1,000;",
                "and--subsequen't lines...",
                "*and, more..",
            ],
        )
        primitive_func = self.primitive().get_function()
        answers = pd.Series([1, 3, 3, 2, 3])
        pd.testing.assert_series_equal(primitive_func(x), answers, check_names=False)

    def test_other_delimeters(self):
        x = pd.Series(["@#$%^&*()<>/[]\\`~-_=+"])
        primitive_func = self.primitive().get_function()
        answers = pd.Series([0])
        pd.testing.assert_series_equal(primitive_func(x), answers, check_names=False)

    def test_multiline(self):
        x = pd.Series(
            [
                "This is a test file",
                "This is second line\nthird line $1000;\nand subsequent lines",
                "and more!",
            ],
        )
        primitive_func = self.primitive().get_function()
        answers = pd.Series([1, 3, 2])
        pd.testing.assert_series_equal(primitive_func(x), answers, check_names=False)

    def test_nans(self):
        x = pd.Series([np.nan, "", "third line."])
        primitive_func = self.primitive().get_function()
        answers = pd.Series([pd.NA, 0, 2])
        pd.testing.assert_series_equal(primitive_func(x), answers, check_names=False)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)
