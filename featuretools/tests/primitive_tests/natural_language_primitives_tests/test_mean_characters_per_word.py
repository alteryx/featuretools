import numpy as np
import pandas as pd
import pytest

from featuretools.primitives import MeanCharactersPerWord
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestMeanCharactersPerWord(PrimitiveTestBase):
    primitive = MeanCharactersPerWord

    def test_sentences(self):
        x = pd.Series(
            [
                "This is a test file",
                "This is second line",
                "third line $1,000",
                "and subsequent lines",
                "and more",
            ],
        )
        primitive_func = self.primitive().get_function()
        answers = pd.Series([3.0, 4.0, 5.0, 6.0, 3.5])
        pd.testing.assert_series_equal(primitive_func(x), answers, check_names=False)

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
        answers = pd.Series([3.0, 4.0, 8.0, 10.5, 4.0])
        pd.testing.assert_series_equal(primitive_func(x), answers, check_names=False)

    def test_multiline(self):
        x = pd.Series(
            [
                "This is a test file",
                "This is second line\nthird line $1000;\nand subsequent lines",
                "and more",
            ],
        )
        primitive_func = self.primitive().get_function()
        answers = pd.Series([3.0, 4.8, 3.5])
        pd.testing.assert_series_equal(primitive_func(x), answers, check_names=False)

    @pytest.mark.parametrize(
        "na_value",
        [None, np.nan, pd.NA],
    )
    def test_nans(self, na_value):
        x = pd.Series([na_value, "", "third line"])
        primitive_func = self.primitive().get_function()
        answers = pd.Series([np.nan, 0, 4.5])
        pd.testing.assert_series_equal(primitive_func(x), answers, check_names=False)

    @pytest.mark.parametrize(
        "na_value",
        [None, np.nan, pd.NA],
    )
    def test_all_nans(self, na_value):
        x = pd.Series([na_value, na_value, na_value])
        primitive_func = self.primitive().get_function()
        answers = pd.Series([np.nan, np.nan, np.nan])
        pd.testing.assert_series_equal(primitive_func(x), answers, check_names=False)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)
