import numpy as np
import pandas as pd

from featuretools.primitives import UpperCaseWordCount


class TestUpperCaseWordCount:
    primitive = UpperCaseWordCount

    def test_strings(self):
        x = pd.Series(
            [
                "This IS a STRING.",
                "Testing AAA",
                "Testing AAA-BBB",
                "Testing AA3",
            ],
            dtype="string",
        )
        primitive_func = self.primitive().get_function()
        answers = pd.Series([2.0, 1.0, 2.0, 1.0], dtype="float64")
        pd.testing.assert_series_equal(primitive_func(x), answers, check_names=False)

    def test_nan(self):
        x = pd.Series(
            [
                np.nan,
                "",
                "This IS a STRING.",
            ],
            dtype="string",
        )
        primitive_func = self.primitive().get_function()
        answers = pd.Series([np.nan, 0.0, 2.0], dtype="float64")
        pd.testing.assert_series_equal(primitive_func(x), answers, check_names=False)
