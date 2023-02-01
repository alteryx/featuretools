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
                "Testing AAA BBB",
                "Testing TEsTIng AA3 AA_33 HELLO",
                "AAA $@()#$@@#$",
            ],
            dtype="string",
        )
        primitive_func = self.primitive().get_function()
        answers = pd.Series([2, 1, 2, 3, 1], dtype="Int64")
        pd.testing.assert_series_equal(
            primitive_func(x).astype("Int64"),
            answers,
            check_names=False,
        )

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
        answers = pd.Series([pd.NA, 0, 2], dtype="Int64")
        pd.testing.assert_series_equal(
            primitive_func(x).astype("Int64"),
            answers,
            check_names=False,
        )
