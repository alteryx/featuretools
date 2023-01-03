import numpy as np
import pandas as pd

from featuretools.primitives import NaturalLanguageToYear


class TestNaturalLanguageToYear:
    primitive = NaturalLanguageToYear

    def test_regular(self):
        primitive_func = NaturalLanguageToYear().get_function()
        array = pd.Series(
            [
                "The year was 1887.",
                "This string has no year",
                "Toy Story (1995)",
                "867-5309",
                "+1 (317) 555-2020",
                "12451997",
                "1997abc3",
                "06-21-2018",
            ],
            dtype="string",
        )
        answer = primitive_func(array)
        correct_answer = pd.Series(
            [
                "1887",
                pd.NA,
                "1995",
                pd.NA,
                "2020",
                pd.NA,
                pd.NA,
                "2018",
            ],
            dtype="string",
        )
        pd.testing.assert_series_equal(answer, correct_answer, check_names=False)

    def test_multiple(self):
        primitive_func = NaturalLanguageToYear().get_function()
        array = pd.Series(
            [
                "1887 and 1888.",
                "This string has no year",
                "Dates: 1995-2001",
                pd.NA,
            ],
            dtype="string",
        )
        answer = primitive_func(array)
        correct_answer = pd.Series(["1887", pd.NA, "1995", pd.NA], dtype="string")
        pd.testing.assert_series_equal(answer, correct_answer, check_names=False)

    def test_nan(self):
        primitive_func = NaturalLanguageToYear().get_function()
        array = pd.Series(
            [
                "The year was 1887.",
                "This string has no year",
                "Toy Story (1995)",
                pd.NA,
            ],
            dtype="string",
        )
        answer = primitive_func(array)
        correct_answer = pd.Series(["1887", pd.NA, "1995", pd.NA], dtype="string")
        pd.testing.assert_series_equal(answer, correct_answer, check_names=False)

    def test_empty(self):
        primitive_func = NaturalLanguageToYear().get_function()
        array = pd.Series(
            [
                "The year was 1887.",
                "This string has no year",
                "Toy Story (1995)",
                "",
            ],
            dtype="string",
        )
        answer = primitive_func(array)
        correct_answer = pd.Series(["1887", pd.NA, "1995", pd.NA], dtype="string")
        pd.testing.assert_series_equal(answer, correct_answer, check_names=False)
