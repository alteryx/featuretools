from datetime import datetime

import pandas as pd

from featuretools.primitives import Season


class TestSeason:
    def test_regular(self):
        primitive_instance = Season()
        primitive_func = primitive_instance.get_function()
        case = pd.date_range(start="2019-01", periods=12, freq="m").to_series()
        answer = pd.Series(
            [
                "winter",
                "winter",
                "spring",
                "spring",
                "spring",
                "summer",
                "summer",
                "summer",
                "fall",
                "fall",
                "fall",
                "winter",
            ],
            dtype="string",
        )
        given_answer = primitive_func(case)
        pd.testing.assert_series_equal(
            given_answer.reset_index(drop=True),
            answer.reset_index(drop=True),
        )

    def test_nat(self):
        primitive_instance = Season()
        primitive_func = primitive_instance.get_function()
        case = pd.Series(
            [
                "NaT",
                "2019-02",
                "2019-03",
                "NaT",
            ],
        ).astype("datetime64[ns]")
        answer = pd.Series([pd.NA, "winter", "winter", pd.NA], dtype="string")
        given_answer = pd.Series(primitive_func(case))
        pd.testing.assert_series_equal(given_answer, answer)

    def test_datetime(self):
        primitive_instance = Season()
        primitive_func = primitive_instance.get_function()
        case = pd.Series(
            [
                datetime(2011, 3, 1),
                datetime(2011, 6, 1),
                datetime(2011, 9, 1),
                datetime(2011, 12, 1),
                # leap year
                datetime(2020, 2, 29),
            ],
        )
        answer = pd.Series(
            ["winter", "spring", "summer", "fall", "winter"],
            dtype="string",
        )
        given_answer = primitive_func(case)
        pd.testing.assert_series_equal(given_answer, answer)
