from datetime import datetime

import numpy as np
import pandas as pd

from featuretools.primitives import (
    CumulativeTimeSinceLastFalse,
)


class TestCumulativeTimeSinceLastFalse:
    primitive = CumulativeTimeSinceLastFalse
    booleans = pd.Series([True, False, True, False, True, True])
    datetimes = pd.Series(
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(len(booleans))],
    )
    answer = pd.Series([np.nan, 0, 6, 0, 6, 12])

    def test_regular(self):
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(self.datetimes, self.booleans)
        assert given_answer.equals(self.answer)

    def test_all_true(self):
        primitive_func = self.primitive().get_function()
        booleans = pd.Series([True, True, True])
        datetimes = pd.Series(
            [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(len(booleans))],
        )
        given_answer = primitive_func(datetimes, booleans)
        answer = pd.Series([np.nan] * 3)
        assert given_answer.equals(answer)

    def test_all_nan(self):
        primitive_func = self.primitive().get_function()
        datetimes = pd.Series([np.nan] * 4)
        booleans = pd.Series([np.nan] * 4)
        given_answer = primitive_func(datetimes, booleans)
        answer = pd.Series([np.nan] * 4)
        assert given_answer.equals(answer)

    def test_some_nans(self):
        primitive_func = self.primitive().get_function()
        booleans = pd.Series(
            [
                True,
                False,
                True,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
            ],
        )
        datetimes = pd.Series([np.nan] * 2)
        datetimes = pd.concat([datetimes, self.datetimes])
        datetimes = pd.concat([datetimes, pd.Series([np.nan] * 2)])
        datetimes = datetimes.reset_index(drop=True)
        answer = pd.Series(
            [
                np.nan,
                np.nan,
                np.nan,
                0,
                6,
                12,
                0,
                0,
                np.nan,
                np.nan,
            ],
        )
        given_answer = primitive_func(datetimes, booleans)
        assert given_answer.equals(answer)
