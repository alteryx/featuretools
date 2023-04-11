from datetime import datetime

import numpy as np
import pandas as pd

from featuretools.primitives import (
    CumulativeTimeSinceLastFalse,
    CumulativeTimeSinceLastTrue,
)
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestCumulativeTimeSinceLastTrue(PrimitiveTestBase):
    primitive = CumulativeTimeSinceLastTrue
    booleans = pd.Series([False, True, False, True, False, False])
    datetimes = pd.Series(
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(len(booleans))],
    )
    answer = pd.Series([np.nan, 0, 6, 0, 6, 12])

    def test_regular(self):
        primitive_func = self.primitive().get_function()
        given_answer = primitive_func(self.datetimes, self.booleans)
        assert given_answer.equals(self.answer)

    def test_all_false(self):
        primitive_func = self.primitive().get_function()
        booleans = pd.Series([False, False, False])
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
                False,
                True,
                False,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
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

    def test_with_featuretools(self, pd_es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(pd_es, aggregation, transform, self.primitive)


class TestCumulativeTimeSinceLastFalse(PrimitiveTestBase):
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

    def test_with_featuretools(self, pd_es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(pd_es, aggregation, transform, self.primitive)
