from datetime import datetime
from math import isnan

import numpy as np
import pandas as pd

from featuretools.primitives import (
    TimeSinceLastFalse,
    TimeSinceLastMax,
    TimeSinceLastMin,
    TimeSinceLastTrue,
)


class TestTimeSinceLastFalse:
    primitive = TimeSinceLastFalse
    cutoff_time = datetime(2011, 4, 9, 11, 31, 27)
    times = pd.Series(
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)]
        + [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)],
    )
    booleans = pd.Series([True] * 5 + [False] * 4)

    def test_booleans(self):
        primitive_func = self.primitive().get_function()
        answer = self.cutoff_time - datetime(2011, 4, 9, 10, 31, 27)
        assert (
            primitive_func(
                self.times,
                self.booleans,
                time=self.cutoff_time,
            )
            == answer.total_seconds()
        )

    def test_booleans_reversed(self):
        primitive_func = self.primitive().get_function()
        answer = self.cutoff_time - datetime(2011, 4, 9, 10, 30, 18)
        reversed_booleans = pd.Series(self.booleans.values[::-1])
        assert (
            primitive_func(
                self.times,
                reversed_booleans,
                time=self.cutoff_time,
            )
            == answer.total_seconds()
        )

    def test_no_false(self):
        primitive_func = self.primitive().get_function()
        times = pd.Series([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)])
        booleans = pd.Series([True] * 5)
        assert isnan(primitive_func(times, booleans, time=self.cutoff_time))

    def test_nans(self):
        primitive_func = self.primitive().get_function()
        times = pd.concat([self.times.copy(), pd.Series([np.nan, pd.NaT])])
        booleans = pd.concat(
            [self.booleans.copy(), pd.Series([np.nan], dtype="boolean")],
        )
        times = times.reset_index(drop=True)
        booleans = booleans.reset_index(drop=True)
        answer = self.cutoff_time - datetime(2011, 4, 9, 10, 31, 27)
        assert (
            primitive_func(
                times,
                booleans,
                time=self.cutoff_time,
            )
            == answer.total_seconds()
        )

    def test_empty(self):
        primitive_func = self.primitive().get_function()
        times = pd.Series([], dtype="datetime64[ns]")
        booleans = pd.Series([], dtype="boolean")
        times = times.reset_index(drop=True)
        answer = primitive_func(
            times,
            booleans,
            time=self.cutoff_time,
        )
        assert pd.isna(answer)


class TestTimeSinceLastMax:
    primitive = TimeSinceLastMax
    cutoff_time = datetime(2011, 4, 9, 11, 31, 27)
    times = pd.Series(
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)]
        + [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)],
    )
    numerics = pd.Series([0, 1, 2, 8, 2, 5, 1, 3, 7])
    actual_time_since = cutoff_time - datetime(2011, 4, 9, 10, 30, 18)
    actual_seconds = actual_time_since.total_seconds()

    def test_primitive_func_1(self):
        primitive_func = self.primitive().get_function()
        assert (
            primitive_func(
                self.times,
                self.numerics,
                time=self.cutoff_time,
            )
            == self.actual_seconds
        )

    def test_no_max(self):
        primitive_func = self.primitive().get_function()
        times = pd.Series([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)])
        numerics = pd.Series([0] * 5)
        actual_time_since = self.cutoff_time - datetime(2011, 4, 9, 10, 30, 0)
        actual_seconds = actual_time_since.total_seconds()
        assert primitive_func(times, numerics, time=self.cutoff_time) == actual_seconds

    def test_nans(self):
        primitive_func = self.primitive().get_function()
        times = pd.concat([self.times.copy(), pd.Series([np.nan, pd.NaT])])
        numerics = pd.concat(
            [self.numerics.copy(), pd.Series([np.nan], dtype="float64")],
        )
        times = times.reset_index(drop=True)
        numerics = numerics.reset_index(drop=True)
        assert (
            primitive_func(
                times,
                numerics,
                time=self.cutoff_time,
            )
            == self.actual_seconds
        )


class TestTimeSinceLastMin:
    primitive = TimeSinceLastMin
    cutoff_time = datetime(2011, 4, 9, 11, 31, 27)
    times = pd.Series(
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)]
        + [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)],
    )
    numerics = pd.Series([1, 0, 2, 8, 2, 5, 1, 3, 7])
    actual_time_since = cutoff_time - datetime(2011, 4, 9, 10, 30, 6)
    actual_seconds = actual_time_since.total_seconds()

    def test_primitive_func_1(self):
        primitive_func = self.primitive().get_function()
        assert (
            primitive_func(
                self.times,
                self.numerics,
                time=self.cutoff_time,
            )
            == self.actual_seconds
        )

    def test_no_max(self):
        primitive_func = self.primitive().get_function()
        times = pd.Series([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)])
        numerics = pd.Series([0] * 5)
        actual_time_since = self.cutoff_time - datetime(2011, 4, 9, 10, 30, 0)
        actual_seconds = actual_time_since.total_seconds()
        assert primitive_func(times, numerics, time=self.cutoff_time) == actual_seconds

    def test_nans(self):
        primitive_func = self.primitive().get_function()
        times = pd.concat(
            [self.times.copy(), pd.Series([np.nan, pd.NaT], dtype="datetime64[ns]")],
        )
        numerics = pd.concat(
            [self.numerics.copy(), pd.Series([np.nan, np.nan], dtype="float64")],
        )
        times = times.reset_index(drop=True)
        numerics = numerics.reset_index(drop=True)
        assert (
            primitive_func(
                times,
                numerics,
                time=self.cutoff_time,
            )
            == self.actual_seconds
        )


class TestTimeSinceLastTrue:
    primitive = TimeSinceLastTrue
    cutoff_time = datetime(2011, 4, 9, 11, 31, 27)
    times = pd.Series(
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)]
        + [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)],
    )
    booleans = pd.Series([True] * 5 + [False] * 4)
    actual_time_since = cutoff_time - datetime(2011, 4, 9, 10, 30, 24)
    actual_seconds = actual_time_since.total_seconds()

    def test_primitive_func_1(self):
        primitive_func = self.primitive().get_function()
        assert (
            primitive_func(
                self.times,
                self.booleans,
                time=self.cutoff_time,
            )
            == self.actual_seconds
        )

    def test_no_true(self):
        primitive_func = self.primitive().get_function()
        times = pd.Series([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)])
        booleans = pd.Series([False] * 5)
        assert isnan(primitive_func(times, booleans, time=self.cutoff_time))

    def test_nans(self):
        primitive_func = self.primitive().get_function()
        times = pd.concat(
            [self.times.copy(), pd.Series([np.nan, pd.NaT], dtype="datetime64[ns]")],
        )
        booleans = pd.concat(
            [self.booleans.copy(), pd.Series([np.nan], dtype="boolean")],
        )
        times = times.reset_index(drop=True)
        booleans = booleans.reset_index(drop=True)
        assert (
            primitive_func(
                times,
                booleans,
                time=self.cutoff_time,
            )
            == self.actual_seconds
        )

    def test_no_cutofftime(self):
        primitive_func = self.primitive().get_function()
        times = pd.Series([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)])
        booleans = pd.Series([False] * 5)
        assert isnan(primitive_func(times, booleans))

    def test_empty(self):
        primitive_func = self.primitive().get_function()
        times = pd.Series([], dtype="datetime64[ns]")
        booleans = pd.Series([], dtype="boolean")
        times = times.reset_index(drop=True)
        answer = primitive_func(
            times,
            booleans,
            time=self.cutoff_time,
        )
        assert pd.isna(answer)
