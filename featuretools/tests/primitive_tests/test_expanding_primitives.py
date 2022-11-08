import pandas as pd
import pytest

from featuretools.primitives.standard.transform.time_series.expanding import (
    ExpandingCount,
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingSTD,
    ExpandingTrend,
)
from featuretools.utils import calculate_trend


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
    ],
)
def test_expanding_count(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    expected = test.expanding(min_periods=min_periods).count().values
    primitive_instance = ExpandingCount(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(window_series_pd.index)
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
    ],
)
def test_expanding_min(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    expected = test.expanding(min_periods=min_periods).min().values
    primitive_instance = ExpandingMin(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
    ],
)
def test_expanding_max(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    expected = test.expanding(min_periods=min_periods).max().values
    primitive_instance = ExpandingMax(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
    ],
)
def test_expanding_std(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    expected = test.expanding(min_periods=min_periods).std().values
    primitive_instance = ExpandingSTD(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
    ],
)
def test_expanding_mean(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    expected = test.expanding(min_periods=min_periods).mean().values
    primitive_instance = ExpandingMean(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))


@pytest.mark.parametrize(
    "min_periods, gap",
    [
        (5, 2),
        (5, 0),
    ],
)
def test_expanding_trend(window_series_pd, min_periods, gap):
    test = window_series_pd.shift(gap)
    expected = test.expanding(min_periods=min_periods).aggregate(calculate_trend).values
    primitive_instance = ExpandingTrend(min_periods=min_periods, gap=gap).get_function()
    actual = primitive_instance(
        numeric=window_series_pd,
        datetime=window_series_pd.index,
    )
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))


@pytest.mark.parametrize(
    "primitive",
    [
        ExpandingMax,
        ExpandingMean,
        ExpandingMin,
        ExpandingSTD,
        ExpandingTrend,
    ],
)
def test_expanding_primitives_throw_warning(window_series_pd, primitive):
    with pytest.raises(ValueError):
        primitive(gap="2H").get_function()(
            numeric=window_series_pd,
            datetime=window_series_pd.index,
        )
