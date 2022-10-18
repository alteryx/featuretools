import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype

from featuretools.primitives import (
    NMostCommon,
    PercentTrue,
    Trend,
    get_aggregation_primitives,
)


def test_nmostcommon_categorical():
    n_most = NMostCommon(3)
    expected = pd.Series([1.0, 2.0, np.nan])

    ints = pd.Series([1, 2, 1, 1]).astype("int64")
    assert pd.Series(n_most(ints)).equals(expected)

    cats = pd.Series([1, 2, 1, 1]).astype("category")
    assert pd.Series(n_most(cats)).equals(expected)

    # Value counts includes data for categories that are not present in data.
    # Make sure these counts are not included in most common outputs
    extra_dtype = CategoricalDtype(categories=[1, 2, 3])
    cats_extra = pd.Series([1, 2, 1, 1]).astype(extra_dtype)
    assert pd.Series(n_most(cats_extra)).equals(expected)


def test_agg_primitives_can_init_without_params():
    agg_primitives = get_aggregation_primitives().values()
    for agg_primitive in agg_primitives:
        agg_primitive()


def test_trend_works_with_different_input_dtypes():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    numeric = pd.Series([1, 2, 3])

    trend = Trend()
    dtypes = ["float64", "int64", "Int64"]

    for dtype in dtypes:
        actual = trend(numeric.astype(dtype), dates)
        assert np.isclose(actual, 1)


def test_percent_true_boolean():
    booleans = pd.Series([True, False, True, pd.NA], dtype="boolean")
    pct_true = PercentTrue()
    pct_true(booleans) == 0.5
