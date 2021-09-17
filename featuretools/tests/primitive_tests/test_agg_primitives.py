import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype

from featuretools.primitives import NMostCommon


def test_nmostcommon_categorical():
    n_most = NMostCommon(3)
    expected = pd.Series([1.0, 2.0, np.nan])

    ints = pd.Series([1, 2, 1, 1]).astype('int64')
    assert pd.Series(n_most(ints)).equals(expected)

    cats = pd.Series([1, 2, 1, 1]).astype('category')
    assert pd.Series(n_most(cats)).equals(expected)

    # Value counts includes data for categories that are not present in data.
    # Make sure these counts are not included in most common outputs
    extra_dtype = CategoricalDtype(categories=[1, 2, 3])
    cats_extra = pd.Series([1, 2, 1, 1]).astype(extra_dtype)
    assert pd.Series(n_most(cats_extra)).equals(expected)
