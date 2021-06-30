import numpy as np
import pandas as pd

from featuretools.utils.latlong_utils import replace_latlong_nan


def test_replace_latlong_nan():
    values = pd.Series([(np.nan, np.nan), np.nan, (10, 5)])
    result = replace_latlong_nan(values)
    assert result[0] == values[0]
    assert result[1] == (np.nan, np.nan)
    assert result[2] == values[2]
