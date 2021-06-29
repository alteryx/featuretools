import numpy as np
import pandas as pd


def replace_latlong_nan(values):
    """replace a single `NaN` value with a tuple: `(np.nan, np.nan)`"""
    return values.where(values.notnull(), pd.Series([(np.nan, np.nan)] * len(values)))
