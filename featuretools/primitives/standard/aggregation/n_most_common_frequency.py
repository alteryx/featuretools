import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical

from featuretools.primitives.base import AggregationPrimitive


class NMostCommonFrequency(AggregationPrimitive):
    """Determines the frequency of the n most common items.

    Args:
        n (int): defines "n" in "n most common". Defaults to
            3.
        skipna (bool): Determines if to use NA/null values.
            Defaults to True to skip NA/null.

    Description:
        Given a list, find the n most common items, and return a series
        showing the frequency of each item. If the list has less than n unique
        values, the resulting series will be padded with nan.

    Examples:
        >>> n_most_common_frequency = NMostCommonFrequency()
        >>> n_most_common_frequency([1, 1, 1, 2, 2, 3, 4, 4]).to_list()
        [3, 2, 2]

        We can increase n to include more items.

        >>> n_most_common_frequency = NMostCommonFrequency(4)
        >>> n_most_common_frequency([1, 1, 1, 2, 2, 3, 4, 4]).to_list()
        [3, 2, 2, 1]

        `NaN`s are skipped by default.

        >>> n_most_common_frequency = NMostCommonFrequency(3)
        >>> n_most_common_frequency([1, 1, 1, 2, 2, 3, 4, 4, None, None, None]).to_list()
        [3, 2, 2]

        However, the way `NaN`s are treated can be controlled.

        >>> n_most_common_frequency = NMostCommonFrequency(3, skipna=False)
        >>> n_most_common_frequency([1, 1, 1, 2, 2, 3, 4, 4, None, None, None]).to_list()
        [3, 3, 2]
    """

    name = "n_most_common_frequency"
    input_types = [ColumnSchema(semantic_tags={"category"})]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={"category"})

    def __init__(self, n=3, skipna=True):
        self.n = n
        self.number_output_features = n
        self.skipna = skipna

    def get_function(self):
        def n_most_common_frequency(data, n=self.n):
            frequencies = data.value_counts(dropna=self.skipna)
            n_most_common = frequencies.iloc[0:n]
            nan_add = n - frequencies.shape[0]
            if nan_add > 0:
                n_most_common = pd.concat(
                    [n_most_common, pd.Series([np.nan] * nan_add)],
                )
            return n_most_common

        return n_most_common_frequency
