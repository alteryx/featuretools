import numpy as np
import pandas as pd

from featuretools.primitives.base import TransformPrimitive
from featuretools.variable_types import Discrete, Id, Numeric


class CumSum(TransformPrimitive):
    """Returns the cumulative sum after grouping"""

    name = "cum_sum"
    input_types = [Numeric]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_sum(values):
            return values.cumsum()

        return cum_sum


class CumCount(TransformPrimitive):
    """Returns the cumulative count after grouping"""

    name = "cum_count"
    input_types = [[Id], [Discrete]]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_count(values):
            return np.arange(1, len(values) + 1)

        return cum_count


class CumMean(TransformPrimitive):
    """Returns the cumulative mean after grouping"""

    name = "cum_mean"
    input_types = [Numeric]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_mean(values):
            return values.cumsum() / pd.Series(range(1, len(values) + 1), index=values.index)

        return cum_mean


class CumMin(TransformPrimitive):
    """Returns the cumulative min after grouping"""

    name = "cum_min"
    input_types = [Numeric]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_min(values):
            return values.cummin()

        return cum_min


class CumMax(TransformPrimitive):
    """Returns the cumulative max after grouping"""

    name = "cum_max"
    input_types = [Numeric]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_max(values):
            return values.cummax()

        return cum_max
