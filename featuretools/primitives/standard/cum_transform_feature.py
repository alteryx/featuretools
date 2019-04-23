import numpy as np

from featuretools.primitives.base import TransformPrimitive
from featuretools.variable_types import Discrete, Id, Numeric


class CumSum(TransformPrimitive):
    """Calculates the cumulative sum.

    Description:
        Given a list of values, return the cumulative sum
        (or running total). There is no set window, so the
        sum at each point is calculated over all prior values.
        `NaN` values will return `NaN`, but in the window of a
        cumulative caluclation, they're ignored.

    Examples:
        >>> cum_sum = CumSum()
        >>> cum_sum([1, 2, 3, 4, None, 5]).tolist()
        [1.0, 3.0, 6.0, 10.0, nan, 15.0]
    """
    name = "cum_sum"
    input_types = [Numeric]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_sum(values):
            return values.cumsum()

        return cum_sum


class CumCount(TransformPrimitive):
    """Calculates the cumulative count.

    Description:
        Given a list of values, return the cumulative count
        (or running count). There is no set window, so the
        count at each point is calculated over all prior
        values. `NaN` values are counted.

    Examples:
        >>> cum_count = CumCount()
        >>> cum_count([1, 2, 3, 4, None, 5]).tolist()
        [1, 2, 3, 4, 5, 6]
    """
    name = "cum_count"
    input_types = [[Id], [Discrete]]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_count(values):
            return np.arange(1, len(values) + 1)

        return cum_count


class CumMean(TransformPrimitive):
    """Calculates the cumulative mean.

    Description:
        Given a list of values, return the cumulative mean
        (or running mean). There is no set window, so the
        mean at each point is calculated over all prior values.
        `NaN` values will return `NaN`, but in the window of a
        cumulative caluclation, they're treated as 0.

    Examples:
        >>> cum_mean = CumMean()
        >>> cum_mean([1, 2, 3, 4, None, 5]).tolist()
        [1.0, 1.5, 2.0, 2.5, nan, 2.5]
    """
    name = "cum_mean"
    input_types = [Numeric]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_mean(values):
            return values.cumsum() / np.arange(1, len(values) + 1)

        return cum_mean


class CumMin(TransformPrimitive):
    """Calculates the cumulative minimum.

    Description:
        Given a list of values, return the cumulative min
        (or running min). There is no set window, so the min
        at each point is calculated over all prior values.
        `NaN` values will return `NaN`, but in the window of a
        cumulative caluclation, they're ignored.

    Examples:
        >>> cum_min = CumMin()
        >>> cum_min([1, 2, -3, 4, None, 5]).tolist()
        [1.0, 1.0, -3.0, -3.0, nan, -3.0]
    """
    name = "cum_min"
    input_types = [Numeric]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_min(values):
            return values.cummin()

        return cum_min


class CumMax(TransformPrimitive):
    """Calculates the cumulative maximum.

    Description:
        Given a list of values, return the cumulative max
        (or running max). There is no set window, so the max
        at each point is calculated over all prior values.
        `NaN` values will return `NaN`, but in the window of a
        cumulative caluclation, they're ignored.

    Examples:
        >>> cum_max = CumMax()
        >>> cum_max([1, 2, 3, 4, None, 5]).tolist()
        [1.0, 2.0, 3.0, 4.0, nan, 5.0]
    """
    name = "cum_max"
    input_types = [Numeric]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_max(values):
            return values.cummax()

        return cum_max
