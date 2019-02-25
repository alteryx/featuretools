from featuretools.primitives.base import GroupByGroupByTransformPrimitive
from featuretools.variable_types import Discrete, Id, Numeric


class CumSum(GroupByTransformPrimitive):
    """Returns the cumulative sum after grouping"""

    name = "cum_sum"
    input_types = [[Numeric, Id],
                   [Numeric, Discrete]]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_sum(values):
            return values.cumsum()

        return cum_sum


class CumCount(GroupByTransformPrimitive):
    """Returns the cumulative count after grouping"""

    name = "cum_count"
    input_types = [[Id], [Discrete]]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_count(values):
            return values.cumcount() + 1

        return cum_count


class CumMean(GroupByTransformPrimitive):
    """Returns the cumulative mean after grouping"""

    name = "cum_mean"
    input_types = [[Numeric, Id],
                   [Numeric, Discrete]]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_mean(values):
            return values.cumsum() / (values.cumcount() + 1)

        return cum_mean


class CumMin(GroupByTransformPrimitive):
    """Returns the cumulative min after grouping"""

    name = "cum_min"
    input_types = [[Numeric, Id],
                   [Numeric, Discrete]]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_min(values):
            return values.cummin()

        return cum_min


class CumMax(GroupByTransformPrimitive):
    """Returns the cumulative max after grouping"""

    name = "cum_max"
    input_types = [[Numeric, Id],
                   [Numeric, Discrete]]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_max(values):
            return values.cummax()

        return cum_max
