from featuretools.primitives.base import TransformPrimitive
from featuretools.variable_types import Discrete, Id, Index, Numeric, TimeIndex


class CumSum(TransformPrimitive):
    """Returns the cumulative sum after grouping"""
     
    name = "cum_sum"
    input_types = [[Numeric, Id],
                   [Numeric, Discrete]]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_sum(values, groups):
            return values.groupby(groups).cumsum()

        return cum_sum

    def generate_name(self, base_feature_names):
        return "CUM_SUM(%s by %s)" % (base_feature_names[0], base_feature_names[1])


class CumCount(TransformPrimitive):
    """Returns the cumulative count after grouping"""
     
    name = "cum_count"
    input_types = [[Id], [Discrete]]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_count(values):
            return values.groupby(values).cumcount().add(1)

        return cum_count

    def generate_name(self, base_feature_names):
        return "CUM_COUNT(%s)" % (base_feature_names[0])


class CumMean(TransformPrimitive):
    """Returns the cumulative mean after grouping"""
     
    name = "cum_mean"
    input_types = [[Numeric, Id],
                   [Numeric, Discrete]]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_mean(values, groups):
            temp = values.groupby(groups)
            return temp.cumsum()/temp.cumcount().add(1)

        return cum_mean

    def generate_name(self, base_feature_names):
        return "CUM_MEAN(%s by %s)" % (base_feature_names[0], base_feature_names[1])


class CumMin(TransformPrimitive):
    """Returns the cumulative min after grouping"""
     
    name = "cum_min"
    input_types = [[Numeric, Id],
                   [Numeric, Discrete]]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_min(values, groups):
            return values.groupby(groups).cummin()

        return cum_min

    def generate_name(self, base_feature_names):
        return "CUM_MIN(%s by %s)" % (base_feature_names[0], base_feature_names[1])


class CumMax(TransformPrimitive):
    """Returns the cumulative max after grouping"""
     
    name = "cum_max"
    input_types = [[Numeric, Id],
                   [Numeric, Discrete]]
    return_type = Numeric
    uses_full_entity = True

    def get_function(self):
        def cum_max(values, groups):
            return values.groupby(groups).cummax()

        return cum_max

    def generate_name(self, base_feature_names):
        return "CUM_MAX(%s by %s)" % (base_feature_names[0], base_feature_names[1])
