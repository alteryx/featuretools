from woodwork.column_schema import ColumnSchema

from featuretools.primitives import AggregationPrimitive


class CustomMax(AggregationPrimitive):
    name = "custom_max"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})

    def get_function(self):
        return lambda x: max(x)


class CustomSum(AggregationPrimitive):
    name = "custom_sum"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})

    def get_function(self):
        return lambda x: sum(x)
