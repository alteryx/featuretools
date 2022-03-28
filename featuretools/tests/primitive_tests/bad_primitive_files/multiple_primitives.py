from woodwork.column_schema import ColumnSchema

from featuretools.primitives import AggregationPrimitive


class CustomMax(AggregationPrimitive):
    name = "custom_max"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})


class CustomSum(AggregationPrimitive):
    name = "custom_sum"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
