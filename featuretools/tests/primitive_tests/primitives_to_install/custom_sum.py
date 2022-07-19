from woodwork.column_schema import ColumnSchema

from featuretools.primitives.core import AggregationPrimitive


class CustomSum(AggregationPrimitive):
    name = "custom_sum"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
