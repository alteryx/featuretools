from woodwork.column_schema import ColumnSchema

from featuretools.primitives.core import AggregationPrimitive


class CustomMean(AggregationPrimitive):
    name = "custom_mean"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
