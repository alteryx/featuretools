from woodwork.column_schema import ColumnSchema

from featuretools.primitives import make_agg_primitive

CustomMax = make_agg_primitive(lambda x: max(x),
                               name="CustomMax",
                               input_types=[ColumnSchema(semantic_tags={'numeric'})],
                               return_type=ColumnSchema(semantic_tags={'numeric'}))

CustomSum = make_agg_primitive(lambda x: sum(x),
                               name="CustomSum",
                               input_types=[ColumnSchema(semantic_tags={'numeric'})],
                               return_type=ColumnSchema(semantic_tags={'numeric'}))
