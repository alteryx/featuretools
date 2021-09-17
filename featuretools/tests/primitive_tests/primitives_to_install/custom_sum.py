from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base import make_agg_primitive

CustomSum = make_agg_primitive(lambda x: sum(x),
                               name="CustomSum",
                               input_types=[ColumnSchema(semantic_tags={'numeric'})],
                               return_type=ColumnSchema(semantic_tags={'numeric'}))
