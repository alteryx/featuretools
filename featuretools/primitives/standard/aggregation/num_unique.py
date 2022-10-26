import pandas as pd
from dask import dataframe as dd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.utils.gen_utils import Library


class NumUnique(AggregationPrimitive):
    """Determines the number of distinct values, ignoring `NaN` values.

    Examples:
        >>> num_unique = NumUnique()
        >>> num_unique(['red', 'blue', 'green', 'yellow'])
        4

        `NaN` values will be ignored.

        >>> num_unique(['red', 'blue', 'green', 'yellow', None])
        4
    """

    name = "num_unique"
    input_types = [ColumnSchema(semantic_tags={"category"})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the number of unique elements in {}"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type == Library.DASK:

            def chunk(s):
                def inner_chunk(x):
                    x = x[:].dropna()
                    return set(x.unique())

                return s.agg(inner_chunk)

            def agg(s):
                def inner_agg(x):
                    x = x[:].dropna()
                    return set().union(*x.values)

                return s.agg(inner_agg)

            def finalize(s):
                return s.apply(lambda x: len(x))

            return dd.Aggregation(self.name, chunk=chunk, agg=agg, finalize=finalize)

        elif agg_type == Library.SPARK:
            return "nunique"

        return pd.Series.nunique
