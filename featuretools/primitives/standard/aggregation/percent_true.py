import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable, Double

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.utils.gen_utils import Library, import_or_none

dd = import_or_none("dask.dataframe")


class PercentTrue(AggregationPrimitive):
    """Determines the percent of `True` values.

    Description:
        Given a list of booleans, return the percent
        of values which are `True` as a decimal.
        `NaN` values are treated as `False`,
        adding to the denominator.

    Examples:
        >>> percent_true = PercentTrue()
        >>> percent_true([True, False, True, True, None])
        0.6
    """

    name = "percent_true"
    input_types = [
        [ColumnSchema(logical_type=BooleanNullable)],
        [ColumnSchema(logical_type=Boolean)],
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    stack_on = []
    stack_on_exclude = []
    default_value = pd.NA
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "the percentage of true values in {}"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type == Library.DASK:

            def chunk(s):
                def format_chunk(x):
                    return x[:].fillna(False)

                chunk_sum = s.agg(lambda x: format_chunk(x).sum())
                chunk_len = s.agg(lambda x: len(format_chunk(x)))
                if chunk_sum.dtype == "bool":
                    chunk_sum = chunk_sum.astype("int64")
                if chunk_len.dtype == "bool":
                    chunk_len = chunk_len.astype("int64")
                return (chunk_sum, chunk_len)

            def agg(val, length):
                return (val.sum(), length.sum())

            def finalize(total, length):
                return total / length

            return dd.Aggregation(self.name, chunk=chunk, agg=agg, finalize=finalize)

        def percent_true(s):
            return s.fillna(False).mean()

        return percent_true
