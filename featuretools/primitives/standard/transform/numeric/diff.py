from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base import TransformPrimitive


class Diff(TransformPrimitive):
    """Computes the difference between the value in a list and the
    previous value in that list.

    Args:
        periods (int): The number of periods by which to shift the index row.
            Default is 0. Periods correspond to rows.

    Description:
        Given a list of values, compute the difference from the previous
        item in the list. The result for the first element of the list will
        always be `NaN`.

    Examples:
        >>> diff = Diff()
        >>> values = [1, 10, 3, 4, 15]
        >>> diff(values).tolist()
        [nan, 9.0, -7.0, 1.0, 11.0]

        You can specify the number of periods to shift the values

        >>> values = [1, 2, 4, 7, 11, 16]
        >>> diff_periods = Diff(periods = 1)
        >>> diff_periods(values).tolist()
        [nan, nan, 1.0, 2.0, 3.0, 4.0]
    """

    name = "diff"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    uses_full_dataframe = True
    description_template = "the difference from the previous value of {}"

    def __init__(self, periods=0):
        self.periods = periods

    def get_function(self):
        def pd_diff(values):
            return values.shift(self.periods).diff()

        return pd_diff
