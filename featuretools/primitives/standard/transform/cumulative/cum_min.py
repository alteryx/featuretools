from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base import TransformPrimitive


class CumMin(TransformPrimitive):
    """Calculates the cumulative minimum.

    Description:
        Given a list of values, return the cumulative min
        (or running min). There is no set window, so the min
        at each point is calculated over all prior values.
        `NaN` values will return `NaN`, but in the window of a
        cumulative caluclation, they're ignored.

    Examples:
        >>> cum_min = CumMin()
        >>> cum_min([1, 2, -3, 4, None, 5]).tolist()
        [1.0, 1.0, -3.0, -3.0, nan, -3.0]
    """

    name = "cum_min"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    uses_full_dataframe = True
    description_template = "the cumulative minimum of {}"

    def get_function(self):
        def cum_min(values):
            return values.cummin()

        return cum_min
