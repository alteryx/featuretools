from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base import TransformPrimitive


class CumMax(TransformPrimitive):
    """Calculates the cumulative maximum.

    Description:
        Given a list of values, return the cumulative max
        (or running max). There is no set window, so the max
        at each point is calculated over all prior values.
        `NaN` values will return `NaN`, but in the window of a
        cumulative caluclation, they're ignored.

    Examples:
        >>> cum_max = CumMax()
        >>> cum_max([1, 2, 3, 4, None, 5]).tolist()
        [1.0, 2.0, 3.0, 4.0, nan, 5.0]
    """

    name = "cum_max"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    uses_full_dataframe = True
    description_template = "the cumulative maximum of {}"

    def get_function(self):
        def cum_max(values):
            return values.cummax()

        return cum_max
