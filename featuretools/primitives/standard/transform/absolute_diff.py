from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base import TransformPrimitive


class AbsoluteDiff(TransformPrimitive):
    """Calculates the absolute difference from the previous element
       in a list of numbers.

    Description:
        The absolute difference from the previous element is computed for
        all elements in the input. The first item in the output will always
        be nan, since there is no previous element for the first element.
        Elements in the input containing nan will be filled using either a
        forward-fill or backward-fill method, specified by the method argument.

    Args:
        method (str): Method to use for filling nan values in reindexed
            Series. Possible values are ['pad', 'ffill', 'backfill', 'bfill'].
            Default is 'ffill'.

            `pad / ffill`: propagate last valid observation forward
                to fill gap

            `backfill / bfill`: propagate next valid observation backward
                to fill gap

        limit (int): The max number of consecutive NaN values in a gap that
            can be filled. Default is None.

    Examples:
        >>> absolute_diff = AbsoluteDiff()
        >>> absolute_diff([2, 5, 15, 3]).tolist()
        [nan, 3.0, 10.0, 12.0]

        Forward filling of input elements using the 'ffill' argument

        >>> absolute_diff_ffill = AbsoluteDiff(method="ffill")
        >>> absolute_diff_ffill([None, 5, 10, 20, None, 10, None]).tolist()
        [nan, nan, 5.0, 10.0, 0.0, 10.0, 0.0]

        Backward filling of input element using the 'bfill' argument

        >>> absolute_diff_bfill = AbsoluteDiff(method="bfill")
        >>> absolute_diff_bfill([None, 5, 10, 20, None, 10, None]).tolist()
        [nan, 0.0, 5.0, 10.0, 10.0, 0.0, nan]

        The number of nan values that are filled can be limited

        >>> absolute_diff_limitfill = AbsoluteDiff(limit=2)
        >>> absolute_diff_limitfill([2, None, None, None, 3, 1]).tolist()
        [nan, 0.0, 0.0, nan, nan, 2.0]

    """

    name = "absolute_diff"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})

    def __init__(self, method="ffill", limit=None):
        if method not in ["backfill", "bfill", "pad", "ffill"]:
            raise ValueError("Invalid method")
        self.method = method
        self.limit = limit

    def get_function(self):
        def absolute_diff(data):
            return data.fillna(method=self.method, limit=self.limit).diff().abs()

        return absolute_diff
