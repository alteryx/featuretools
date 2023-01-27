from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable

from featuretools.primitives.base import TransformPrimitive


class SameAsPrevious(TransformPrimitive):
    """Determines if a value is equal to the previous value in a list.

    Description:
        Compares a value in a list to the previous value and returns True if
        the value is equal to the previous value or False otherwise. The
        first item in the output will always be False, since there is no previous
        element for the first element comparison.

        Any nan values in the input will be filled using either a forward-fill
        or backward-fill method, specified by the fill_method argument. The number
        of consecutive nan values that get filled can be limited with the limit
        argument. Any nan values left after filling will result in False being
        returned for any comparison involving the nan value.

    Args:
        fill_method (str): Method for filling gaps in series. Valid
        options are `backfill`, `bfill`, `pad`, `ffill`.
        `pad / ffill`: fill gap with last valid observation.
        `backfill / bfill`: fill gap with next valid observation.
        Default is `pad`.

        limit (int): The max number of consecutive NaN values in a gap that
            can be filled. Default is None.

    Examples:
        >>> same_as_previous = SameAsPrevious()
        >>> same_as_previous([1, 2, 2, 4]).tolist()
        [False, False, True, False]

        The fill method for nan values can be specified

        >>> same_as_previous_fillna = SameAsPrevious(fill_method="bfill")
        >>> same_as_previous_fillna([1, None, 2, 4]).tolist()
        [False, False, True, False]

        The number of nan values that are filled can be limited

        >>> same_as_previous_limitfill = SameAsPrevious(limit=2)
        >>> same_as_previous_limitfill([1, None, None, None, 2, 3]).tolist()
        [False, True, True, False, False, False]
    """

    name = "same_as_previous"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(BooleanNullable)

    def __init__(self, fill_method="pad", limit=None):
        if fill_method not in ["backfill", "bfill", "pad", "ffill"]:
            raise ValueError("Invalid fill_method")
        self.fill_method = fill_method
        self.limit = limit

    def get_function(self):
        def same_as_previous(x):
            x = x.fillna(method=self.fill_method, limit=self.limit)
            x = x.eq(x.shift())
            # first value will always be false, since there is no previous value
            x.iloc[0] = False
            return x

        return same_as_previous
