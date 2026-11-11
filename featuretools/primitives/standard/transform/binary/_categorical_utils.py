import pandas as pd


def align_categoricals(x_vals, y_vals):
    """Make two categorical Series safe to compare.

    pandas only compares categoricals when their dtypes match (same
    categories and same ordering). When they differ, compare the
    underlying values instead of the category codes.
    """
    if (
        isinstance(x_vals.dtype, pd.CategoricalDtype)
        and isinstance(y_vals.dtype, pd.CategoricalDtype)
        and x_vals.dtype != y_vals.dtype
    ):
        return x_vals.astype(object), y_vals.astype(object)
    return x_vals, y_vals
