import pandas as pd

from featuretools.utils.gen_utils import import_or_none, is_instance

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")


def to_pandas(df, index=None, sort_index=False, int_index=False):
    """
    Testing util to convert dataframes to pandas. If a pandas dataframe is passed in, just returns the dataframe.

    Args:
        index (str, optional): column name to set as index, defaults to None
        sort_index (bool, optional): whether to sort the dataframe on the index after setting it, defaults to False
        int_index (bool, optional): Converts computed dask index to Int64Index to avoid errors, defaults to False

    Returns:
        Pandas DataFrame
    """
    if isinstance(df, (pd.DataFrame, pd.Series)):
        return df

    if is_instance(df, (dd, dd), ("DataFrame", "Series")):
        pd_df = df.compute()
    if is_instance(df, (ps, ps), ("DataFrame", "Series")):
        pd_df = df.to_pandas()

    if index:
        pd_df = pd_df.set_index(index)
    if sort_index:
        pd_df = pd_df.sort_index()
    if int_index and is_instance(df, dd, "DataFrame"):
        pd_df.index = pd.Index(pd_df.index, dtype="Int64")

    return pd_df


def get_df_tags(df):
    """Gets a DataFrame's semantic tags without index or time index tags for Woodwork init"""
    semantic_tags = {}
    for col_name in df.columns:
        semantic_tags[col_name] = df.ww.semantic_tags[col_name] - {
            "time_index",
            "index",
        }

    return semantic_tags
