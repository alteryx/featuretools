import dask.dataframe as dd
import pandas as pd

from featuretools.utils.gen_utils import import_or_none, is_instance

ks = import_or_none('databricks.koalas')


def to_pandas(df, index=None, sort_index=False, int_index=False):
    '''
    Testing util to convert dataframes to pandas. If a pandas dataframe is passed in, just returns the dataframe.

    Args:
        index (str, optional): column name to set as index, defaults to None
        sort_index (bool, optional): whether to sort the dataframe on the index after setting it, defaults to False
        int_index (bool, optional): Converts computed dask index to Int64Index to avoid errors, defaults to False

    Returns:
        Pandas DataFrame
    '''
    if isinstance(df, (pd.DataFrame, pd.Series)):
        return df

    if isinstance(df, (dd.DataFrame, dd.Series)):
        pd_df = df.compute()
    if is_instance(df, (ks, ks), ('DataFrame', 'Series')):
        pd_df = df.to_pandas()

    if index:
        pd_df = pd_df.set_index(index)
    if sort_index:
        pd_df = pd_df.sort_index()
    if int_index and isinstance(df, dd.DataFrame):
        pd_df.index = pd.Int64Index(pd_df.index)

    return pd_df
