import pandas as pd
from woodwork.logical_types import NaturalLanguage

import featuretools as ft


def load_retail(id="demo_retail_data", nrows=None, return_single_table=False):
    """Returns the retail entityset example.
    The original dataset can be found `here <https://archive.ics.uci.edu/ml/datasets/online+retail>`_.

    We have also made some modifications to the data. We
    changed the column names, converted the ``customer_id``
    to a unique fake ``customer_name``, dropped duplicates,
    added columns for ``total`` and ``cancelled`` and
    converted amounts from GBP to USD. You can download the modified CSV in gz `compressed (7 MB)
    <https://oss.alteryx.com/datasets/online-retail-logs-2018-08-28.csv.gz>`_
    or `uncompressed (43 MB)
    <https://oss.alteryx.com/datasets/online-retail-logs-2018-08-28.csv>`_ formats.

    Args:
        id (str):  Id to assign to EntitySet.
        nrows (int):  Number of rows to load of the underlying CSV.
            If None, load all.
        return_single_table (bool): If True, return a CSV rather than an EntitySet. Default is False.

    Examples:

        .. ipython::
            :verbatim:

            In [1]: import featuretools as ft

            In [2]: es = ft.demo.load_retail()

            In [3]: es
            Out[3]:
            Entityset: demo_retail_data
              DataFrames:
                orders (shape = [22190, 3])
                products (shape = [3684, 3])
                customers (shape = [4372, 2])
                order_products (shape = [401704, 7])

        Load in subset of data

        .. ipython::
            :verbatim:

            In [4]: es = ft.demo.load_retail(nrows=1000)

            In [5]: es
            Out[5]:
            Entityset: demo_retail_data
              DataFrames:
                orders (shape = [67, 5])
                products (shape = [606, 3])
                customers (shape = [50, 2])
                order_products (shape = [1000, 7])
    """
    es = ft.EntitySet(id)
    csv_s3_gz = (
        "https://oss.alteryx.com/datasets/online-retail-logs-2018-08-28.csv.gz?library=featuretools&version="
        + ft.__version__
    )
    csv_s3 = (
        "https://oss.alteryx.com/datasets/online-retail-logs-2018-08-28.csv?library=featuretools&version="
        + ft.__version__
    )
    # Try to read in gz compressed file
    try:
        df = pd.read_csv(csv_s3_gz, nrows=nrows, parse_dates=["order_date"])
    # Fall back to uncompressed
    except Exception:
        df = pd.read_csv(csv_s3, nrows=nrows, parse_dates=["order_date"])
    if return_single_table:
        return df

    es.add_dataframe(
        dataframe_name="order_products",
        dataframe=df,
        index="order_product_id",
        make_index=True,
        time_index="order_date",
        logical_types={"description": NaturalLanguage},
    )

    es.normalize_dataframe(
        new_dataframe_name="products",
        base_dataframe_name="order_products",
        index="product_id",
        additional_columns=["description"],
    )

    es.normalize_dataframe(
        new_dataframe_name="orders",
        base_dataframe_name="order_products",
        index="order_id",
        additional_columns=["customer_name", "country", "cancelled"],
    )

    es.normalize_dataframe(
        new_dataframe_name="customers",
        base_dataframe_name="orders",
        index="customer_name",
    )
    es.add_last_time_indexes()

    return es
