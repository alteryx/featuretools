import os
from builtins import str

import pandas as pd

import featuretools as ft
from featuretools.config import config as ft_config


def load_retail(id='demo_retail_data', nrows=None):
    '''
    Returns the retail entityset example.

    Args:
        id (str):  Id to assign to EntitySet.
        nrows (int):  Number of rows to load of item_purchases
            entity. If None, load all.

    Examples:

        .. ipython::
            :verbatim:

            In [1]: import featuretools as ft

            In [2]: es = ft.demo.load_retail()

            In [3]: es
            Out[3]:
            Entityset: demo_retail_data
              Entities:
                invoices (shape = [25900, 3])
                items (shape = [4070, 3])
                customers (shape = [4373, 3])
                item_purchases (shape = [541909, 6])

        Load in subset of data

        .. ipython::
            :verbatim:

            In [2]: es = ft.demo.load_retail(nrows=1000)

            In [3]: es
            Out[3]:
            Entityset: demo_retail_data
              Entities:
                invoices (shape = [66, 3])
                items (shape = [590, 3])
                customers (shape = [49, 3])
                item_purchases (shape = [1000, 6])

    '''
    demo_save_path = make_retail_pathname(nrows)

    es = ft.EntitySet(id)
    csv_s3 = "s3://featuretools-static/uk_online_retail.csv"

    if not os.path.isfile(demo_save_path):
        df = pd.read_csv(csv_s3,
                         nrows=nrows,
                         parse_dates=["InvoiceDate"])
        df.to_csv(demo_save_path)

    df = pd.read_csv(demo_save_path,
                     nrows=nrows,
                     parse_dates=["InvoiceDate"])

    df.rename(columns={"Unnamed: 0": 'item_purchase_id'}, inplace=True)

    es.entity_from_dataframe("item_purchases",
                             dataframe=df,
                             index="item_purchase_id",
                             time_index="InvoiceDate")

    es.normalize_entity(new_entity_id="items",
                        base_entity_id="item_purchases",
                        index="StockCode",
                        additional_variables=["Description"])

    es.normalize_entity(new_entity_id="invoices",
                        base_entity_id="item_purchases",
                        index="InvoiceNo",
                        additional_variables=["CustomerID", "Country"])

    es.normalize_entity(new_entity_id="customers",
                        base_entity_id="invoices",
                        index="CustomerID",
                        additional_variables=["Country"])
    es.add_last_time_indexes()

    return es


def make_retail_pathname(nrows):
    file_name = 'uk_online_retail_' + str(nrows) + '.csv'
    return os.path.join(ft_config['csv_save_location'], file_name)
