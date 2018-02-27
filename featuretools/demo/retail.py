import os
from builtins import str

import pandas as pd

import featuretools as ft
from featuretools.config import config as ft_config


def load_retail(id='demo_retail_data', nrows=None):
    '''
    Returns the retail entityset example

    Args:
        id (str):  id to assign to EntitySet
        nrows (int):  number of rows to load of item_purchases
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
                transactions (shape = [541909, 6])

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
                transactions (shape = [1000, 6])

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

    df.rename(columns={'Unnamed: 0': 'transaction_id',
                       'InvoiceNo': 'invoice_id',
                       'StockCode': 'product_id',
                       'Description': 'description',
                       'Quantity': 'quantity',
                       'InvoiceDate': 'invoice_date',
                       'UnitPrice': 'price',
                       'CustomerID': 'customer_id',
                       'Country': 'country'},
              inplace=True)

    es.entity_from_dataframe("transactions",
                             dataframe=df,
                             index="transaction_id",
                             time_index="invoice_date")

    es.normalize_entity(new_entity_id="items",
                        base_entity_id="transactions",
                        index="product_id",
                        additional_variables=["description"])

    es.normalize_entity(new_entity_id="invoices",
                        base_entity_id="transactions",
                        index="invoice_id",
                        additional_variables=["customer_id", "country"])

    es.normalize_entity(new_entity_id="customers",
                        base_entity_id="invoices",
                        index="customer_id",
                        additional_variables=["country"])
    es.add_last_time_indexes()

    return es


def make_retail_pathname(nrows):
    file_name = 'uk_online_retail_' + str(nrows) + '.csv'
    return os.path.join(ft_config['csv_save_location'], file_name)
