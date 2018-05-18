import os
from builtins import str

import pandas as pd

import featuretools as ft
import featuretools.variable_types as vtypes
from featuretools.config import config as ft_config


def load_retail(id='demo_retail_data', nrows=None, use_cache=True):
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
                orders (shape = [25900, 3])
                products (shape = [4070, 3])
                customers (shape = [4373, 3])
                order_products (shape = [541909, 6])

        Load in subset of data

        .. ipython::
            :verbatim:

            In [2]: es = ft.demo.load_retail(nrows=1000)

            In [3]: es
            Out[3]:
            Entityset: demo_retail_data
              Entities:
                orders (shape = [66, 3])
                products (shape = [590, 3])
                customers (shape = [49, 3])
                order_products (shape = [1000, 6])

    '''
    demo_save_path = make_retail_pathname(nrows)

    es = ft.EntitySet(id)
    csv_s3 = "https://s3.amazonaws.com/featurelabs-static/online-retail-logs.csv"

    if not use_cache or not os.path.isfile(demo_save_path):

        df = pd.read_csv(csv_s3,
                         nrows=nrows,
                         parse_dates=["order_date"])
        df.to_csv(demo_save_path, index_label='order_product_id')

    df = pd.read_csv(demo_save_path,
                     nrows=nrows,
                     parse_dates=["order_date"])

    es.entity_from_dataframe("order_products",
                             dataframe=df,
                             index="order_product_id",
                             variable_types={'description': vtypes.Text})

    es.normalize_entity(new_entity_id="products",
                        base_entity_id="order_products",
                        index="product_id",
                        additional_variables=["description"])

    es.normalize_entity(new_entity_id="orders",
                        base_entity_id="order_products",
                        index="order_id",
                        additional_variables=[
                            "customer_id", "country", "order_date"],
                        make_time_index="order_date")

    es.normalize_entity(new_entity_id="customers",
                        base_entity_id="orders",
                        index="customer_id",
                        additional_variables=["country"])
    es.add_last_time_indexes()

    return es


def make_retail_pathname(nrows):
    file_name = 'online_retail_logs_' + str(nrows) + '.csv'
    return os.path.join(ft_config['csv_save_location'], file_name)
