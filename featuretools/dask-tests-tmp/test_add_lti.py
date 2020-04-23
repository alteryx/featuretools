# flake8: noqa
import os
from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

import utils

import featuretools as ft


def run_test():
    client = Client()
    data_path = os.path.join("data", "instacart", "dask_data")
    order_products = dd.read_csv([os.path.join(data_path, "order_products_*.csv")])
    orders = dd.read_csv([os.path.join(data_path, "orders_*.csv")])

    pd_order_products = order_products.compute()
    pd_orders = orders.compute()

    print("Creating entityset...")
    order_products_vtypes = {
        "order_id": ft.variable_types.Id,
        "reordered": ft.variable_types.Boolean,
        "product_name": ft.variable_types.Categorical,
        "aisle_id": ft.variable_types.Categorical,
        "department": ft.variable_types.Categorical,
        "order_time": ft.variable_types.Datetime,
        "order_product_id": ft.variable_types.Index,
    }

    order_vtypes = {
        "order_id": ft.variable_types.Index,
        "user_id": ft.variable_types.Id,
        "order_time": ft.variable_types.DatetimeTimeIndex,
    }

    es = ft.EntitySet("instacart")
    es.entity_from_dataframe(entity_id="order_products",
                             dataframe=order_products,
                             index="order_product_id",
                             variable_types=order_products_vtypes,
                             time_index="order_time")

    es.entity_from_dataframe(entity_id="orders",
                             dataframe=orders,
                             index="order_id",
                             variable_types=order_vtypes,
                             time_index="order_time")

    pd_es = ft.EntitySet("instacart")
    pd_es.entity_from_dataframe(entity_id="order_products",
                                dataframe=pd_order_products,
                                index="order_product_id",
                                time_index="order_time")

    pd_es.entity_from_dataframe(entity_id="orders",
                                dataframe=pd_orders,
                                index="order_id",
                                time_index="order_time")

    print("Adding relationships...")
    es.add_relationship(ft.Relationship(es["orders"]["order_id"], es["order_products"]["order_id"]))
    pd_es.add_relationship(ft.Relationship(pd_es["orders"]["order_id"], pd_es["order_products"]["order_id"]))

    print("Normalizing entity...")
    es.normalize_entity(base_entity_id="orders", new_entity_id="users", index="user_id")
    pd_es.normalize_entity(base_entity_id="orders", new_entity_id="users", index="user_id")

    print("Adding last time indexes...")
    es.add_last_time_indexes()
    pd_es.add_last_time_indexes()

    return es, pd_es


if __name__ == "__main__":
    dask_es, pd_es = run_test()
    print('done, computing ltis')
    ltis = {}
    for entity in dask_es.entities:
        ltis[entity.id] = (entity.last_time_index.compute().sort_index(), )
        ltis[entity.id][0].index.name = None
    for entity in pd_es.entities:
        ltis[entity.id] += (entity.last_time_index.sort_index(), )

    print("checking if series are equal")
    for eid, (d_lti, p_lti) in ltis.items():
        print(eid)
        pd.testing.assert_series_equal(d_lti, p_lti, check_names=False)
    print('LTIs are equal')
