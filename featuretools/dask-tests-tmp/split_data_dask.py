# flake8: noqa
import math
import os
from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client

import featuretools as ft


def main():
    client = Client()
    data_dir = "data/instacart"
    output_dir = "data/instacart/dask_data"

    print("Reading raw data...")
    order_products = pd.concat([pd.read_csv(os.path.join(data_dir, "order_products__prior.csv")),
                                pd.read_csv(os.path.join(data_dir, "order_products__train.csv"))])
    orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    orders = dd.from_pandas(orders, npartitions=4)
    departments = pd.read_csv(os.path.join(data_dir, "departments.csv"))
    products = pd.read_csv(os.path.join(data_dir, "products.csv"))

    print("Cleaning up columns. Please be patient, this may take some time...")

    def add_time(df):
        df.reset_index(drop=True)
        df["order_time"] = np.nan
        days_since = df.columns.tolist().index("days_since_prior_order")
        hour_of_day = df.columns.tolist().index("order_hour_of_day")
        order_time = df.columns.tolist().index("order_time")

        df.iloc[0, order_time] = pd.Timestamp('Jan 1, 2015') +  pd.Timedelta(df.iloc[0, hour_of_day], "h")
        for i in range(1, df.shape[0]):
            df.iloc[i, order_time] = df.iloc[i - 1, order_time] \
                + pd.Timedelta(df.iloc[i, days_since], "d") \
                                        + pd.Timedelta(df.iloc[i, hour_of_day], "h")

        to_drop = ["order_number", "order_dow", "order_hour_of_day", "days_since_prior_order", "eval_set"]
        df.drop(to_drop, axis=1, inplace=True)

        return df

    print("Processing orders...")
    orders = orders.groupby("user_id").apply(add_time).compute()
    print("Processing order_products...")
    order_products = order_products.merge(products).merge(departments)
    order_products = order_products.merge(orders[["order_id", "order_time"]])
    # order_products["order_product_id"] = order_products["order_id"].astype(str) + "_" + order_products["add_to_cart_order"].astype(str)
    order_products["order_product_id"] = order_products["order_id"] * 1000 + order_products["add_to_cart_order"]
    order_products = order_products.drop(["product_id", "department_id", "add_to_cart_order"], axis=1)
    try:
        os.mkdir(output_dir)
    except:
        pass

    print("Saving order_products...")
    order_products.to_csv(os.path.join(output_dir, "order_products_dask.csv"), index=False)

    print("Saving orders...")
    orders.to_csv(os.path.join(output_dir, "orders_dask.csv"), index=False)

    client.close()


if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))
