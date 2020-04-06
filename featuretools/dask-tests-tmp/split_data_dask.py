# flake8: noqa
import math
import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
from tqdm import tqdm

import featuretools as ft
from dask.distributed import Client


def main():
    data_dir = "data/instacart"
    output_dir = "data/instacart/dask_data"
    num_rows_per_file = 1000000
    blocksize = '50Mb'

    print("Reading raw data...")
    order_products = dd.concat([dd.read_csv(os.path.join(data_dir, "order_products__prior.csv"), blocksize=blocksize),
                                dd.read_csv(os.path.join(data_dir, "order_products__train.csv"), blocksize=blocksize)]).persist()
    orders = dd.read_csv(os.path.join(data_dir, "orders.csv"), blocksize=blocksize).persist()
    departments = dd.read_csv(os.path.join(data_dir, "departments.csv"), blocksize=blocksize).persist()
    products = dd.read_csv(os.path.join(data_dir, "products.csv"), blocksize=blocksize).persist()

    print("Cleaning up columns. Please be patient, this may take some time...")
    order_products = order_products.merge(products).merge(departments)

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

    # tqdm.pandas()
    # orders = orders.groupby("user_id").progress_apply(add_time)
    orders = orders.groupby("user_id").apply(add_time)
    order_products = order_products.merge(orders[["order_id", "order_time"]])
    # order_products["order_product_id"] = order_products["order_id"].astype(str) + "_" + order_products["add_to_cart_order"].astype(str)
    order_products["order_product_id"] = order_products["order_id"] * 1000 + order_products["add_to_cart_order"]
    order_products = order_products.drop(["product_id", "department_id", "add_to_cart_order"], axis=1)
    breakpoint()
    try:
        os.mkdir(output_dir)
    except:
        pass

    print("Saving order_products...")
    order_products.to_csv(os.path.join(output_dir, "order_products_*.csv"), index=False)
    # for i in tqdm(range(math.ceil(len(order_products) / num_rows_per_file))):
    #     order_products.iloc[i * num_rows_per_file:num_rows_per_file * (i + 1)].to_csv(os.path.join(output_dir, "order_products_{}.csv".format(i)), index=False)

    print("Saving orders...")
    orders.to_csv(os.path.join(output_dir, "orders_*.csv"), index=False)
    # for i in tqdm(range(math.ceil(len(orders) / num_rows_per_file))):
    #     orders.iloc[i * num_rows_per_file:num_rows_per_file * (i + 1)].to_csv(os.path.join(output_dir, "orders_{}.csv".format(i)), index=False)

if __name__ == "__main__":
    client = Client()
    main()
