# flake8: noqa
import math
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import featuretools as ft


def main():
    data_dir = "data/instacart"
    output_dir = "data/instacart/dask_data"
    num_rows_per_file = 1000000

    print("Reading raw data...")
    order_products = pd.concat([pd.read_csv(os.path.join(data_dir,"order_products__prior.csv")),
                                pd.read_csv(os.path.join(data_dir, "order_products__train.csv"))])
    orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    departments = pd.read_csv(os.path.join(data_dir, "departments.csv"))
    products = pd.read_csv(os.path.join(data_dir, "products.csv"))

    print("Cleaning up columns. Please be patient, this may take some time...")
    order_products = order_products.merge(products).merge(departments)

    def add_time(df):
        df.reset_index(drop=True)
        df["days_since_first"] = df["days_since_prior_order"].cumsum().fillna(0)
        df["order_time"] = pd.Timestamp('Jan 1, 2015') + pd.to_timedelta(df["days_since_first"], "d") + pd.to_timedelta(df['order_hour_of_day'], "h")
        to_drop = ["order_number", "order_dow", "order_hour_of_day", "days_since_prior_order", "eval_set", "days_since_first"]
        df.drop(to_drop, axis=1, inplace=True)
        return df

    tqdm.pandas()
    orders = orders.groupby("user_id").progress_apply(add_time)
    order_products = order_products.merge(orders[["order_id", "order_time"]])
    # order_products["order_product_id"] = order_products["order_id"].astype(str) + "_" + order_products["add_to_cart_order"].astype(str)
    order_products["order_product_id"] = order_products["order_id"] * 1000 + order_products["add_to_cart_order"]
    order_products.drop(["product_id", "department_id", "add_to_cart_order"], axis=1, inplace=True)

    try:
        os.mkdir(output_dir)
    except:
        pass

    print("Saving order_products...")
    for i in tqdm(range(math.ceil(len(order_products) / num_rows_per_file))):
        order_products.iloc[i * num_rows_per_file:num_rows_per_file * (i + 1)].to_csv(os.path.join(output_dir, "order_products_{}.csv".format(i)), index=False)

    print("Saving orders...")
    for i in tqdm(range(math.ceil(len(orders) / num_rows_per_file))):
        orders.iloc[i * num_rows_per_file:num_rows_per_file * (i + 1)].to_csv(os.path.join(output_dir, "orders_{}.csv".format(i)), index=False)

if __name__ == "__main__":
    main()
