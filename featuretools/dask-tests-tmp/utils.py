# flake8: noqa
import os

import dask.dataframe as dd
import numpy as np
import pandas as pd

import featuretools as ft


def load_entityset(data_dir):
    order_products = pd.read_csv(os.path.join(data_dir, "order_products__prior.csv"))
    orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    departments = pd.read_csv(os.path.join(data_dir, "departments.csv"))
    products = pd.read_csv(os.path.join(data_dir, "products.csv"))

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

    orders = orders.groupby("user_id").apply(add_time)
    order_products = order_products.merge(orders[["order_id", "order_time"]])
    order_products["order_product_id"] = order_products["order_id"].astype(str) + "_" + order_products["add_to_cart_order"].astype(str)
    order_products.drop(["product_id", "department_id", "add_to_cart_order"], axis=1, inplace=True)
    es = ft.EntitySet("instacart")

    es.entity_from_dataframe(entity_id="order_products",
                             dataframe=order_products,
                             index="order_product_id",
                             variable_types={"aisle_id": ft.variable_types.Categorical, "reordered": ft.variable_types.Boolean},
                             time_index="order_time")

    es.entity_from_dataframe(entity_id="orders",
                             dataframe=orders,
                             index="order_id",
                             time_index="order_time")

    # es.entity_from_dataframe(entity_id="products",
    #                          dataframe=products,
    #                          index="product_id")

    es.add_relationship(ft.Relationship(es["orders"]["order_id"], es["order_products"]["order_id"]))
    # es.add_relationship(ft.Relationship(es["products"]["product_id"], es["order_products"]["order_id"]))

    es.normalize_entity(base_entity_id="orders", new_entity_id="users", index="user_id")
    es.add_last_time_indexes()

    # order_products["department"].value_counts().head(10).index.values.tolist()
    es["order_products"]["department"].interesting_values = ['produce', 'dairy eggs', 'snacks', 'beverages', 'frozen', 'pantry', 'bakery', 'canned goods', 'deli', 'dry goods pasta']
    es["order_products"]["product_name"].interesting_values = ['Banana', 'Bag of Organic Bananas', 'Organic Baby Spinach', 'Organic Strawberries', 'Organic Hass Avocado', 'Organic Avocado', 'Large Lemon', 'Limes', 'Strawberries', 'Organic Whole Milk']

    return es


def make_labels(es, training_window, cutoff_time,
                product_name, prediction_window):

    prediction_window_end = cutoff_time + prediction_window
    t_start = cutoff_time - training_window

    orders = es["orders"].df
    ops = es["order_products"].df

    # if isinstance(orders, dd.core.DataFrame):
    #     orders = orders.compute()

    # if isinstance(ops, dd.core.DataFrame):
    #     ops = ops.compute()

    training_data = ops[(ops["order_time"] <= cutoff_time) & (ops["order_time"] > t_start)]
    prediction_data = ops[(ops["order_time"] > cutoff_time) & (ops["order_time"] < prediction_window_end)]
    users_in_training = training_data.merge(orders)["user_id"].unique()
    # users_in_training = training_data.merge(orders)
    # users_in_training["in_training"] = True
    # users_in_training = users_in_training[["user_id", "in_training"]]

    valid_pred_data = prediction_data.merge(orders)
    # valid_pred_data = valid_pred_data.merge(users_in_training, how="left", on="user_id")
    # valid_pred_data = valid_pred_data[valid_pred_data["in_training"] == True].drop(columns=["in_training"])

    if isinstance(users_in_training, dd.core.Series):
        users_in_training = users_in_training.compute()

    valid_pred_data = valid_pred_data[valid_pred_data["user_id"].isin(users_in_training)]

    def bought_product(df):
        return (df["product_name"] == product_name).any()

    labels = valid_pred_data.groupby("user_id").apply(bought_product).reset_index()
    labels["cutoff_time"] = cutoff_time
    #  rename and reorder
    labels.columns = ["user_id", "label", "time", ]
    labels = labels[["user_id", "time", "label"]]
    if isinstance(labels, dd.core.DataFrame):
        labels = labels.compute()
    return labels


def dask_make_labels(es, **kwargs):
    label_times = make_labels(es, **kwargs)
    return label_times, es


def calculate_feature_matrix(label_times, features):
    label_times, es = label_times
    fm = ft.calculate_feature_matrix(features,
                                     entityset=es,
                                     cutoff_time=label_times,
                                     cutoff_time_in_index=True,
                                     verbose=False)

    X = merge_features_labels(fm, label_times)
    return X


def merge_features_labels(fm, labels):
    return fm.reset_index().merge(labels)


def feature_importances(model, features, n=10):
    importances = model.feature_importances_
    zipped = sorted(zip(features, importances), key=lambda x: -x[1])
    for i, f in enumerate(zipped[:n]):
        print("%d: Feature: %s, %.3f" % (i + 1, f[0].get_name(), f[1]))

    return [f[0] for f in zipped[:n]]
