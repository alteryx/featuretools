# flake8: noqa
import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client
from dask_ml.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import utils

import featuretools as ft

if __name__ == "__main__":
    # client = Client(processes=False)
    client = Client()

    # data_path = "data/instacart/partitioned_data/part_1/"
    # blocksize = "10MB"
    # order_products = dd.read_csv(os.path.join(data_path, "order_products__prior.csv"), blocksize=blocksize)
    # orders = dd.read_csv(os.path.join(data_path, "orders.csv"), blocksize=blocksize)
    # departments = dd.read_csv(os.path.join(data_path, "departments.csv"), blocksize=blocksize)
    # products = dd.read_csv(os.path.join(data_path, "products.csv"), blocksize=blocksize)
    # order_products = order_products.merge(products).merge(departments)

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

    # orders = orders.groupby("user_id").apply(add_time, meta={'order_id':'int64', 'user_id':'int64', 'order_time':'object'})
    # order_products = order_products.merge(orders[["order_id", "order_time"]])
    # order_products["order_product_id"] = order_products["order_id"]*1000 + order_products["add_to_cart_order"]
    # order_products = order_products.drop(["product_id", "department_id", "add_to_cart_order"], axis=1)

    # from dask.distributed import wait
    # order_products = client.persist(order_products)
    # wait(order_products)

    # orders = client.persist(orders)
    # wait(orders)

    # es = ft.EntitySet("instacart")
    # es.entity_from_dataframe(entity_id="order_products",
    #                         dataframe=order_products,
    #                         index="order_product_id",
    #                         variable_types={"aisle_id": ft.variable_types.Categorical, "reordered": ft.variable_types.Boolean},
    #                         time_index="order_time")

    # es.entity_from_dataframe(entity_id="orders",
    #                         dataframe=orders,
    #                         index="order_id",
    #                         time_index="order_time")

    # es.add_relationship(ft.Relationship(es["orders"]["order_id"], es["order_products"]["order_id"]))
    # es.normalize_entity(base_entity_id="orders", new_entity_id="users", index="user_id")
    # es.add_last_time_indexes()
    # es["order_products"]["department"].interesting_values = ['produce', 'dairy eggs', 'snacks', 'beverages', 'frozen', 'pantry', 'bakery', 'canned goods', 'deli', 'dry goods pasta']
    # es["order_products"]["product_name"].interesting_values = ['Banana', 'Bag of Organic Bananas', 'Organic Baby Spinach', 'Organic Strawberries', 'Organic Hass Avocado', 'Organic Avocado', 'Large Lemon', 'Limes', 'Strawberries', 'Organic Whole Milk']
    # print(es)

    # label_times = utils.make_labels(es=es,
    #                                 product_name = "Banana",
    #                                 cutoff_time = pd.Timestamp('March 15, 2015'),
    #                                 prediction_window = ft.Timedelta("4 weeks"),
    #                                 training_window = ft.Timedelta("60 days"))
    # print(label_times.head(5))
    # label_times["label"].compute().value_counts()

    # feature_matrix, features = ft.dfs(target_entity="users",
    #                                 cutoff_time=label_times,
    #                                 # training_window=ft.Timedelta("60 days"), # same as above
    #                                 agg_primitives=['count'],
    #                                 trans_primitives=['weekday'],
    #                                 entityset=es,
    #                                 verbose=True)
    # # encode categorical values
    # fm_encoded, features_encoded = ft.encode_features(feature_matrix,
    #                                                 features,
    #                                                 verbose=True)

    # print("Number of features %s" % len(features_encoded))
    # fm_encoded.head(10)

    # X = utils.merge_features_labels(fm_encoded, label_times)
    # X = X.drop(["user_id", "time"], axis=1)
    # X = X.fillna(0)
    # y = X.pop("label")

    # import joblib
    # clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)

    # ### CONTINUE HERE
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # clf.fit(X_train, y_train)
    # clf.score(X, y)
    # y_pred = clf.predict(X_test)
    # from sklearn import metrics

    # print(metrics.accuracy_score(y_test, y_pred))
    # print(metrics.confusion_matrix(y_test, y_pred))

    data_path = os.path.join("data", "instacart", "dask_data")
    # dirnames = [os.path.join(path, d) for d in os.listdir(path)]
    # order_products_files = [path + "/order_products__prior.csv" for path in dirnames][:2]
    # orders_files = [path + "/orders.csv" for path in dirnames][:2]
    # departments_files = [path + "/departments.csv" for path in dirnames][:2]
    # products_files = [path + "/products.csv" for path in dirnames][:2]

    # blocksize = "10MB"
    order_products = dd.read_csv(os.path.join(data_path, "order_products_*.csv"))
    orders = dd.read_csv(os.path.join(data_path, "orders_*.csv"))
    departments = dd.read_csv(os.path.join(data_path, "departments_*.csv"))
    products = dd.read_csv(os.path.join(data_path, "products_*.csv"))

    order_products = order_products.merge(products).merge(departments)

    orders = orders.groupby("user_id").apply(add_time, meta={'order_id': 'int64', 'user_id': 'int64', 'order_time': 'object'})
    order_products = order_products.merge(orders[["order_id", "order_time"]])
    order_products["order_product_id"] = order_products["order_id"] * 1000 + order_products["add_to_cart_order"]
    order_products = order_products.drop(["product_id", "department_id", "add_to_cart_order"], axis=1)

    # breakpoint()
    # order_products = client.persist(order_products)
    # wait(order_products)
    # orders = client.persist(orders)
    # wait(orders)

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

    es.add_relationship(ft.Relationship(es["orders"]["order_id"], es["order_products"]["order_id"]))
    es.normalize_entity(base_entity_id="orders", new_entity_id="users", index="user_id")
    es.add_last_time_indexes()
    es["order_products"]["department"].interesting_values = ['produce', 'dairy eggs', 'snacks', 'beverages', 'frozen', 'pantry', 'bakery', 'canned goods', 'deli', 'dry goods pasta']
    es["order_products"]["product_name"].interesting_values = ['Banana', 'Bag of Organic Bananas', 'Organic Baby Spinach', 'Organic Strawberries', 'Organic Hass Avocado', 'Organic Avocado', 'Large Lemon', 'Limes', 'Strawberries', 'Organic Whole Milk']
    print(es)

    label_times = utils.make_labels(es=es,
                                    product_name="Banana",
                                    cutoff_time=pd.Timestamp('March 15, 2015'),
                                    prediction_window=ft.Timedelta("4 weeks"),
                                    training_window=ft.Timedelta("60 days"))
    print(label_times.head(5))
    label_times["label"].compute().value_counts()

    feature_matrix, features = ft.dfs(target_entity="users",
                                      cutoff_time=label_times,
                                      # training_window=ft.Timedelta("60 days"), # same as above
                                      agg_primitives=['count'],
                                      trans_primitives=['weekday'],
                                      entityset=es,
                                      verbose=True)
    # encode categorical values
    fm_encoded, features_encoded = ft.encode_features(feature_matrix,
                                                      features,
                                                      verbose=True)

    print("Number of features %s" % len(features_encoded))
    fm_encoded.head(10)

    X = utils.merge_features_labels(fm_encoded, label_times)
    X = X.drop(["user_id", "time"], axis=1)
    X = X.fillna(0)
    y = X.pop("label")

    import joblib
    clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)

    ### CONTINUE HERE
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf.fit(X_train, y_train)
    clf.score(X, y)
    y_pred = clf.predict(X_test)
    from sklearn import metrics

    metrics.accuracy_score(y_test, y_pred)
    metrics.confusion_matrix(y_test, y_pred)
