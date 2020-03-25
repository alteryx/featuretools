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
    client = Client()
    breakpoint()
    data_path = os.path.join("data", "instacart", "dask_data")

    order_products = dd.read_csv(os.path.join(data_path, "order_products_*.csv"))
    orders = dd.read_csv(os.path.join(data_path, "orders_*.csv"))

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf.fit(X_train, y_train)
    clf.score(X, y)
    y_pred = clf.predict(X_test)
    from sklearn import metrics

    metrics.accuracy_score(y_test, y_pred)
    metrics.confusion_matrix(y_test, y_pred)
