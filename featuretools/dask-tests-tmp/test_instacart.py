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
    order_products = dd.read_csv([os.path.join(data_path, "order_products_dask.csv")])
    orders = dd.read_csv([os.path.join(data_path, "orders_dask.csv")])
    overall_start = datetime.now()
    start = datetime.now()

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
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))

    start = datetime.now()
    print("Adding relationships...")
    es.add_relationship(ft.Relationship(es["orders"]["order_id"], es["order_products"]["order_id"]))
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))

    print("Normalizing entity...")
    start = datetime.now()
    es.normalize_entity(base_entity_id="orders", new_entity_id="users", index="user_id")
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))

    print("Adding last time indexes...")
    start = datetime.now()
    es.add_last_time_indexes()
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))

    es["order_products"]["department"].interesting_values = ['produce', 'dairy eggs', 'snacks', 'beverages', 'frozen', 'pantry', 'bakery', 'canned goods', 'deli', 'dry goods pasta']
    es["order_products"]["product_name"].interesting_values = ['Banana', 'Bag of Organic Bananas', 'Organic Baby Spinach', 'Organic Strawberries', 'Organic Hass Avocado', 'Organic Avocado', 'Large Lemon', 'Limes', 'Strawberries', 'Organic Whole Milk']
    print(es)

    print("Creating label times...")
    start = datetime.now()
    label_times = utils.make_labels(es=es,
                                    product_name="Banana",
                                    cutoff_time=pd.Timestamp('March 1, 2015'),
                                    prediction_window=ft.Timedelta("4 weeks"),
                                    training_window=ft.Timedelta("60 days"))
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))

    print("Running DFS...")
    start = datetime.now()
    feature_matrix, features = ft.dfs(target_entity="users",
                                      cutoff_time=label_times,
                                      # training_window=ft.Timedelta("60 days"),  # same as above
                                      entityset=es,
                                      trans_primitives=["day", "year", "month", "weekday", "haversine", "num_words", "num_characters"],
                                      agg_primitives=["sum", "std", "max", "skew", "min", "mean", "count", "percent_true"],
                                      verbose=True)

    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))

    print("Computing feature matrix...")
    start = datetime.now()
    if isinstance(feature_matrix, pd.DataFrame):
        feature_matrix = feature_matrix.reset_index()
    feature_matrix = feature_matrix.merge(label_times)
    if isinstance(feature_matrix, dd.core.DataFrame):
        feature_matrix = feature_matrix.compute()
    feature_matrix.to_csv('fm_dask.csv', index=False)
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))
    print("Shape: {}".format(feature_matrix.shape))
    print("Memory: {} MB".format(feature_matrix.memory_usage().sum() / 1000000))
    return
    print("Encoding categorical features...")
    start = datetime.now()
    # encode categorical values
    fm_encoded, features_encoded = ft.encode_features(feature_matrix,
                                                      features,
                                                      verbose=True)
    print("Number of features %s" % len(features_encoded))
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))

    print("Training model...")
    X = fm_encoded.drop(["user_id", "time"], axis=1)
    X = X.fillna(0)
    y = X.pop("label")

    clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)

    scores = cross_val_score(estimator=clf, X=X, y=y, cv=3,
                             scoring="roc_auc", verbose=True)

    print("AUC %.2f +/- %.2f" % (scores.mean(), scores.std()))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf.fit(X_train, y_train)

    print("Making predictions...")
    print(clf.score(X, y))
    y_pred = clf.predict(X_test)

    print("Making predictions...")
    from sklearn import metrics
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))

    overall_end = datetime.now()
    overall_elapsed = (overall_end - overall_start).total_seconds()
    print("Total elapsed time: {} sec".format(overall_elapsed))

if __name__ == "__main__":
    run_test()
