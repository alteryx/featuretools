import featuretools as ft
import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os
import utils


data_path = "data/instacart/partitioned_data/part_1/"
blocksize = "10MB"
order_products = dd.read_csv(os.path.join(data_path, "order_products__prior.csv"), blocksize=blocksize)
orders = dd.read_csv(os.path.join(data_path, "orders.csv"), blocksize=blocksize)
departments = dd.read_csv(os.path.join(data_path, "departments.csv"), blocksize=blocksize)
products = dd.read_csv(os.path.join(data_path, "products.csv"), blocksize=blocksize)
order_products = order_products.merge(products).merge(departments)

def add_time(df):
    df.reset_index(drop=True)
    df["order_time"] = np.nan
    days_since = df.columns.tolist().index("days_since_prior_order")
    hour_of_day = df.columns.tolist().index("order_hour_of_day")
    order_time = df.columns.tolist().index("order_time")

    df.iloc[0, order_time] = pd.Timestamp('Jan 1, 2015') +  pd.Timedelta(df.iloc[0, hour_of_day], "h")
    for i in range(1, df.shape[0]):
        df.iloc[i, order_time] = df.iloc[i-1, order_time] \
                                    + pd.Timedelta(df.iloc[i, days_since], "d") \
                                    + pd.Timedelta(df.iloc[i, hour_of_day], "h")

    to_drop = ["order_number", "order_dow", "order_hour_of_day", "days_since_prior_order", "eval_set"]
    df.drop(to_drop, axis=1, inplace=True)
    return df

orders = orders.groupby("user_id").apply(add_time, meta={'order_id':'int64', 'user_id':'int64', 'order_time':'object'})
order_products = order_products.merge(orders[["order_id", "order_time"]])
# order_products["order_product_id"] = order_products["order_id"].astype(str) + "_" + order_products["add_to_cart_order"].astype(str)
order_products["order_product_id"] = order_products["order_id"]*1000 + order_products["add_to_cart_order"]
order_products = order_products.drop(["product_id", "department_id", "add_to_cart_order"], axis=1)

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
                                product_name = "Banana",
                                cutoff_time = pd.Timestamp('March 15, 2015'),
                                prediction_window = ft.Timedelta("4 weeks"),
                                training_window = ft.Timedelta("60 days"))
print(label_times.head(5))
label_times["label"].compute().value_counts()

feature_matrix, features = ft.dfs(target_entity="users", 
                                  cutoff_time=label_times,
                                  training_window=ft.Timedelta("60 days"), # same as above
                                  entityset=es,
                                  verbose=True)
# encode categorical values
# fm_encoded, features_encoded = ft.encode_features(feature_matrix,
#                                                   features)

# print("Number of features %s" % len(features_encoded))
# fm_encoded.head(10)
