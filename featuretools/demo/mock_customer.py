from __future__ import division

from builtins import range

import pandas as pd
from numpy import random
from numpy.random import choice

import featuretools as ft
from featuretools.variable_types import Categorical, ZIPCode


def load_mock_customer(n_customers=5, n_products=5, n_sessions=35, n_transactions=500,
                       random_seed=0, return_single_table=False, return_entityset=False):
    """Return dataframes of mock customer data"""

    random.seed(random_seed)
    last_date = pd.to_datetime('12/31/2013')
    first_date = pd.to_datetime('1/1/2008')
    first_bday = pd.to_datetime('1/1/1970')

    join_dates = [random.uniform(0, 1) * (last_date - first_date) + first_date
                  for _ in range(n_customers)]
    birth_dates = [random.uniform(0, 1) * (first_date - first_bday) + first_bday
                   for _ in range(n_customers)]

    customers_df = pd.DataFrame({"customer_id": range(1, n_customers + 1)})
    customers_df["zip_code"] = choice(["60091", "13244"], n_customers,)
    customers_df["join_date"] = pd.Series(join_dates).dt.round('1s')
    customers_df["date_of_birth"] = pd.Series(birth_dates).dt.round('1d')

    products_df = pd.DataFrame({"product_id": pd.Categorical(range(1, n_products + 1))})
    products_df["brand"] = choice(["A", "B", "C"], n_products)

    sessions_df = pd.DataFrame({"session_id": range(1, n_sessions + 1)})
    sessions_df["customer_id"] = choice(customers_df["customer_id"], n_sessions)
    sessions_df["device"] = choice(["desktop", "mobile", "tablet"], n_sessions)

    transactions_df = pd.DataFrame({"transaction_id": range(1, n_transactions + 1)})
    transactions_df["session_id"] = choice(sessions_df["session_id"], n_transactions)
    transactions_df = transactions_df.sort_values("session_id").reset_index(drop=True)
    transactions_df["transaction_time"] = pd.date_range('1/1/2014', periods=n_transactions, freq='65s')  # todo make these less regular
    transactions_df["product_id"] = pd.Categorical(choice(products_df["product_id"], n_transactions))
    transactions_df["amount"] = random.randint(500, 15000, n_transactions) / 100

    # calculate and merge in session start
    # based on the times we came up with for transactions
    session_starts = transactions_df.drop_duplicates("session_id")[["session_id", "transaction_time"]].rename(columns={"transaction_time": "session_start"})
    sessions_df = sessions_df.merge(session_starts)

    if return_single_table:
        return transactions_df.merge(sessions_df).merge(customers_df).merge(products_df).reset_index(drop=True)
    elif return_entityset:
        es = ft.EntitySet(id="transactions")
        es = es.entity_from_dataframe(entity_id="transactions",
                                      dataframe=transactions_df,
                                      index="transaction_id",
                                      time_index="transaction_time",
                                      variable_types={"product_id": Categorical})

        es = es.entity_from_dataframe(entity_id="products",
                                      dataframe=products_df,
                                      index="product_id")

        es = es.entity_from_dataframe(entity_id="sessions",
                                      dataframe=sessions_df,
                                      index="session_id",
                                      time_index="session_start")

        es = es.entity_from_dataframe(entity_id="customers",
                                      dataframe=customers_df,
                                      index="customer_id",
                                      time_index="join_date",
                                      variable_types={"zip_code": ZIPCode})

        rels = [ft.Relationship(es["products"]["product_id"],
                                es["transactions"]["product_id"]),
                ft.Relationship(es["sessions"]["session_id"],
                                es["transactions"]["session_id"]),
                ft.Relationship(es["customers"]["customer_id"],
                                es["sessions"]["customer_id"])]
        es = es.add_relationships(rels)
        es.add_last_time_indexes()
        return es

    return {"customers": customers_df,
            "sessions": sessions_df,
            "transactions": transactions_df,
            "products": products_df}
