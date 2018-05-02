.. _feature-engineering-language:
.. currentmodule:: featuretools

Defining Features
==================

While Deep Feature Synthesis can automatically combine primitives to create feature definitions, users of Featuretools can also manually define features. User manually define features for many reasons.



"Seed" DFS with domain specific features
****************************************

.. ipython:: python
    :suppress:

    import featuretools as ft
    import pandas as pd

    customers_df = pd.DataFrame({"id": [1, 2],
                             "zip_code": ["60091", "02139"]})

    sessions_df = pd.DataFrame({"id": [1, 2, 3],
                                "customer_id": [1, 2, 1],
                                "session_start": [pd.Timestamp("2017-02-22"),
                                                  pd.Timestamp("2016-12-22"),
                                                  pd.Timestamp("2017-01-12")],
                                "session_duration":[12.3, 33, 43]})

    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "session_id": [1, 2, 1, 3, 4, 5],
                                    "amount": [100.40, 20.63, 33.32, 13.12, 67.22, 1.00],
                                    "transaction_time": pd.date_range(start="10:00", periods=6, freq="10s")})

    entities = {
           "customers" : (customers_df, "id"), # time index is optional
           "sessions" : (sessions_df, "id", "session_start"),
           "transactions" : (transactions_df, "id", "transaction_time")
    }
    relationships = [("sessions", "id", "transactions", "session_id"),
                     ("customers", "id", "sessions", "customer_id")]
    es = ft.EntitySet("session_data", entities, relationships)

.. ipython:: python

    expensive_purchase = Feature(es["transactions"]["amount"]) > 20
    expensive_purchase = expensive_purchase.rename("expensive_purchase")

    features = ft.dfs(entityset=es,
                      target_entity="customers",
                      agg_primitives=["percent_true"],
                      seed_features=[expensive_purchase],
                      features_only=True)
    features


Use high level primitives
*************************

.. code-block:: python

    # average time between sessions per customer
    AvgTimeBetween(es["sessions"]["id"], es["customers"])


More concise to write than SQL
******************************

.. code-block:: python

    # SQL                                                         # Featuretools
    SELECT c.id,                                                  Count(es["events"]["id"],
           Coalesce(a.num_events, 0)                                    es["customers"],
    FROM   customers c                                                  use_previous="10 days")
           left outer join (SELECT customer_id,
                                   Count(*) AS num_events
                            FROM   EVENTS
                            WHERE  (timestamp >
                                    current_date - interval '10' day)
                            GROUP  BY customer_id) AS a
                        ON c.id = a.customer_id;





