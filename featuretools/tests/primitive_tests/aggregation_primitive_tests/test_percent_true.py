import pandas as pd
from woodwork.logical_types import BooleanNullable

import featuretools as ft


def test_percent_true_default_value_with_dfs():
    es = ft.EntitySet(id="customer_data")

    customers_df = pd.DataFrame(data={"customer_id": [1, 2]})
    transactions_df = pd.DataFrame(
        data={"tx_id": [1], "customer_id": [1], "is_foo": [True]},
    )

    es.add_dataframe(
        dataframe_name="customers_df",
        dataframe=customers_df,
        index="customer_id",
    )
    es.add_dataframe(
        dataframe_name="transactions_df",
        dataframe=transactions_df,
        index="tx_id",
        logical_types={"is_foo": BooleanNullable},
    )

    es = es.add_relationship(
        "customers_df",
        "customer_id",
        "transactions_df",
        "customer_id",
    )

    feature_matrix, _ = ft.dfs(
        entityset=es,
        target_dataframe_name="customers_df",
        agg_primitives=["percent_true"],
    )

    assert pd.isna(feature_matrix["PERCENT_TRUE(transactions_df.is_foo)"][2])
