import dask.dataframe as dd

import featuretools as ft
from featuretools.entityset import EntitySet, Relationship


def test_create_entity_from_dask_df(es):
    log_dask = dd.from_pandas(es["log"].df, npartitions=2)
    es = es.entity_from_dataframe(
        entity_id="log_dask",
        dataframe=log_dask,
        index="id",
        time_index="datetime",
        variable_types=es["log"].variable_types
    )

    assert es["log"].df.equals(es["log_dask"].df.compute())


def test_build_es_from_scratch():
    es = ft.demo.load_mock_customer(return_entityset=True)
    data = ft.demo.load_mock_customer()

    transactions_df = data["transactions"].merge(data["sessions"]).merge(data["customers"])
    transactions_dd = dd.from_pandas(transactions_df, npartitions=4)
    products_dd = dd.from_pandas(data["products"], npartitions=4)

    dask_es = EntitySet(id="dask_es")
    dask_es.entity_from_dataframe(entity_id="transactions",
                                  dataframe=transactions_dd,
                                  index="transaction_id",
                                  time_index="transaction_time",
                                  variable_types={"product_id": ft.variable_types.Categorical,
                                                  "zip_code": ft.variable_types.ZIPCode})
    dask_es.entity_from_dataframe(entity_id="products",
                                  dataframe=products_dd,
                                  index="product_id")

    new_rel = Relationship(dask_es["products"]["product_id"],
                           dask_es["transactions"]["product_id"])

    dask_es = dask_es.add_relationship(new_rel)

    dask_es = dask_es.normalize_entity(base_entity_id="transactions",
                                       new_entity_id="sessions",
                                       index="session_id",
                                       make_time_index="session_start",
                                       additional_variables=["device",
                                                             "customer_id",
                                                             "zip_code",
                                                             "session_start",
                                                             "join_date"])

    dask_es = dask_es.normalize_entity(base_entity_id="sessions",
                                       new_entity_id="customers",
                                       index="customer_id",
                                       make_time_index="join_date",
                                       additional_variables=["zip_code", "join_date"])

    feature_matrix, feature_defs = ft.dfs(entityset=dask_es,
                                          target_entity="products")
