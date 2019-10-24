import dask.dataframe as dd


def test_create_entity_from_dask_df(es):
    log_dask = dd.from_pandas(es['log'].df, npartitions=2)
    print(es['log'].df)
    print(es['log'].df.columns)
    es = es.entity_from_dataframe(
        entity_id="log_dask",
        dataframe=log_dask,
        index="id",
        time_index="datetime"
    )

    print(es)
