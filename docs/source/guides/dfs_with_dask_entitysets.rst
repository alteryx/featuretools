DFS with Dask Entitysets
========================
Creating a feature matrix for very large datasets can be problematic if the underlying pandas dataframes that make up the entities cannot easily fit in memory. To help get around this issue, Featuretools supports creating entities and entitysets from Dask dataframes. These entitysets can then be passed to ``ft.dfs`` to create a feature matrix, which will be returned as a Dask dataframe. In addition to working on larger than memory datasets, this approach also allows users to take advantage of the parallel processing capabilities offered by Dask.

This guide will provide an overview of how to use Dask entitysets and then generate a feature matrix from this entityset. If you are already familiar with creating a feature matrix from pandas dataframes, this process will seem quite familiar, as there are no differences in the process. There are, however, some limitations when using Dask dataframes, and those limitations are reviewed in more detail below.

Creating Entities and Entitysets
--------------------------------
For this example, we will create a very small pandas dataframe and then convert this into a Dask dataframe for the remainder of the process. Normally, you would just read your data directly into a Dask dataframe without the intermediate step of using pandas.

.. ipython:: python

    import featuretools as ft
    import pandas as pd
    import dask.dataframe as dd
    id = [0, 1, 2, 3, 4]
    values = [12, -35, 14, 103, -51]
    df = pd.DataFrame({"id": id, "values": values})
    dask_df = dd.from_pandas(df, npartitions=2)
    dask_df


Now that we have our dask dataframe, we can start to create our entityset. Because the current implementation does not support variable type inference for Dask entities, we must pass a dictionary of variable types using the ``variable_types`` parameter when calling ``es.entity_from_dataframe()``. Aside from needing to supply the variable types, the rest of the process of creating an entityset is the same as if we were using pandas dataframes.

.. ipython:: python

    es = ft.EntitySet(id="dask_es")
    es = es.entity_from_dataframe(entity_id="dask_entity",
                                  dataframe=dask_df,
                                  index="id",
                                  variable_types={"id": ft.variable_types.Id,
                                                  "values": ft.variable_types.Numeric})
    es


Notice that when we print our entityset, the number of rows for the ``dask_entity`` entity is returned as a Dask delayed object. This is because obtaining the length of a Dask dataframe requires an expensive compute operation to sum up the lengths of all the individual partitions that make up the dataframe and we are skipping this operation to improve processing speed.


Running DFS
-----------
We can pass the entityset we created above to the Featuretools ``dfs`` process in order to create a feature matrix. If the entityset we pass to ``dfs`` is make of Dask entities, the feature matrix we get back will be a Dask dataframe.

.. ipython:: python

    feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="dask_entity",
                                      trans_primitives=["negate"])
    feature_matrix


This feature matrix can be saved to disk or computed and brought into memory, using the appropriate Dask dataframe methods.

.. ipython:: python

    fm_computed = feature_matrix.compute()
    fm_computed


While this is a simple example to illustrate the process of using Dask dataframes with Featuretools, this process will also work with entitysets containing multiple entities, as well as with aggregation primitives.

Limitations
-----------
There are many parts of Featuretools that are difficult to implement in a distributed environment and several primitives that are not well suited to operate on distributed dataframes. As a result, there are some limitations when using Dask dataframes to create entitysets. The most significant limitations are reviewed in more detail in this section.

Supported Primitives
********************
Coming soon...

Entity Limitations
******************
Coming soon...

Entityset Limitations
*********************
Coming soon...

DFS Limitations
***************
Coming soon...

Other Limitations
*****************
Coming soon...
