DFS with Dask Entitysets
========================
Creating a feature matrix from a very large dataset can be problematic if the underlying pandas dataframes that make up the entities cannot easily fit in memory. To help get around this issue, Featuretools supports creating entities and entitysets from Dask dataframes. These entitysets can then be passed to ``ft.dfs`` to create a feature matrix, which will be returned as a Dask dataframe. In addition to working on larger than memory datasets, this approach also allows users to take advantage of the parallel processing capabilities offered by Dask.

This guide will provide an overview of how to create Dask entitysets and then generate a feature matrix from this entityset. If you are already familiar with creating a feature matrix starting from pandas dataframes, this process will seem quite familiar, as there are no differences in the process. There are, however, some limitations when using Dask dataframes, and those limitations are reviewed in more detail below.

Creating Entities and Entitysets
--------------------------------
For this example, we will create a very small pandas dataframe and then convert this into a Dask dataframe to use in the remainder of the process. Normally when using Dask, you would just read your data directly into a Dask dataframe without the intermediate step of using pandas.

.. ipython:: python

    import featuretools as ft
    import pandas as pd
    import dask.dataframe as dd
    id = [0, 1, 2, 3, 4]
    values = [12, -35, 14, 103, -51]
    df = pd.DataFrame({"id": id, "values": values})
    dask_df = dd.from_pandas(df, npartitions=2)
    dask_df


Now that we have our Dask dataframe, we can start to create our entityset. The current implementation does not support variable type inference for Dask entities, so we must pass a dictionary of variable types using the ``variable_types`` parameter when calling ``es.entity_from_dataframe()``. Aside from needing to supply the variable types, the rest of the process of creating an entityset is the same as if we were using pandas dataframes.

.. ipython:: python

    es = ft.EntitySet(id="dask_es")
    es = es.entity_from_dataframe(entity_id="dask_entity",
                                  dataframe=dask_df,
                                  index="id",
                                  variable_types={"id": ft.variable_types.Id,
                                                  "values": ft.variable_types.Numeric})
    es


Notice that when we print our entityset, the number of rows for the ``dask_entity`` entity is returned as a Dask ``Delayed`` object. This is because obtaining the length of a Dask dataframe requires an expensive compute operation to sum up the lengths of all the individual partitions that make up the dataframe and that operation is not performed by default.


Running DFS
-----------
We can pass the entityset we created above to ``featuretools.dfs`` in order to create a feature matrix. If the entityset we pass to ``dfs`` is made of Dask entities, the feature matrix we get back will be a Dask dataframe.

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
There are many parts of Featuretools that are difficult to implement in a distributed environment and several primitives that are not well suited to operate on distributed dataframes. As a result, there are some limitations when creating Dask entitysets and then using these entitysets to generate a feature matrix. The most significant limitations are reviewed in more detail in this section.

Supported Primitives
********************
When creating a feature matrix from a Dask entityset, only certain primitives can be used. Computation of certain features is quite expensive in a distributed environment, and as a result only a subset of Featuretools primitives are currently supported when using a Dask entityset.

To obtain a list of the primitives that can be used with Dask entitysets, you can call ``featuretools.list_primitives()``. This will return a table of all primitives. Any primitive that can be used with a Dask entityset will have a value of ``True`` in the ``dask_compatible`` column.


.. ipython:: python

    primitives_df = ft.list_primitives()
    dask_compatible_df = primitives_df[primitives_df["dask_compatible"] == True]
    dask_compatible_df.head()
    dask_compatible_df.tail()


Entityset Limitations
*********************
When creating a Featuretools ``Entity`` from Dask dataframes, variable type inference is not performed as it is when creating entities from pandas dataframes. This is done to improve speed as sampling the data to infer the variable types would require an expensive compute operation on the underlying Dask dataframe. As a consequence of, this users must define the variable types for each column in the supplied Dataframe. This step is needed so that the deep feature synthesis process can build the proper features based on the column types. A list of available variable types can be obtained by running ``featuretools.variable_types.find_variable_types()``.

By default, Featuretools checks that entities created from pandas dataframes have unique index values. Because performing this same check with Dask would require an expensive compute operation, this check is not performed when creating an entity from a Dask dataframe. When using Dask dataframes, users must ensure that the supplied index values are unique.

Entity Limitations
******************
When creating a Featuretools ``Entityset`` that will be made of Dask entities, there is only one major limitation to be aware of. All of the entities used to create the entityset must be of the same type, either all Dask entities or all pandas entities. Featuretools does not support creating mixed entitysets containing a mix of Dask and pandas entities.

DFS Limitations
***************
There are a few key limitations when generating a feature matrix from a Dask entityset.

If a ``cutoff_time`` parammeter is passed to ``featuretools.dfs()`` it must either be a single cutoff time value, or a pandas dataframe. The current implementation does not support the use of a Dask dataframe for cutoff time values.

Additionally, Featuretools does not currently support the use of the ``approximate`` or ``training_window`` paramaters when working with Dask entitiysets, but should in future releases.

Other Limitations
*****************
In some instances, generating a feature matrix with a large number of features has resulted in memory issues on Dask workers. The underlying reason for this is that the partition size of the feature matrix grows too large for Dask to handle as the number of feature columns grows large. This issue is most prevalent when the feature matrix contains a large number of columns compared to the dataframes that make up the entities. Possible solutions to this problem include reducing the partition size used when creating the entity dataframes or increasing the memory available on Dask workers.

Currently ``featuretools.encode_features()`` does not work with a Dask dataframe as input. This will hopefully be resolved in a future release of Featuretools.
