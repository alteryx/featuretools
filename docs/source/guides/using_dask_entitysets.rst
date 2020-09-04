Using Dask EntitySets (BETA)
============================
.. note::
    Support for Dask EntitySets is still in Beta. While the key functionality has been implemented, development is ongoing to add the remaining functionality.

    All planned improvements to the Featuretools/Dask integration are `documented on Github <https://github.com/FeatureLabs/featuretools/issues?q=is%3Aopen+is%3Aissue+label%3ADask>`_. If you see an open issue that is important for your application, please let us know by upvoting or commenting on the issue. If you encounter any errors using Dask entities, or find missing functionality that does not yet have an open issue, please create a `new issue on Github <https://github.com/FeatureLabs/featuretools/issues>`_.

Creating a feature matrix from a very large dataset can be problematic if the underlying pandas dataframes that make up the entities cannot easily fit in memory. To help get around this issue, Featuretools supports creating ``Entity`` and ``EntitySet`` objects from Dask dataframes. A Dask ``EntitySet`` can then be passed to ``featuretools.dfs`` or ``featuretools.calculate_feature_matrix`` to create a feature matrix, which will be returned as a Dask dataframe. In addition to working on larger than memory datasets, this approach also allows users to take advantage of the parallel and distributed processing capabilities offered by Dask.

This guide will provide an overview of how to create a Dask ``EntitySet`` and then generate a feature matrix from it. If you are already familiar with creating a feature matrix starting from pandas dataframes, this process will seem quite familiar, as there are no differences in the process. There are, however, some limitations when using Dask dataframes, and those limitations are reviewed in more detail below.

Creating Entities and EntitySets
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


Now that we have our Dask dataframe, we can start to create the ``EntitySet``. The current implementation does not support variable type inference for Dask entities, so we must pass a dictionary of variable types using the ``variable_types`` parameter when calling ``es.entity_from_dataframe()``. Aside from needing to supply the variable types, the rest of the process of creating an ``EntitySet`` is the same as if we were using pandas dataframes.

.. ipython:: python

    es = ft.EntitySet(id="dask_es")
    es = es.entity_from_dataframe(entity_id="dask_entity",
                                  dataframe=dask_df,
                                  index="id",
                                  variable_types={"id": ft.variable_types.Id,
                                                  "values": ft.variable_types.Numeric})
    es


Notice that when we print our ``EntitySet``, the number of rows for the ``dask_entity`` entity is returned as a Dask ``Delayed`` object. This is because obtaining the length of a Dask dataframe may require an expensive compute operation to sum up the lengths of all the individual partitions that make up the dataframe and that operation is not performed by default.


Running DFS
-----------
We can pass the ``EntitySet`` we created above to ``featuretools.dfs`` in order to create a feature matrix. If the ``EntitySet`` we pass to ``dfs`` is made of Dask entities, the feature matrix we get back will be a Dask dataframe.

.. ipython:: python

    feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="dask_entity",
                                      trans_primitives=["negate"])
    feature_matrix


This feature matrix can be saved to disk or computed and brought into memory, using the appropriate Dask dataframe methods.

.. ipython:: python

    fm_computed = feature_matrix.compute()
    fm_computed


While this is a simple example to illustrate the process of using Dask dataframes with Featuretools, this process will also work with an ``EntitySet`` containing multiple entities, as well as with aggregation primitives.

Limitations
-----------
The key functionality of Featuretools is available for use with a Dask ``EntitySet``, and work is ongoing to add the remaining functionality that is available when using a pandas ``EntitySet``. There are, however, some limitations to be aware of when creating a Dask ``Entityset`` and then using it to generate a feature matrix. The most significant limitations are reviewed in more detail in this section.

.. note::
    If the limitations of using a Dask ``EntitySet`` are problematic for your problem, you may still be able to compute a larger-than-memory feature matrix by partitioning your data as described in :doc:`performance`.

Supported Primitives
********************
When creating a feature matrix from a Dask ``EntitySet``, only certain primitives can be used. Primitives that rely on the order of the entire dataframe or require an entire column for computation are currently not supported when using a Dask ``EntitySet``. Multivariable and time-dependent aggregation primitives also are not currently supported.

To obtain a list of the primitives that can be used with a Dask ``EntitySet``, you can call ``featuretools.list_primitives()``. This will return a table of all primitives. Any primitive that can be used with a Dask ``EntitySet`` will have a value of ``True`` in the ``dask_compatible`` column.


.. ipython:: python

    primitives_df = ft.list_primitives()
    dask_compatible_df = primitives_df[primitives_df["dask_compatible"] == True]
    dask_compatible_df.head()
    dask_compatible_df.tail()

Primitive Limitations
*********************
At this time, custom primitives created with ``featuretools.primitives.make_trans_primitive()`` or ``featuretools.primitives.make_agg_primitive()`` cannot be used for running deep feature synthesis on a Dask ``EntitySet``. While it is possible to create custom primitives for use with a Dask ``EntitySet`` by extending the proper primitive class, there are several potential problems in doing so, and those issues are beyond the scope of this guide.

Entity Limitations
******************
When creating a Featuretools ``Entity`` from Dask dataframes, variable type inference is not performed as it is when creating entities from pandas dataframes. This is done to improve speed as sampling the data to infer the variable types would require an expensive compute operation on the underlying Dask dataframe. As a consequence, users must define the variable types for each column in the supplied Dataframe. This step is needed so that the deep feature synthesis process can build the proper features based on the column types. A list of available variable types can be obtained by running ``featuretools.variable_types.find_variable_types()``.

By default, Featuretools checks that entities created from pandas dataframes have unique index values. Because performing this same check with Dask would require an expensive compute operation, this check is not performed when creating an entity from a Dask dataframe. When using Dask dataframes, users must ensure that the supplied index values are unique.

When an ``Entity`` is created from a pandas dataframe, the ordering of the underlying dataframe rows is maintained. For a Dask ``Entity``, the ordering of the dataframe rows is not guaranteed, and Featuretools does not attempt to maintain row order in a Dask ``Entity``. If ordering is important, close attention must be paid to any output to avoid issues.

The ``Entity.add_interesting_values()`` method is not supported when using a Dask ``Entity``.  If needed, users can manually set ``interesting_values`` on entities by assigning them directly with syntax similar to this: ``es["entity_name"]["variable_name"].interesting_values = ["Value 1", "Value 2"]``.

EntitySet Limitations
*********************
When creating a Featuretools ``EntitySet`` that will be made of Dask entities, all of the entities used to create the ``EntitySet`` must be of the same type, either all Dask entities or all pandas entities. Featuretools does not support creating an ``EntitySet`` containing a mix of Dask and pandas entities.

Additionally, the ``EntitySet.add_interesting_values()`` method is not supported when using a Dask ``EntitySet``. Users can manually set ``interesting_values`` on entities, as described above.

DFS Limitations
***************
There are a few key limitations when generating a feature matrix from a Dask ``EntitySet``.

If a ``cutoff_time`` parameter is passed to ``featuretools.dfs()`` it should be a single cutoff time value, or a pandas dataframe. The current implementation will still work if a Dask dataframe is supplied for cutoff times, but a ``.compute()`` call will be made on the dataframe to convert it into a pandas dataframe. This conversion will result in a warning, and the process could take a considerable amount of time to complete depending on the size of the supplied dataframe.

Additionally, Featuretools does not currently support the use of the ``approximate`` or ``training_window`` parameters when working with Dask entitiysets, but should in future releases.

Finally, if the output feature matrix contains a boolean column with ``NaN`` values included, the column type may have a different datatype than the same feature matrix generated from a pandas ``EntitySet``.  If feature matrix column data types are critical, the feature matrix should be inspected to make sure the types are of the proper types, and recast as necessary.

Other Limitations
*****************
In some instances, generating a feature matrix with a large number of features has resulted in memory issues on Dask workers. The underlying reason for this is that the partition size of the feature matrix grows too large for Dask to handle as the number of feature columns grows large. This issue is most prevalent when the feature matrix contains a large number of columns compared to the dataframes that make up the entities. Possible solutions to this problem include reducing the partition size used when creating the entity dataframes or increasing the memory available on Dask workers.

Currently ``featuretools.encode_features()`` does not work with a Dask dataframe as input. This will hopefully be resolved in a future release of Featuretools.

The utility function ``featuretools.make_temporal_cutoffs()`` will not work properly with Dask inputs for ``instance_ids`` or ``cutoffs``. However, as noted above, if a ``cutoff_time`` dataframe is supplied to ``dfs``, the supplied dataframe should be a pandas dataframe, and this can be generated by supplying pandas inputs to ``make_temporal_cutoffs()``.

The use of ``featuretools.remove_low_information_features()`` cannot currently be used with a Dask feature matrix.

When manually defining a ``Feature``, the ``use_previous`` parameter cannot be used if this feature will be applied to calculate a feature matrix from a Dask ``EntitySet``.
