Using Koalas EntitySets (BETA)
==============================
.. note::
    Support for Koalas EntitySets is still in Beta. While the key functionality has been implemented, development is ongoing to add the remaining functionality.

    All planned improvements to the Featuretools/Koalas integration are `documented on Github <https://github.com/FeatureLabs/featuretools/issues?q=is%3Aopen+is%3Aissue+label%3AKoalas>`_. If you see an open issue that is important for your application, please let us know by upvoting or commenting on the issue. If you encounter any errors using Koalas entities, or find missing functionality that does not yet have an open issue, please create a `new issue on Github <https://github.com/FeatureLabs/featuretools/issues>`_.

Creating a feature matrix from a very large dataset can be problematic if the underlying pandas dataframes that make up the entities cannot easily fit in memory. To help get around this issue, Featuretools supports creating ``Entity`` and ``EntitySet`` objects from Koalas dataframes. A Koalas ``EntitySet`` can then be passed to ``featuretools.dfs`` or ``featuretools.calculate_feature_matrix`` to create a feature matrix, which will be returned as a Koalas dataframe. In addition to working on larger than memory datasets, this approach also allows users to take advantage of the parallel and distributed processing capabilities offered by Koalas and Spark.

This guide will provide an overview of how to create a Koalas ``EntitySet`` and then generate a feature matrix from it. If you are already familiar with creating a feature matrix starting from pandas dataframes, this process will seem quite familiar, as there are no differences in the process. There are, however, some limitations when using Koalas dataframes, and those limitations are reviewed in more detail below.

Creating Entities and EntitySets
--------------------------------
Koalas ``EntitySets`` require Koalas and PySpark. Both can be installed directly with ``pip install featuretools[koalas]``. Java is also required for PySpark and may need to be installed, see `the Spark documentation <https://spark.apache.org/docs/latest/index.html>`_ for more details. We will create a very small Koalas dataframe for this example. Koalas dataframes can also be created from pandas dataframes, Spark dataframes, or read in directly from a file.

.. ipython:: python

    import featuretools as ft
    import databricks.koalas as ks
    id = [0, 1, 2, 3, 4]
    values = [12, -35, 14, 103, -51]
    koalas_df = ks.DataFrame({"id": id, "values": values})
    koalas_df


Now that we have our Koalas dataframe, we can start to create the ``EntitySet``. The current implementation does not support variable type inference for Koalas entities, so we must pass a dictionary of variable types using the ``variable_types`` parameter when calling ``es.entity_from_dataframe()``. Aside from needing to supply the variable types, the rest of the process of creating an ``EntitySet`` is the same as if we were using pandas dataframes.

.. ipython:: python

    es = ft.EntitySet(id="koalas_es")
    es = es.entity_from_dataframe(entity_id="koalas_entity",
                                  dataframe=koalas_df,
                                  index="id",
                                  variable_types={"id": ft.variable_types.Id,
                                                  "values": ft.variable_types.Numeric})
    es



Running DFS
-----------
We can pass the ``EntitySet`` we created above to ``featuretools.dfs`` in order to create a feature matrix. If the ``EntitySet`` we pass to ``dfs`` is made of Koalas entities, the feature matrix we get back will be a Koalas dataframe.

.. ipython:: python

    feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="koalas_entity",
                                      trans_primitives=["negate"])
    feature_matrix


This feature matrix can be saved to disk or converted to a pandas dataframe and brought into memory, using the appropriate Koalas dataframe methods.

While this is a simple example to illustrate the process of using Koalas dataframes with Featuretools, this process will also work with an ``EntitySet`` containing multiple entities, as well as with aggregation primitives.

Limitations
-----------
The key functionality of Featuretools is available for use with a Koalas ``EntitySet``, and work is ongoing to add the remaining functionality that is available when using a pandas ``EntitySet``. There are, however, some limitations to be aware of when creating a Koalas ``Entityset`` and then using it to generate a feature matrix. The most significant limitations are reviewed in more detail in this section.

.. note::
    If the limitations of using a Koalas ``EntitySet`` are problematic for your problem, you may still be able to compute a larger-than-memory feature matrix by partitioning your data as described in :doc:`performance`.

Supported Primitives
********************
When creating a feature matrix from a Koalas ``EntitySet``, only certain primitives can be used. Primitives that rely on the order of the entire dataframe or require an entire column for computation are currently not supported when using a Koalas ``EntitySet``. Multivariable and time-dependent aggregation primitives also are not currently supported.

To obtain a list of the primitives that can be used with a Koalas ``EntitySet``, you can call ``featuretools.list_primitives()``. This will return a table of all primitives. Any primitive that can be used with a Koalas ``EntitySet`` will have a value of ``True`` in the ``koalas_compatible`` column.


.. ipython:: python

    primitives_df = ft.list_primitives()
    koalas_compatible_df = primitives_df[primitives_df["koalas_compatible"] == True]
    koalas_compatible_df.head()
    koalas_compatible_df.tail()

Primitive Limitations
*********************
At this time, custom primitives created with ``featuretools.primitives.make_trans_primitive()`` or ``featuretools.primitives.make_agg_primitive()`` cannot be used for running deep feature synthesis on a Koalas ``EntitySet``. While it is possible to create custom primitives for use with a Koalas ``EntitySet`` by extending the proper primitive class, there are several potential problems in doing so, and those issues are beyond the scope of this guide.

Entity Limitations
******************
When creating a Featuretools ``Entity`` from Koalas dataframes, variable type inference is not performed as it is when creating entities from pandas dataframes. This is done to improve speed as sampling the data to infer the variable types could require expensive computation on the underlying Koalas dataframe. As a consequence, users must define the variable types for each column in the supplied Dataframe. This step is needed so that the deep feature synthesis process can build the proper features based on the column types. A list of available variable types can be obtained by running ``featuretools.variable_types.find_variable_types()``.

By default, Featuretools checks that entities created from pandas dataframes have unique index values. Because performing this same check with Koalas could be computationally expensive, this check is not performed when creating an entity from a Koalas dataframe. When using Koalas dataframes, users must ensure that the supplied index values are unique.

When an ``Entity`` is created from a pandas dataframe, the ordering of the underlying dataframe rows is maintained. For a Koalas ``Entity``, the ordering of the dataframe rows is not guaranteed, and Featuretools does not attempt to maintain row order in a Koalas ``Entity``. If ordering is important, close attention must be paid to any output to avoid issues.

The ``Entity.add_interesting_values()`` method is not supported when using a Koalas ``Entity``.  If needed, users can manually set ``interesting_values`` on entities by assigning them directly with syntax similar to this: ``es["entity_name"]["variable_name"].interesting_values = ["Value 1", "Value 2"]``.

EntitySet Limitations
*********************
When creating a Featuretools ``EntitySet`` that will be made of Koalas entities, all of the entities used to create the ``EntitySet`` must be of the same type, either all Koalas entities, all Dask entities, or all pandas entities. Featuretools does not support creating an ``EntitySet`` containing a mix of Koalas, Dask, and pandas entities.

Additionally, the ``EntitySet.add_interesting_values()`` method is not supported when using a Koalas ``EntitySet``. Users can manually set ``interesting_values`` on entities, as described above.

DFS Limitations
***************
There are a few key limitations when generating a feature matrix from a Koalas ``EntitySet``.

If a ``cutoff_time`` parameter is passed to ``featuretools.dfs()`` it should be a single cutoff time value, or a pandas dataframe. The current implementation will still work if a Koalas dataframe is supplied for cutoff times, but a ``.to_pandas()`` call will be made on the dataframe to convert it into a pandas dataframe. This conversion will result in a warning, and the process could take a considerable amount of time to complete depending on the size of the supplied dataframe.

Additionally, Featuretools does not currently support the use of the ``approximate`` or ``training_window`` parameters when working with Koalas entitiysets, but should in future releases.

Finally, if the output feature matrix contains a boolean column with ``NaN`` values included, the column type may have a different datatype than the same feature matrix generated from a pandas ``EntitySet``.  If feature matrix column data types are critical, the feature matrix should be inspected to make sure the types are of the proper types, and recast as necessary.

Other Limitations
*****************
Currently ``featuretools.encode_features()`` does not work with a Koalas dataframe as input. This will hopefully be resolved in a future release of Featuretools.

The utility function ``featuretools.make_temporal_cutoffs()`` will not work properly with Koalas inputs for ``instance_ids`` or ``cutoffs``. However, as noted above, if a ``cutoff_time`` dataframe is supplied to ``dfs``, the supplied dataframe should be a pandas dataframe, and this can be generated by supplying pandas inputs to ``make_temporal_cutoffs()``.

The use of ``featuretools.remove_low_information_features()`` cannot currently be used with a Koalas feature matrix.

When manually defining a ``Feature``, the ``use_previous`` parameter cannot be used if this feature will be applied to calculate a feature matrix from a Koalas ``EntitySet``.
