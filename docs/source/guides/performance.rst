.. _performance:

Improving Computational Performance
===================================

Feature engineering is a computationally expensive task. While Featuretools comes with reasonable default settings for feature calculation, there are a number of built-in approaches to improve computational performance based on dataset and problem specific considerations.

Reduce number of unique cutoff times
------------------------------------
Each row in a feature matrix created by Featuretools is calculated at a specific cutoff time that represents the last point in time that data from any entity in an entity set can be used to calculate the feature. As a result, calculations incur an overhead in finding the subset of allowed data for each distinct time in the calculation.

.. note::

    Featuretools is very precise in how it deals with time. For more information, see :doc:`/getting_started/handling_time`.

If there are many unique cutoff times, it is often worthwhile to figure out how to have fewer. This can be done manually by figuring out which unique times are necessary for the prediction problem or automatically using :ref:`approximate <approximate>`.

Parallel Feature Computation
----------------------------
Computational performance can often be improved by parallelizing the feature calculation process. There are several different approaches that can be used to perform parallel feature computation with Featuretools. An overview of the most commonly used approaches is provided below.

Computation with Dask and Koalas EntitySets (BETA)
**************************************************
.. note::
    Support for Dask EntitySets and Koalas EntitySets is still in Beta. While the key functionality has been implemented, development is ongoing to add the remaining functionality.

    All planned improvements to the Featuretools/Dask and Featuretools/Koalas integration are documented on Github (`Dask issues <https://github.com/FeatureLabs/featuretools/issues?q=is%3Aopen+is%3Aissue+label%3ADask>`_, `Koalas issues <https://github.com/FeatureLabs/featuretools/issues?q=is%3Aopen+is%3Aissue+label%3AKoalas>`_). If you see an open issue that is important for your application, please let us know by upvoting or commenting on the issue. If you encounter any errors using Dask or Koalas entities, or find missing functionality that does not yet have an open issue, please create a `new issue on Github <https://github.com/FeatureLabs/featuretools/issues>`_.

Dask or Koalas can be used with Featuretools to perform parallel feature computation with virtually no changes to the workflow required. Featuretools supports creating an ``EntitySet`` directly from Dask or Koalas dataframes instead of using pandas dataframes, enabling the parallel and distributed computation capabilities of Dask or Spark to be used. By creating an ``EntitySet`` directly from Dask or Koalas dataframes, Featuretools can be used to generate a larger-than-memory feature matrix, something that may be difficult with other approaches. When computing a feature matrix from an ``EntitySet`` created from Dask or Koalas dataframes, the resulting feature matrix will be returned as a Dask or Koalas dataframe depending on which type was used.

These methods do have some limitations in terms of the primitives that are available and the optional parameters that can be used when calculating the feature matrix. For more information on generating a feature matrix with this approach, refer to the guides :doc:`/guides/using_dask_entitysets` and :doc:`/guides/using_koalas_entitysets`.

Simple Parallel Feature Computation
***********************************
If using a pandas ``EntitySet``, Featuretools can optionally compute features on multiple cores. The simplest way to control the amount of parallelism is to specify the ``n_jobs`` parameter::

    fm = ft.calculate_feature_matrix(features=features,
                                     entityset=entityset,
                                     cutoff_time=cutoff_time,
                                     n_jobs=2,
                                     verbose=True)

The above command will start 2 processes to compute chunks of the feature matrix in parallel. Each process receives its own copy of the entity set, so memory use will be proportional to the number of parallel processes. Because the entity set has to be copied to each process, there is overhead to perform this operation before calculation can begin. To avoid this overhead on successive calls to ``calculate_feature_matrix``, read the section below on using a persistent cluster.

Adjust chunk size
+++++++++++++++++
By default, Featuretools calculates rows with the same cutoff time simultaneously. The `chunk_size` parameter limits the maximum number of rows that will be grouped and then calculated together. If calculation is done using parallel processing, the default chunk size is set to be ``1 / n_jobs`` to ensure the computation can be spread across available workers. Normally, this behavior works well, but if there are only a few unique cutoff times it can lead to higher peak memory usage (due to more intermediate calculations stored in memory) or limited parallelism (if the number of chunks is less than `n_jobs`).

By setting ``chunk_size``, we can limit the maximum number of rows in each group to specific number or a percentage of the overall data when calling ``ft.dfs`` or ``ft.calculate_feature_matrix``::

    # use maximum  100 rows per chunk
    feature_matrix, features_list = ft.dfs(entityset=es,
                                           target_entity="customers",
                                           chunk_size=100)


We can also set chunk size to be a percentage of total rows::

    # use maximum 5% of all rows per chunk
    feature_matrix, features_list = ft.dfs(entityset=es,
                                           target_entity="customers",
                                           chunk_size=.05)

Using persistent cluster
++++++++++++++++++++++++
Behind the scenes, Featuretools uses `Dask's <http://dask.pydata.org/>`_ distributed scheduler to implement multiprocessing. When you only specify the ``n_jobs`` parameter, a cluster will be created for that specific feature matrix calculation and destroyed once calculations have finished. A drawback of this is that each time a feature matrix is calculated, the entity set has to be transmitted to the workers again. To avoid this, we would like to reuse the same cluster between calls. The way to do this is by creating a cluster first and telling featuretools to use it with the ``dask_kwargs`` parameter::

    import featuretools as ft
    from dask.distributed import LocalCluster

    cluster = LocalCluster()
    fm_1 = ft.calculate_feature_matrix(features=features_1,
                                       entityset=entityset,
                                       cutoff_time=cutoff_time,
                                       dask_kwargs={'cluster': cluster},
                                       verbose=True)

The 'cluster' value can either be the actual cluster object or a string of the address the cluster's scheduler can be reached at. The call below would also work. This second feature matrix calculation will not need to resend the entityset data to the workers because it has already been saved on the cluster.::

    fm_2 = ft.calculate_feature_matrix(features=features_2,
                                       entityset=entityset,
                                       cutoff_time=cutoff_time,
                                       dask_kwargs={'cluster': cluster.scheduler.address},
                                       verbose=True)

.. note::

    When using a persistent cluster, Featuretools publishes a copy of the ``EntitySet`` to the cluster the first time it calculates a feature matrix. Based on the ``EntitySet``'s metadata the cluster will reuse it for successive computations. This means if two ``EntitySets`` have the same metadata but different row values (e.g. new data is added to the ``EntitySet``), Featuretools won’t recopy the second ``EntitySet`` in later calls. A simple way to avoid this scenario is to use a unique ``EntitySet`` id.

Using the distributed dashboard
+++++++++++++++++++++++++++++++
Dask.distributed has a web-based diagnostics dashboard that can be used to analyze the state of the workers and tasks. It can also be useful for tracking memory use or visualizing task run-times. An in-depth description of the web interface can be found `here <https://distributed.readthedocs.io/en/latest/web.html>`_.

.. image:: /images/dashboard.png

The dashboard requires an additional python package, bokeh, to work. Once bokeh is installed, the web interface will be launched by default when a LocalCluster is created. The cluster created by featuretools when using ``n_jobs`` does not enable the web interface automatically. To do so, the port to launch the main web interface on must be specified in ``dask_kwargs``::

    fm = ft.calculate_feature_matrix(features=features,
                                     entityset=entityset,
                                     cutoff_time=cutoff_time,
                                     n_jobs=2,
                                     dask_kwargs={'diagnostics_port': 8787}
                                     verbose=True)

Parallel Computation by Partitioning Data
*****************************************
As an alternative to Featuretools' parallelization, the data can be partitioned and the feature calculations run on multiple cores or a cluster using Dask or Apache Spark with PySpark. This approach may be necessary with a large pandas ``EntitySet`` because the current parallel implementation sends the entire ``EntitySet`` to each worker which may exhaust the worker memory. Dask and Spark allow Featuretools to scale to multiple cores on a single machine or multiple machines on a cluster.

.. note::
    Partitioning data is not necessary when using a Dask ``EntitySet``, as the Dask dataframes that make up the ``EntitySet`` are already partitioned. Partitioning is only needed when working with pandas entities.

When an entire dataset is not required to calculate the features for a given set of instances, we can split the data into independent partitions and calculate on each partition. For example, imagine we are calculating features for customers and the features are "number of other customers in this zip code" or "average age of other customers in this zip code". In this case, we can load in data partitioned by zip code. As long as we have all of the data for a zip code when calculating, we can calculate all features for a subset of customers.

An example of this approach can be seen in the `Predict Next Purchase demo notebook <https://github.com/featuretools/predict_next_purchase>`_. In this example, we partition data by customer and only load a fixed number of customers into memory at any given time. We implement this easily using `Dask <https://dask.pydata.org/>`_, which could also be used to scale the computation to a cluster of computers. A framework like `Spark <https://spark.apache.org/>`_ could be used similarly.

An additional example of partitioning data to distribute on multiple cores or a cluster using Dask can be seen in the `Featuretools on Dask notebook <https://github.com/Featuretools/Automated-Manual-Comparison/blob/main/Loan%20Repayment/notebooks/Featuretools%20on%20Dask.ipynb>`_. This approach is detailed in the `Parallelizing Feature Engineering with Dask article <https://medium.com/feature-labs-engineering/scaling-featuretools-with-dask-ce46f9774c7d>`_ on the Feature Labs engineering blog. Dask allows for simple scaling to multiple cores on a single computer or multiple machines on a cluster.

For a similar partition and distribute implementation using Apache Spark with PySpark, refer to the `Feature Engineering on Spark notebook <https://github.com/Featuretools/predict-customer-churn/blob/main/churn/4.%20Feature%20Engineering%20on%20Spark.ipynb>`_. This implementation shows how to carry out feature engineering on a cluster of EC2 instances using Spark as the distributed framework. A write-up of this approach is described in the `Featuretools on Spark article <https://blog.featurelabs.com/featuretools-on-spark-2/>`_ on the Feature Labs engineering blog.
