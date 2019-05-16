.. _performance:

Improving Computational Performance
===================================

Feature engineering is a computationally expensive task. While Featuretools comes with reasonable default settings for feature calculation, there are a number of built-in approaches to improve computational performance based on dataset and problem specific considerations.

Reduce number of unique cutoff times
------------------------------------
Each row in a feature matrix created by Featuretools is calculated at a specific cutoff time that represents the last point in time that data from any entity in an entity set can be used to calculate the feature. As a result, calculations incur an overhead in finding the subset of allowed data for each distinct time in the calculation.

.. note::

    Featuretools is very precise in how it deals with time. For more information, see :doc:`/automated_feature_engineering/handling_time`.

If you have many unique cutoff times, it is often worthwhile to figure out how to have fewer. This can be done manually by figuring out which unique times are necessary for your prediction problem or automatically using :ref:`approximate <approximate>`.


Adjust chunk size when calculating feature matrix
-------------------------------------------------
When Featuretools calculates a feature matrix, it first groups the rows to be calculated into chunks. Each chunk is a collection of rows that will be computed at the same time. The results of calculating each chunk are combined into the single feature matrix that is returned to you as the user.

If you wish to optimize chunk size, picking the right value will depend on the memory you have available and how often you’d like to get progress updates.

Peak memory usage
^^^^^^^^^^^^^^^^^
If rows have the same cutoff time they are placed in the same chunk until the chunk is full, so they can be calculated simultaneously. By increasing the size of a chunk, it is more likely that there is room for all rows for a given cutoff time to be grouped together. This allows us to minimize the overhead of finding allowable data. The downside is that computation now requires more memory per chunk. If the machine in use can’t handle the larger peak memory usage this can start to slow down the process more than the time saved by removing the overhead.

Frequency of progress updates and overall runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After the completion of a chunk, there is a progress update along with an estimation of remaining compute time provided to the user. Smaller chunks mean more fine-grained updates to the user. Even more, the average runtime over many smaller chunks is a better estimate of the overall remaining runtime.

However, if we make the chunk size too small, we may split up rows that share the same cutoff time into separate chunks. This means Featuretools will have to slice the data for that cutoff time multiple times, resulting in repeated work. Additionally, if chunks get really small (e.g. one row per chunk), then the overhead from other parts of the calculation process will start to contribute more significantly to overall runtime.

We can control chunk size using the ``chunk_size`` argument to ``dfs`` or ``calculate_feature_matrix``. By default, the chunk size is set to 10% of all rows in the feature matrix. We can modify it like this::

    # use 100 rows per chunk
    feature_matrix, features_list = ft.dfs(entityset=es,
                                           target_entity="customers",
                                           chunk_size=100)


We can also set chunksize to be a percentage of total rows or variable based on cutoff times. ::

    # use 5% of rows per chunk
    feature_matrix, features_list = ft.dfs(entityset=es,
                                           target_entity="customers",
                                           chunk_size=.05)

    # use one chunk per unique cutoff time
    feature_matrix, features_list = ft.dfs(entityset=es,
                                           target_entity="customers",
                                           chunk_size="cutoff time")


To understand the impact of chunk size on one Entity Set with multiple entities, see the graph below

.. image:: /images/chunk_size_graph.png

For a more in-depth look at chunk sizes see the :doc:`chunking`

Partition and Distribute Data
-----------------------------
When an entire dataset is not required to calculate the features for a given set of instances, we can split the data into independent partitions and calculate on each partition. For example, imagine we are calculating features for customers and the features are "number of other customers in this zip code" or "average age of other customers in this zip code". In this case, we can load in data partitioned by zip code. As long as we have all of the data for a zip code when calculating, we can calculate all features for a subset of customers.

An example of this approach can be seen in the `Predict Next Purchase demo notebook <https://github.com/featuretools/predict_next_purchase>`_. In this example, we partition data by customer and only load a fixed number of customers into memory at any given time. We implement this easily using `Dask <https://dask.pydata.org/>`_, which could also be used to scale the computation to a cluster of computers. A framework like `Spark <https://spark.apache.org/>`_ could be used similarly.

An additional example of partitioning data to distribute on multiple cores or a cluster using Dask can be seen in the `Featuretools on Dask notebook <https://github.com/Featuretools/Automated-Manual-Comparison/blob/master/Loan%20Repayment/notebooks/Featuretools%20on%20Dask.ipynb>`_. This approach is detailed in the `Parallelizing Feature Engineering with Dask article <https://medium.com/feature-labs-engineering/scaling-featuretools-with-dask-ce46f9774c7d>`_ on the Feature Labs engineering blog. Dask allows for simple scaling to multiple cores on a single computer or multiple machines on a cluster.

For a similar partition and distribute implementation using Apache Spark with PySpark, refer to the `Feature Engineering on Spark notebook <https://github.com/Featuretools/predicting-customer-churn/blob/master/churn/4.%20Feature%20Engineering%20on%20Spark.ipynb>`_. This implementation shows how to carry out feature engineering on a cluster of EC2 instances using Spark as the distributed framework. A write-up of this approach is described in the `Featuretools on Spark article <https://blog.featurelabs.com/featuretools-on-spark-2/>`_ on the Feature Labs engineering blog.

Feature Labs
------------
`Feature Labs <https://www.featurelabs.com>`_ provides tools and support to organizations that want to scale their usage of Featuretools. More information is available `here <https://www.featurelabs.com/featuretools>`_.

If you would like to test `Feature Labs APIs <https://docs.featurelabs.com/>`_ for running Featuretools natively on Apache Spark or Dask, please let us know `here <https://forms.gle/TtFTH5QKM4gZtu7U7>`_.
