.. _performance:

Improving computational performance
===================================

Feature engineering is a computationally expensive task. While Featuretools comes with reasonable default settings for feature calculation, there are a number of built-in approaches to improve computational performance based on dataset and problem specific considerations.

Reduce number of unique cutoff times
------------------------------------
Each row in a feature matrix created by Featuretools is calculated at a specific cutoff time that represents the last point in time that data from any entity in an entity set can be used to calculate the feature. As a result, calculations incur an overhead in finding the subset of allowed data for each distinct time in the calculation.

.. note::

    Featuretools is very precise in how it deals with time. For more information, see :doc:`/automated_feature_engineering/handling_time`.

If there are a large number of unique cutoff times relative to the number of instances for which we are calculating features, this overhead can outweigh the time needed to calculate the features. Therefore, by reducing the number of unique cutoff times, we minimize the overhead from searching for and extracting data for feature calculations.


Approximating features by rounding cutoff time
----------------------------------------------
One way to decrease the number of unique cutoff times is to round cutoff times to an nearby earlier point in time. An earlier cutoff time is always valid for predictive modeling — it just means we’re not using some of the data we could potentially use while calculating that feature. In that way, we gain computational speed by losing some information.

To understand when approximation is useful, consider calculating features for a model to predict fraudulent credit card transactions. In this case, an important feature might be, "the average transaction amount for this card in the past". While this value can change every time there is a new transaction, updating it less frequently might not impact accuracy.

.. note::

    The bank BBVA used approximation when building a predictive model for credit card fraud using Featuretools. For more details, see the "Real-time deployment considerations" section of the `white paper <https://arxiv.org/pdf/1710.07709.pdf>`_ describing the work.

The frequency of approximation is controlled using the ``approximate`` parameter to ``dfs`` or ``calculate_feature_matrix``. For example, the following code would approximate aggregation features at 1 day intervals::

    fm = ft.calculate_feature_matrix(entityset=entityset
                                     features=feature_list,
                                     cutoff_time=cutoff_times,
                                     approximate="1 day")

In this computation, features that can be approximated will be calculated at 1 day intervals, while features that cannot be approximated (e.g "is the current transaction > $50") will be calculated at the exact cutoff time.

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

Feature Labs
------------
`Feature Labs <http://featurelabs.com>`_ provides tools and support to organizations that want to scale their usage of Featuretools.
