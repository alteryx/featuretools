.. _performance:

Improving Computational Performance
===================================

Feature engineering is a computationally expensive task. While Featuretools comes with reasonable default settings for feature calculation, there are a number of built-in approaches to improve computational performance based on dataset and problem specific considerations.

Reduce number of unique cutoff times
------------------------------------
Each row in a feature matrix created by Featuretools is calculated at a specific cutoff time that represents the last point in time that data from any entity in an entity set can be used to calculate the feature. As a result, calculations incur an overhead in finding the subset of allowed data for each distinct time in the calculation.

.. note::

    Featuretools is very precise in how it deals with time. For more information, see :doc:`/automated_feature_engineering/handling_time`.

If there are many unique cutoff times, it is often worthwhile to figure out how to have fewer. This can be done manually by figuring out which unique times are necessary for the prediction problem or automatically using :ref:`approximate <approximate>`.


Adjust chunk size
-----------------
Featuretools calculates rows with the same cutoff simultaneously. The `chunk_size` parameter limits the maximum number of rows that will be grouped and then calculated together.

By default, there is no limit to the number of rows that will be calculated at once for a particular cutoff time unless you are using parallel processing in which case the chunk size is set to be ``1 / n_jobs`` to ensure the computation can be spread across available workers. Normally this behavior is works well, but if the resulting feature matrix has only a few cutoff times it can lead to high memory usage or limited parallelism (if `n_jobs > 1`).

By setting a cutoff time, we can limit the maximum number of rows in each group to specific number or a percentage of the overall data

We can control chunk size using the ``chunk_size`` argument to ``dfs`` or ``calculate_feature_matrix``::

    # use 100 rows per chunk
    feature_matrix, features_list = ft.dfs(entityset=es,
                                           target_entity="customers",
                                           chunk_size=100)


We can also set chunk size to be a percentage of total rows::

    # use 5% of rows per chunk
    feature_matrix, features_list = ft.dfs(entityset=es,
                                           target_entity="customers",
                                           chunk_size=.05)


The smaller chunk size may result in lower peak memory usage during calculation a group of rows or greater opportunity for parallelism.


Partition and Distribute Data
-----------------------------
When an entire dataset is not required to calculate the features for a given set of instances, we can split the data into independent partitions and calculate on each partition. For example, imagine we are calculating features for customers and the features are "number of other customers in this zip code" or "average age of other customers in this zip code". In this case, we can load in data partitioned by zip code. As long as we have all of the data for a zip code when calculating, we can calculate all features for a subset of customers.

An example of this approach can be seen in the `Predict Next Purchase demo notebook <https://github.com/featuretools/predict_next_purchase>`_. In this example, we partition data by customer and only load a fixed number of customers into memory at any given time. We implement this easily using `Dask <https://dask.pydata.org/>`_, which could also be used to scale the computation to a cluster of computers. A framework like `Spark <https://spark.apache.org/>`_ could be used similarly.

An additional example of partitioning data to distribute on multiple cores or a cluster using Dask can be seen in the `Featuretools on Dask notebook <https://github.com/Featuretools/Automated-Manual-Comparison/blob/master/Loan%20Repayment/notebooks/Featuretools%20on%20Dask.ipynb>`_. This approach is detailed in the `Parallelizing Feature Engineering with Dask article <https://medium.com/feature-labs-engineering/scaling-featuretools-with-dask-ce46f9774c7d>`_ on the Feature Labs engineering blog. Dask allows for simple scaling to multiple cores on a single computer or multiple machines on a cluster.

For a similar partition and distribute implementation using Apache Spark with PySpark, refer to the `Feature Engineering on Spark notebook <https://github.com/Featuretools/predicting-customer-churn/blob/master/churn/4.%20Feature%20Engineering%20on%20Spark.ipynb>`_. This implementation shows how to carry out feature engineering on a cluster of EC2 instances using Spark as the distributed framework. A write-up of this approach is described in the `Featuretools on Spark article <https://blog.featurelabs.com/featuretools-on-spark-2/>`_ on the Feature Labs engineering blog.

Featuretools Enterprise
-----------------------
If you don't want to build it yourself, Featuretools Enterprise has native integrations with Apache Spark and Dask. More information is available `here <https://www.featurelabs.com/featuretools>`__.

If you would like to test `Featuretools Enterprise APIs <https://docs.featurelabs.com/>`_ for running Featuretools natively on Apache Spark or Dask, please let us know `here <https://forms.gle/TtFTH5QKM4gZtu7U7>`__.
