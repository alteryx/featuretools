.. _scaling:

Scaling to big data
===================

Featuretools is optimized for usage with datasets that fit in memory and can be processed on a single machine. However, there are many approaches to scale to datasets that are larger than memory or stored in a distributed environment.

Reduce number of unique cutoff times
------------------------------------
Every feature calculation incurs overhead from dealing with time. If we have a large number of cutoff times relative to the number of instances we are calculating feature for, this overhead begins to add up. When possible try to reduce the number of unique times that features are calculated.

Load chunks of data
--------------------
When an entire dataset is not required to calculate the features for a given set of instances, we can calculate by chunk. For example, if are calculating feature for customers and our highest level features are "number of customers in this zip code" or "average age of customers in this zip code". In this case, we can load data in by zip code. As long as we have all of the data for a zip code when calculating, we can calculate all features for a subset of customers. If we store our data in a database or in HDF5 using `Pytables <http://www.pytables.org>`_, it we can work with larger than memory datasets this way.


Use Spark or Dask to distribute computation
---------------------------------------------
If the data is so big that loading in chunks isn't an option, we can distribute the data and parallelize the computation using frameworks like `Spark <https://spark.apache.org/docs/latest/api/python/index.html>`_ or `Dask <http://dask.pydata.org/en/latest/>`_. Both of these systems support a dataframe interface that can easily be used to partition data as needed. Because Featuretools is a python library, it is easy to integrate.


Feature Labs
------------
`Feature Labs <http://featurelabs.com>`_ provides tools and support to organizations that want to scale their usage of Featuretools.
