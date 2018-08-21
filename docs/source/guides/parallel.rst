.. _parallel:

Parallel Feature Computation
============================
Featuretools can optionally compute features on multiple cores. The simplest way to control the amount of parallelism is to specify the ``n_jobs`` parameter::

    fm = ft.calculate_feature_matrix(features=features,
                                     entityset=entityset,
                                     cutoff_time=cutoff_time,
                                     n_jobs=2,
                                     verbose=True)

The above command will start 2 processes to compute chunks of the feature matrix in parallel. Each process receives its own copy of the entity set, so memory use will be proportional to the number of parallel processes. Because the entity set has to be copied to each process, there is overhead to perform this operation before calculation can begin. To avoid this overhead on successive calls to ``calculate_feature_matrix``, read the section below on using a persistent cluster. 

Using persistent cluster
------------------------
Behind the scenes, Featuretools uses `dask's <http://dask.pydata.org/>`_ distributed scheduler to implement multiprocessing. When you only specify the ``n_jobs`` parameter, a cluster will be created for that specific feature matrix calculation and destroyed once calculations have finished. A drawback of this is that each time a feature matrix is calculated, the entity set has to be transmitted to the workers again. To avoid this, we would like to reuse the same cluster between calls. The way to do this is by creating a cluster first and telling featuretools to use it with the ``dask_kwargs`` parameter::

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


Using the distributed dashboard
-------------------------------
Dask.distributed has a web-based diagnostics dashboard that can be used to analyze the state of the workers and tasks. It can also be useful for tracking memory use or visualizing task run-times. An in-depth description of the web interface can be found `here <https://distributed.readthedocs.io/en/latest/web.html>`_.

.. image:: /images/dashboard.png

The dashboard requires an additional python package, bokeh, to work. Once bokeh is installed, the web interface will be launched by default when a LocalCluster is created. The cluster created by featuretools when using ``n_jobs`` does not enable the web interface automatically. To do so, the port to launch the main web interface on must be specified in ``dask_kwargs``::

    fm = ft.calculate_feature_matrix(features=features,
                                     entityset=entityset,
                                     cutoff_time=cutoff_time,
                                     n_jobs=2,
                                     dask_kwargs={'diagnostics_port': 8787}
                                     verbose=True)
                                     
Parallel Computation by Partioning Data
-------------------------------
As an alternative to Featuretool's parallelization, the data can be partitioned and run on multiple cores or a cluster using Dask or PySpark. This approach may be necessary with a large `Entityset` because the current parallel implementation sends the entire `EntitySet` to each worker which may exhaust the worker memory. For more information on partitioning the data and using Dask, see :doc:`/guides/performance`. Dask allows Featuretools to scale to multiple cores on a single machine or multiple machines on a cluster.  
