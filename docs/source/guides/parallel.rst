.. _parallel:

Computing feature values in parallel
====================================
Featuretools can optionally compute features on multiple cores using dask's distributed scheduler. The simplest way to do this is to specify the number of cores to use using the ``n_jobs`` parameter::

    fm = ft.calculate_feature_matrix(features=features,
                                     entityset=entityset,
                                     cutoff_time=cutoff_time,
                                     n_jobs=2,
                                     verbose=True)

A local distributed cluster with 2 workers will be created (since ``njobs`` was set to 2) and copies of the features and entityset sent to each worker. Chunks of the feature matrix will be calculated by each worker and returned the original process. Since each worker recieves a copy of the entityset there is a significant increase in memory use with each additional worker used.  There is also some intial time spent transmitting the entityset to the worker processes before calculation can begin.

Using persistent cluster
------------------------
Using the ``n_jobs`` parameter, the distributed cluster will be created for that specific feature matrix calculation and destroyed once calculations have finished.  A drawback of this is that each time a feature matrix is calculated, the entityset has to be transmitted to the workers again.  The time spent on this data transmission step can be cut down to just the first transmission by using the same cluster again.  The way to do this is by creating a cluster first and telling featuretools to use it with the ``dask_kwargs`` parameter::

    import featuretools as ft
    from dask.distributed import LocalCluster

    cluster = LocalCluster()
    fm_1 = ft.calculate_feature_matrix(features=features_1,
                                     entityset=entityset,
                                     cutoff_time=cutoff_time,
                                     dask_kwargs={'cluster': cluster},
                                     verbose=True)

The 'cluster' value can either be the actual cluster object or a string of the address the cluster's scheduler can be reached at.  The call below would also work. This second feature matrix calculation will not need to resend the entityset data to the workers because it has already been saved on the cluster.::

    fm_2 = ft.calculate_feature_matrix(features=features_2,
                                     entityset=entityset,
                                     cutoff_time=cutoff_time,
                                     dask_kwargs={'cluster': cluster.scheduler.address},
                                     verbose=True)


Using the distributed dashboard
-------------------------------
Dask.distributed has a web-based diagnostics dashboard that can be used to analyze the state of the workers and task.  An in-depth description of the web interface can be found `here <https://distributed.readthedocs.io/en/latest/web.html>`_.  The dashboard requires an additional python package, bokeh, to work.  Once bokeh is installed, the web interface will be launched by default when a LocalCluster is created. The cluster created by featuretools when using ``n_jobs`` does not enable the web interface automatically.  To do so, the port to launch the main web interface on must be specified in ``dask_kwargs``::

    fm = ft.calculate_feature_matrix(features=features,
                                     entityset=entityset,
                                     cutoff_time=cutoff_time,
                                     n_jobs=2,
                                     dask_kwargs={'diagnostics_port': 8787}
                                     verbose=True)
