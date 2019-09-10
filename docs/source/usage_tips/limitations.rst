Limitations
-----------
In-memory
*********

Featuretools is intended to be run on datasets that can fit in memory on one machine. For advice on handing large dataset refer to :ref:`performance`.

If you would like to test `Feature Labs APIs <https://docs.featurelabs.com/>`_ for running Featuretools natively on Apache Spark or Dask, please let us know `here <https://forms.gle/TtFTH5QKM4gZtu7U7>`_.

Bring your own labels
*********************

If you are doing supervised machine learning, you must supply your own labels and cutoff times. To structure this process, you can use `Compose <https://compose.featurelabs.com>`_, which is an open source project for automatically generating labels with cutoff times.