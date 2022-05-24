Limitations
-----------
In-memory
*********

Featuretools is intended to be run on datasets that can fit in memory on one machine. For advice on handing large dataset refer to :ref:`Improving Computational Performance <performance>`.

Bring your own labels
*********************

If you are doing supervised machine learning, you must supply your own labels and cutoff times. To structure this process, you can use `Compose <https://compose.featurelabs.com>`_, which is an open source project for automatically generating labels with cutoff times.
