.. _handling-time:
.. currentmodule:: featuretools

Handling time
=============


When performing feature engineering to learn a model to predict the future, the value to predict will be associated with a time. In this case, it is paramount to only incorporate data prior to this "cutoff time" when calculating the feature values.

Featuretools is designed to take time into consideration when required. By specifying a cutoff time, we can control what portions of the data are used when calculating features.


.. ipython:: python

    import featuretools as ft

    es = ft.demo.load_mock_customer(return_entityset=True)


**Motivating Example**

  Consider the problem to predict if a customer is likely to buy an upgrade to their membership plan. To do this, you first identify historical examples of customers who upgraded and others who did not. For each customer, you can only use the interactions s/he had prior to upgrading or not upgrading their membership. This is a requirement -- by definition.

  The example above illustrates the importance of time in calculating features. Other situations are more subtle, and hence when building predictive models it is important identify if time is a consideration. If feature calculation does not account for time, it may include data in calculations that is past the outcome we want to predict and may cause the well known problem of *Label Leakage*.

.. todo: include citation for label leakage paper.

Cutoff times
------------
We can specify the time for each instance of the ``target_entity`` to calculate features. The timestamp represents the last time data can be used for calculating features. This is specified using a dataframe of cutoff times. Below we show an example of this dataframe for our customers example.

.. ipython:: python

    import pandas as pd
    cutoff_times = pd.DataFrame({"customer_id": [1, 2, 3, 4, 5],
                                 "time": pd.date_range('2014-01-01 01:41:50', periods=5, freq='25min')})
    cutoff_times


.. In many real world scenarios, these cutoff times may also come from human observations or annotations of a real world phenomena.

.. We build a list of cutoff times below. Each row has the id of the row we want features for and the time to calculate feature for that instance.
.. These cutoff times are usually provided to the feature engineering along with labels associated with each entity-instance.

Time index for an entity
------------------------
Given the cutoff time for each instance of the target entity, Featuretools needs to automatically identify the data points that are prior to this time point across all entities. In most temporal datasets, entities have a column that specifies the point in time when data in that row became available.

Users specify this point in time when a particular row became known by defining a time index for each entity. Read about setting the time index in :doc:`/loading_data/using_entitysets`.


Running DFS with cutoff times
-----------------------------

We provide the cutoff times as a parameter to DFS.

.. ipython:: python

    feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="customers",
                                      cutoff_time=cutoff_times)
    feature_matrix

There is one row in the feature matrix corresponding to a row in ``cutoff_times``. The feature values in this row use only data prior to the cutoff time. Additionally, the returned feature matrix will be ordered by the time the rows was calculated, not by the order of cutoff_times. We can add the cutoff time to the returned feature matrix by using ``cutoff_time_in_index`` as shown below

.. ipython:: python

    feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="customers",
                                      cutoff_time=cutoff_times,
                                      cutoff_time_in_index=True)
    feature_matrix

It is often the case that we want our labels in our calculated feature matrix so that the ordering is consistent between the labels and the rows of the feature matrix. However, adding labels to the initial dataframe means that you would have to explicitly prohibit ``dfs`` from building features with that column. To bypass this, we can provide additional columns to cutoff times which will be added directly the feature matrix. While the first two columns will be used as an index and cutoff time regardless of their order in the dataframe, any additional columns will appear as features in the resulting feature matrix. 

.. ipython:: python

    cutoff_times['label'] = pd.Series([0, 0, 1, 0, 1])

    feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="customers",
                                      cutoff_time=cutoff_times,
                                      cutoff_time_in_index=True)

    feature_matrix['label']

Running DFS with training windows
---------------------------------

Training windows are an extension of cutoff times: starting from the cutoff time and moving backwards through time, only data within that window of time will be used to calculate features. This example creates a window that only includes transactions that occurred less than 1 hour before the cutoff


.. ipython:: python

    window_fm, window_features = ft.dfs(entityset=es,
                                        target_entity="customers",
                                        cutoff_time=cutoff_times,
                                        cutoff_time_in_index=True,
                                        training_window="1 hour")
    window_fm


We can see that that the counts for the same feature are lower when we shorten the training window

.. ipython:: python

    feature_matrix[["COUNT(transactions)"]]
    window_fm[["COUNT(transactions)"]]


.. Using Timedeltas
.. ----------------
.. To represent timespans, we can use the :class:`.Timedelta` class. Timedelta provides a simple human readable format to define lengths of time in absolute and relative units. For example we can define a timespan 7 days, or of three log events:

.. .. code-block:: python

..     offset_1 = ft.Timedelta(7, "days")

..     # Note: observation entity is defined
..     offset_2 = ft.Timedelta(3, "observations", es["logs"])
