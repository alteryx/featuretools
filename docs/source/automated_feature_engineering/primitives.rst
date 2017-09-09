.. _primitives:

Feature primitives
~~~~~~~~~~~~~~~~~~

Feature primitives are the building blocks of Featuretools. They define individual computations that can be applied to raw datasets to create new features. Because a primitive only constrains the input and output data types, they can be applied across datasets and can stack to create new calculations.

Why primitives?
***************

The space of potential functions that humans use to create a feature is expansive. By breaking common feature engineering calculations down into primitive components, we are able to capture the underlying structure of the features humans create today.

A primitive only constrains the input and output data types. This means they can be used to transfer calculations known in one domain to another. Consider a feature which is often calculated by data scientists for transactional or event logs data: `average time between events`. This feature is incredibly valuable in predicting fraudulent behavior or future customer engagement.

DFS achieves the same feature by stacking two primitives ``TimeSincePrevious`` and ``Mean``

.. ipython:: python
    :suppress:

    import featuretools as ft
    es = ft.demo.load_mock_customer(return_entityset=True)
    es

.. ipython:: python

    from featuretools.primitives import TimeSincePrevious, Mean

    feature_defs = ft.dfs(entityset=es,
                          target_entity="customers",
                          agg_primitives=[Mean],
                          trans_primitives=[TimeSincePrevious])
    feature_defs

.. .. note::

..     When ``dfs`` is called with ``features_only=True``, only feature definitions are returned as output. By default this parameter is set to ``False``. This parameter is used quickly inspect the feature definitions before the spending time calculating the feature matrix.


A second advantage of primitives is that they can be used to quickly enumerate many interesting features in a parameterized way. This is used by Deep Feature Synthesis to get several different ways of summarizing the time since the previous event.


.. ipython:: python

    from featuretools.primitives import Mean, Max, Min, Std, Skew

    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity="customers",
                                          agg_primitives=[Mean, Max, Min, Std, Skew],
                                          trans_primitives=[TimeSincePrevious])

    feature_matrix[["MEAN(sessions.time_since_previous_by_customer_id)",
                    "MAX(sessions.time_since_previous_by_customer_id)",
                    "MIN(sessions.time_since_previous_by_customer_id)",
                    "STD(sessions.time_since_previous_by_customer_id)",
                    "SKEW(sessions.time_since_previous_by_customer_id)"]]



Aggregation vs Transform Primitive
**********************************

In the example above, we use two types of primitives.

**Aggregation primitives:** These primitives take related instances as an input and output a single value. They are applied across a parent-child relationship in an entity set. E.g: ``Count``, ``Sum``, ``AvgTimeBetween``.

**Transform primitives:** These primitives take one or more variables from an entity as an input and output a new variable for that entity. They are applied to a single entity. E.g: ``Hour``, ``TimeSincePrevious``, ``Absolute``.




.. Built in Primitives
.. *******************

.. ======================    ==================================================
..  Primitive type           Primitives
.. ======================    ==================================================
..  Aggregation              Min, Max, Count, Sum, Std, Mean, Median, Mode,
..  Datetime transform       Minute, Second, Weekday, Weekend, Hour, Day, Week, Month, Year
..  Cumulative transform     CumCount, CumSum, CumMean, CumMax, CumMin, Diff
..  Combine                  isin, AND, OR, NOT
..  Transform                TimeSince, Absolute
.. ======================    ==================================================



Creating Custom Primitives
**************************

The library of primitives in Featuretools is constantly expanding. In a future release, users will be able to include their primitives through a Custom Primitive API. To contribute a primitive, a user will


  * Specify the type of primitive ``Aggregation`` or ``Transform``
  * Define the input and output data types
  * Write a function in python to do the calculation
  * Annotate with attributes to constrain how it is applied


Once a primitive is contributed, it can stack with existing primitives to generate complex patterns. This enables primitives known to be important for one problem to automatically be transfered to another.
