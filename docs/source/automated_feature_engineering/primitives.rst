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

The code to define custom primitives at runtime already exists.  See the :meth:`make_trans_primitive <featuretools.primitives.make_trans_primitive>` and :meth:`make_agg_primitive <featuretools.primitives.make_agg_primitive>` functions.  A few basic arguments, ``function``, ``input_types``, and ``return_type``, need to be defined for either type of primitive.

* ``function`` is the function to be applied to the input values.
* ``input_types`` is a list that specifies the variable types of features this primitive accepts as input.  In the case of the ``Max`` primitive, it expects one numeric feature as input.  If a primitive requires multiple features as input, ``input_types`` has multiple entries, eg ``[Numeric, Numeric]`` would mean the primitive requires two Numeric features as input.  If a primitive can have multiple input configurations, `input_types` becomes a list of lists.  For example, if a primitive could either have one Numeric feature as input or two, ``input_types`` becomes ``[[Numeric], [Numeric, Numeric]]``.
* ``return_type`` is the variable type of the output.  Since ``Max`` returns a singular value, it's type is None.

Optional arguments when creating a new primitive:

* ``name``: the name of the primitive. If no name is provided, the name of the function will be used instead.
* ``description``: a brief text description of the primitive.
* ``uses_calc_time``: if the function uses the timestamp for when the feature is being calculated, name that variable 'time' and set this to True

Optional arguments when creating new aggregation primitives: `stack_on_self`, `stack_on`, `stack_on_exclude`, `base_of`, `base_of_exclude`. Use them to define the stacking behavior for the new primitive.

Now let's look at an example of a custom primitive:

.. ipython:: python
    :suppress:

    import featuretools as ft


    es = ft.demo.load_retail()

.. ipython:: python

    from featuretools.variable_types import Numeric, DatetimeTimeIndex
    from featuretools.primitives import make_agg_primitive, Feature
    from featuretools import calculate_feature_matrix
    import numpy as np


    def mean_weekly_sum(y, x):
        y.index = x
        return y.fillna(0).resample("7d").mean().fillna(0).mean()

    MeanWeeklySum = make_agg_primitive(function=mean_weekly_sum,
                                       input_types=[Numeric, DatetimeTimeIndex],
                                       return_type=Numeric,
                                       stack_on_self=False,
                                       base_of=[])

    number_per_week = MeanWeeklySum([Feature(es['item_purchases']['Quantity']),
                                     Feature(es['item_purchases']['InvoiceDate'])],
                                    es['items'])

    feature_matrix = calculate_feature_matrix([number_per_week])
    feature_matrix.head(10)

If a new primitive has default arguments in its function, those defaults can be overwritten by including those arguments when initializing the feature.  See the example below:

.. ipython:: python

    from featuretools.primitives import make_trans_primitive
    from featuretools.variable_types import Variable, Boolean


    def is_in(array, list_of_outputs=None):
        import pandas as pd
        if list_of_outputs is None:
            list_of_outputs = []
        return pd.Series(array).isin(list_of_outputs)

    IsIn = make_trans_primitive(is_in,
                                [Variable],
                                Boolean,
                                description="For each value of the base feature, checks whether it is in a list that provided.",)

    isin_feature = IsIn(Feature(es['item_purchases']['StockCode']),
                                list_of_outputs=['84029G'])

    feature_matrix = calculate_feature_matrix([isin_feature])
    feature_matrix.head(10)
