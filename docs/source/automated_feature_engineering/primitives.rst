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



Defining Custom Primitives
**************************

The library of primitives in Featuretools is constantly expanding.  Users can define their own primitive using the APIs below.  To define a primitive, a user will


  * Specify the type of primitive ``Aggregation`` or ``Transform``
  * Define the input and output data types
  * Write a function in python to do the calculation
  * Annotate with attributes to constrain how it is applied


Once a primitive is defined, it can stack with existing primitives to generate complex patterns. This enables primitives known to be important for one domain to automatically be transfered to another.

Simple Custom Primitives
========================
.. ipython :: python

    from featuretools.primitives import make_agg_primitive, make_trans_primitive
    from featuretools.variable_types import Text, Numeric
    import numpy as np


    Absolute = make_trans_primitive(function=lambda array: np.absolute(array),
                                    input_types=[Numeric],
                                    return_type=Numeric)

    Mean = make_agg_primitive(function=np.nanmean,
                              input_types=[Numeric],
                              return_type=Numeric)

Both :meth:`make_agg_primitive <featuretools.primitives.make_agg_primitive>` and :meth:`make_trans_primitive <featuretools.primitives.make_trans_primitive>` require three arguments to create a new primitive class: ``function``, ``input_types``, and ``return_type``.

Functions With Additonal Arguments
==================================
One caveat with the make\_primitive functions is that the required arguments of ``function`` must be input features.  Here we create a function for ``StringCount``, a primitive which counts the number of occurrences of a string in a ``Text`` input.  Since ``string`` is not a feature, it needs to be a keyword argument to ``string_count``.

.. ipython:: python

    def string_count(array, string=None):
        '''
        ..note:: this is a naive implementation used for clarity
        '''
        assert string is not None, "string to count needs to be defined"
        counts = [element.count(string) for element in array]
        return counts

Now that we have the function we create the primitive using the ``make_trans_primitive`` function.

.. ipython:: python

    StringCount = make_trans_primitive(function=string_count,
                                       input_types=[Text],
                                       return_type=Numeric)

Passing in ``string="test"`` as a keyword argument when creating a StringCount feature will make "test" the value used for string when ``string_count`` is called to calculate the feature values.  Now we use this primitive to create a feature and calculate the feature values.

.. ipython:: python

    from featuretools.tests.testing_utils import make_ecommerce_entityset

    es = make_ecommerce_entityset()
    es["customers"].df["favorite_quote"].sort_index()
    count_the_feat = StringCount(es['customers']['favorite_quote'], string="the")
    feature_matrix = ft.calculate_feature_matrix(features=[count_the_feat])
    feature_matrix

Multiple Input Types
====================
If a primitive requires multiple features as input, ``input_types`` has multiple elements, eg ``[Numeric, Numeric]`` would mean the primitive requires two Numeric features as input.  Below is an example of a primitive that has multiple input features.

.. ipython:: python

    from featuretools.variable_types import Datetime, Timedelta, Variable
    from featuretools.primitives import Feature, Equals
    import pandas as pd

    def count_sunday(to_count, datetime):
        '''
        Counts non-null values that occurred on Sundays
        '''
        days = pd.DatetimeIndex(datetime).weekday.values
        df = pd.DataFrame({'to_count': to_count, 'time': days})
        return df[df['time'] == 6]['to_count'].dropna().count()

    CountSunday = make_agg_primitive(function=count_sunday,
                                     input_types=[Variable, Datetime],
                                     return_type=Numeric)

    count_sunday_log_entries = CountSunday([es["log"]["value"],
                                            es["log"]["datetime"]],
                                           es["sessions"])
    feature_matrix = ft.calculate_feature_matrix(features=[count_sunday_log_entries])
    feature_matrix
