.. _primitives:

Feature primitives
~~~~~~~~~~~~~~~~~~

Feature primitives are the building blocks of Featuretools. They define individual computations that can be applied to raw datasets to create new features. Because a primitive only constrains the input and output data types, they can be applied across datasets and can stack to create new calculations.

Why primitives?
***************

The space of potential functions that humans use to create a feature is expansive. By breaking common feature engineering calculations down into primitive components, we are able to capture the underlying structure of the features humans create today.

A primitive only constrains the input and output data types. This means they can be used to transfer calculations known in one domain to another. Consider a feature which is often calculated by data scientists for transactional or event logs data: `average time between events`. This feature is incredibly valuable in predicting fraudulent behavior or future customer engagement.

DFS achieves the same feature by stacking two primitives ``"time_since_previous"`` and ``"mean"``

.. ipython:: python
    :suppress:

    import featuretools as ft
    es = ft.demo.load_mock_customer(return_entityset=True)
    es

.. ipython:: python

    feature_defs = ft.dfs(entityset=es,
                          target_entity="customers",
                          agg_primitives=["mean"],
                          trans_primitives=["time_since_previous"],
                          features_only=True)
    feature_defs

.. note::

    When ``dfs`` is called with ``features_only=True``, only feature definitions are returned as output. By default this parameter is set to ``False``. This parameter is used quickly inspect the feature definitions before the spending time calculating the feature matrix.


A second advantage of primitives is that they can be used to quickly enumerate many interesting features in a parameterized way. This is used by Deep Feature Synthesis to get several different ways of summarizing the time since the previous event.


.. ipython:: python

    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity="customers",
                                          agg_primitives=["mean", "max", "min", "std", "skew"],
                                          trans_primitives=["time_since_previous"])

    feature_matrix[["MEAN(sessions.TIME_SINCE_PREVIOUS(session_start))",
                    "MAX(sessions.TIME_SINCE_PREVIOUS(session_start))",
                    "MIN(sessions.TIME_SINCE_PREVIOUS(session_start))",
                    "STD(sessions.TIME_SINCE_PREVIOUS(session_start))",
                    "SKEW(sessions.TIME_SINCE_PREVIOUS(session_start))"]]



Aggregation vs Transform Primitive
**********************************

In the example above, we use two types of primitives.

**Aggregation primitives:** These primitives take related instances as an input and output a single value. They are applied across a parent-child relationship in an entity set. E.g: ``"count"``, ``"sum"``, ``"avg_time_between"``.


.. graphviz:: graphs/agg_feat.dot


**Transform primitives:** These primitives take one or more variables from an entity as an input and output a new variable for that entity. They are applied to a single entity. E.g: ``"hour"``, ``"time_since_previous"``, ``"absolute"``.


.. graphviz:: graphs/trans_feat.dot


The above graphs were generated using the :func:`graph_feature <featuretools.graph_feature>` function. These feature lineage graphs help to visually show how primitives were stacked to generate a feature.


For a DataFrame that lists and describes each built-in primitive in Featuretools, call ``ft.list_primitives()``.  In addition, a list of all available primitives can be obtained by visiting `primitives.featurelabs.com <https://primitives.featurelabs.com/>`__.


.. ipython:: python

    ft.list_primitives().head(5)

.. ======================       ==================================================
..  Primitive type              Primitives
.. ======================       ==================================================
..  Aggregation                 min, max, count, sum, std, mean, median, mode,
..  Datetime transform          minute, second, weekday, is_weekend, hour, day, week, month, year
..  Cumulative transform        cum_count, cum_sum, cum_mean, cum_max, cum_min, diff
..  Combine                     is_in, and, or, not
..  Transform                   time_since, absolute, percentile
..  Uses Full Entity Transform  percentile
.. ===========================  ==================================================



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
    from featuretools.variable_types import NaturalLanguage, Numeric

    def absolute(column):
        return abs(column)

    Absolute = make_trans_primitive(function=absolute,
                                    input_types=[Numeric],
                                    return_type=Numeric)

Above we created a new transform primitive that can be used with Deep Feature Synthesis using :meth:`make_trans_primitive <featuretools.primitives.make_trans_primitive>` and a python function we defined.  Additionally, we annotated the input data types that the primitive can be applied to and the data type it returns.

Similarly, we can make a new aggregation primitive using :meth:`make_agg_primitive <featuretools.primitives.make_agg_primitive>`.

.. ipython :: python

    def maximum(column):
        return max(column)

    Maximum = make_agg_primitive(function=maximum,
                              input_types=[Numeric],
                              return_type=Numeric)


Because we defined an aggregation primitive, the function takes in a list of values but only returns one.

Now that we've defined two primitives, we can use them with the dfs function as if they were built-in primitives.

.. ipython :: python

    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity="sessions",
                                          agg_primitives=[Maximum],
                                          trans_primitives=[Absolute],
                                          max_depth=2)

    feature_matrix[["customers.MAXIMUM(transactions.amount)", "MAXIMUM(transactions.ABSOLUTE(amount))"]].head(5)

Word Count Example
=========================
Here we define a function, ``word_count``, which counts the number of words in each row of an input and returns a  list of the counts.

.. ipython :: python

    def word_count(column):
        '''
        Counts the number of words in each row of the column. Returns a list
        of the counts for each row.
        '''
        word_counts = []
        for value in column:
            words = value.split(None)
            word_counts.append(len(words))
        return word_counts

Next, we need to create a custom primitive from the ``word_count`` function.

.. ipython :: python

    WordCount = make_trans_primitive(function=word_count,
                                     input_types=[NaturalLanguage],
                                     return_type=Numeric)

.. ipython :: python

    from featuretools.tests.testing_utils import make_ecommerce_entityset
    es = make_ecommerce_entityset()

Since WordCount is a transform primitive, we need to add it to the list of transform primitives DFS can use when generating features.

.. ipython :: python

    feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="sessions",
                                      agg_primitives=["sum", "mean", "std"],
                                      trans_primitives=[WordCount])

    feature_matrix[["customers.WORD_COUNT(favorite_quote)", "STD(log.WORD_COUNT(comments))", "SUM(log.WORD_COUNT(comments))", "MEAN(log.WORD_COUNT(comments))"]]

By adding some aggregation primitives as well, Deep Feature Synthesis was able to make four new features from one new primitive.

Multiple Input Types
====================
If a primitive requires multiple features as input, ``input_types`` has multiple elements, eg ``[Numeric, Numeric]`` would mean the primitive requires two Numeric features as input.  Below is an example of a primitive that has multiple input features.

.. ipython:: python

    from featuretools.variable_types import Datetime, Timedelta, Variable
    import pandas as pd

    def mean_sunday(numeric, datetime):
        '''
        Finds the mean of non-null values of a feature that occurred on Sundays
        '''
        days = pd.DatetimeIndex(datetime).weekday.values
        df = pd.DataFrame({'numeric': numeric, 'time': days})
        return df[df['time'] == 6]['numeric'].mean()

    MeanSunday = make_agg_primitive(function=mean_sunday,
                                     input_types=[Numeric, Datetime],
                                     return_type=Numeric)

    feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="sessions",
                                      agg_primitives=[MeanSunday],
                                      trans_primitives=[],
                                      max_depth=1)
    feature_matrix[["MEAN_SUNDAY(log.value, datetime)", "MEAN_SUNDAY(log.value_2, datetime)"]]
