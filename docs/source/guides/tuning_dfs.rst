Tuning Deep Feature Synthesis
=============================

There are several parameters that can be tuned to change the output of DFS.


.. ipython:: python

    import featuretools as ft
    es = ft.demo.load_mock_customer(return_entityset=True)
    es

Using "Seed Features"
*********************

Seed features are manually defined, problem specific, features a user provides to DFS. Deep Feature Synthesis will then automatically stack new features on top of these features when it can.

By using seed features, we can include domain specific knowledge in feature engineering automation.

.. ipython:: python

    from featuretools.primitives import PercentTrue

    expensive_purchase = ft.Feature(es["transactions"]["amount"]) > 125

    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity="customers",
                                          agg_primitives=[PercentTrue],
                                          seed_features=[expensive_purchase])
    feature_matrix[['PERCENT_TRUE(transactions.amount > 125)']]

We can now see that ``PERCENT_TRUE`` was automatically applied to this boolean variable.

Add "interesting" values to variables
*************************************

Sometimes we want to create features that are conditioned on a second value before we calculate. We call this extra filter a "where clause".

By default, where clauses are built using the ``interesting_values`` of a variable.


.. Interesting values can be automatically added to all variables by calling `EntitySet.add_interesting_values` or `Entity.add_interesting_values`. We can manually specify interesting values by directly as well.

.. Currently, interesting values are only considered for variables of type :class:`.variable_types.Categorical`, :class:`.variable_types.Ordinal`, and :class:`.variable_types.Boolean`.

.. ipython:: python

    es["sessions"]["device"].interesting_values = ["desktop", "mobile", "tablet"]


We then specify the aggregation primitive to make where clauses for using ``where_primitives``

.. ipython:: python

    from featuretools.primitives import Count, AvgTimeBetween

    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity="customers",
                                          agg_primitives=[Count, AvgTimeBetween],
                                          where_primitives=[Count, AvgTimeBetween],
                                          trans_primitives=[])
    feature_matrix

Now, we have several new potentially useful features. For example, the two features below tell us *how many sessions a customer completed on a tablet*, and *the time between those sessions*.

.. ipython:: python

    feature_matrix[["COUNT(sessions WHERE device = tablet)", "AVG_TIME_BETWEEN(sessions.session_start WHERE device = tablet)"]]

We can see that customer who only had 0 or 1 sessions on a tablet, had ``NaN`` values for average time between such sessions.


Encoding categorical features
*****************************

Machine learning algorithms typically expect all numeric data. When Deep Feature Synthesis generates categorical features, we need to encode them.

.. ipython:: python

    from featuretools.primitives import Mode

    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity="customers",
                                          agg_primitives=[Mode],
                                          max_depth=1)

    feature_matrix

This feature matrix contains 2 categorical variables, ``zip_code`` and ``MODE(sessions.device)``. We can use the feature matrix and feature definitions to encode these categorical values. Featuretools offers functionality to apply one hot encoding to the output of DFS.

.. ipython:: python

    feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_defs)
    feature_matrix_enc

The returned feature matrix is now all numeric. Additionally, we get a new set of feature definitions that contain the encoded values.

.. ipython:: python

  print features_enc

These features can be used to calculate the same encoded values on new data. For more information on feature engineering in production, read :doc:`/guides/deployment`.


.. todos: drop contains, drop exact, max feature
