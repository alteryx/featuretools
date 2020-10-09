Feature Selection
=================

Featuretools provides users with the ability to remove features that are
unlikely to be useful in building an effective machine learning model.
Reducing the number of features in the feature matrix can both produce
better results in the model as well as reduce the computational cost
involved in prediction.

Featuretools enables users to perform feature selection on the results
of Deep Feature Synthesis with three functions:

-  ``ft.selection.remove_highly_null_features``
-  ``ft.selection.remove_single_value_features``
-  ``ft.selection.remove_highly_correlated_features``

Each of the functions takes in a calculated feature matrix and can
optionally take in the list of Feature definitions as well. If just a feature
matrix is used as input, then just a feature matrix will be returned,
but if both the matrix and the Feature definitions are used, 
then the results of feature selection will, similarly, be the
matrix and the Feature definitions.

We will describe each of these functions in depth, but first we must
create an entity set with which we can run ``ft.dfs``.

.. code:: ipython3

    import pandas as pd
    import featuretools as ft
    
    from featuretools.selection import (
        remove_low_information_features,
        remove_highly_correlated_features,
        remove_highly_null_features,
        remove_single_value_features,
    )
    
    from featuretools.primitives import NaturalLanguage
    
    
    df1 = pd.DataFrame({'id': [0, 1, 2, 3], 
                        'categories':['a','a','b','b'], 
                        'bools':[True, True, False, False],
                        'half_nulls': [None, None, 88, 100],
                        'quarter_nulls': [None, 1,1, 1],
                        'all_nulls': pd.Series([None, None, None,None], dtype='float'),
                        "diff_ints": [34, 11, 29, 91],
                       
                       })
    df2 = pd.DataFrame({
        'id': [0, 1, 2, 3],
        "first_id": [0, 1, 1, 3],
        "words": ["test", "this is a short sentence", "foo bar", "baz"],
        "corr_words": [4, 24, 7, 3],
        'corr_1': [99, 100, 77, 33],
        'corr_2': [99, 100, 77, 33],
    })
    
    entities = {
            "first": (df1, 'id'),
            "second": (df2, 'id',  None, {'words': NaturalLanguage}),
        }
    
    es = ft.EntitySet("data", entities, )
    es

Remove Highly Null Features
---------------------------

We might have a dataset with columns that have many null values, and,
after Deep Feature Synthesis, it becomes apparent that many of the
features created from those columns will also have many null values. In
this case, we might want to remove any columns whose null values pass a
certain threshold. Below is our feature matrix with such a case:

.. code:: ipython3

    
    
    fm, features = ft.dfs(entityset=es,
                              target_entity="first",
                              trans_primitives=['add_numeric'],
                              agg_primitives=[],
                              max_depth=2)
    fm

We look at the above feature matrix and decide to remove the highly null
features

.. code:: ipython3

    remove_highly_null_features(fm)

Notice that calling ``remove_highly_null_features`` didn’t remove every
column that contains a null. By default, we only remove columns where
the percentage of null values is above 95%. If we want to lower that
threshold, we can set the ``pct_null_threshold`` paramter ourselves.

.. code:: ipython3

    remove_highly_null_features(fm, pct_null_threshold=.5)

Now we’re left with a feature matrix containing mostly populated data!

Remove Single Value Features
----------------------------

Another situation we might run into is one where our calculated features
don’t have any variance. In those cases, we are likely to want to remove
the uninteresting columns. For that, we use
``remove_single_value_features``.

.. code:: ipython3

    fm, features = ft.dfs(entityset=es,
                              target_entity="first",
                              trans_primitives=['is_null'],
                              agg_primitives=[],
                              max_depth=2)
    fm

The example of using ``IsNull`` as a primitive highlights a case where
many columns all have the same value. Lets remove them:

.. code:: ipython3

    remove_single_value_features(fm)

Notice that we’ve actually lost two of the three columns with null
values because, with the function used as it is above, null values are
not considered in whether a column has only one unique value. If we’d
like to consider ``NaN`` its own value, we can set
``count_nan_as_value`` to ``True``.

.. code:: ipython3

    remove_single_value_features(fm, count_nan_as_value=True)

Remove Highly Correlated Features
---------------------------------

The last feature selection function we have allows us to remove columns
that would likely be redundant to the model we’re attempting to build by
considering the correlation between pairs of calculated features.

When two columns are determined to be highly correlated, we remove the
more complex of the two. For example, say we have two features:

-  ``col``
-  ``-(col)``

We can see that ``-(col)`` is just the negation of ``col``, and so we
can guess those columns are going to be highly correlated. ``-(col)``
has has the ``Negate`` primitive applied to it, so it is more complex
than the identity feature ``col``. Therefore, if we only want one of
``col`` and ``-(col)``, we should keep the identity feature. For
features that don’t have an obvious difference in complexity, we discard
the feature that comes later in the feature matrix.

Let’s try this out on our data:

.. code:: ipython3

    fm, features = ft.dfs(entityset=es,
                              target_entity="second",
                              trans_primitives=['negate', 'num_characters'],
                              agg_primitives=[],
                              max_depth=2)
    fm

Note that we have some pretty clear correlations here between all the
columns and their negations, but we also have ``corr_words`` that
matches the feature ``NUM_CHARACTERS(words)``.

Now, using ``remove_highly_correlated_features``,our default threshold
for correlation is 95% correlated, and we get all of the obviously
correlated features removed, leaving just the less complex features.

.. code:: ipython3

    remove_highly_correlated_features(fm)

pct_corr_threshold
^^^^^^^^^^^^^^^^^^

We can lower the threshold at which to remove correlated features if
we’d like to be more restrictive by using the ``pct_corr_threshold``
parameter.

.. code:: ipython3

    remove_highly_correlated_features(fm,pct_corr_threshold=.9)

features_to_check
^^^^^^^^^^^^^^^^^

If we only want to check a subset of columns, we can set
``features_to_check`` to the list of columns whose correlation we’d like
to check, and no columns outside of that list will be removed.

.. code:: ipython3

    remove_highly_correlated_features(fm, features_to_check=['corr_1', 'corr_2', '-(corr_1)' ,'-(corr_2)'])

features_to_keep
^^^^^^^^^^^^^^^^

To avoid having specific columns removed from the feature matrix, we can
include a list of ``features_to_keep``, and these features will not be
removed

.. code:: ipython3

    remove_highly_correlated_features(fm, features_to_keep=['-(corr_1)' ,'-(corr_2)'])
