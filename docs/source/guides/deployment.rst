.. _deployment:
.. currentmodule:: featuretools

Deployment
==========

Deployment of machine learning models requires repeating feature engineering steps on new data. In some cases, these steps need to be performed in near real-time. Featuretools has capabilities to ease the deployment of feature engineering.


Saving Features
***************

First, let's build some generate some training and test data in the same format. We use a random seed to generate different data for the test.

.. note ::

    Features saved in one version of Featuretools are not guaranteed to load in another. This means the features might need to be re-created after upgrading Featuretools.

.. ipython:: python

    import featuretools as ft

    es_train = ft.demo.load_mock_customer(return_entityset=True)
    es_test = ft.demo.load_mock_customer(return_entityset=True, random_seed=33)

Now let's build some features definitions using DFS. Because we have categorical features, we also encode them with one hot encoding based on the values in the training data.

.. ipython:: python

    feature_matrix, feature_defs = ft.dfs(entityset=es_train,
                                          target_entity="customers")

    feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_defs)
    feature_matrix_enc


Now, we can use :meth:`featuretools.save_features` to save a list features to a json file

.. ipython:: python

    ft.save_features(features_enc, "feature_definitions.json")



Calculating Feature Matrix for New Data
***************************************

We can use :meth:`featuretools.load_features` to read in a list of saved features to calculate for our new entity set.

.. ipython:: python

    saved_features = ft.load_features('feature_definitions.json')

.. ipython:: python
    :suppress:

    import os
    os.remove("feature_definitions.json")


After we load the features back in, we can calculate the feature matrix.

.. ipython:: python

    feature_matrix = ft.calculate_feature_matrix(saved_features, es_test)
    feature_matrix

As you can see above, we have the exact same features as before, but calculated on using our test data.


