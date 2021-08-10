.. _glossary:
.. currentmodule:: featuretools

Glossary
========

.. glossary::
    :sorted:

    feature
        A transformation of data used for machine learning.  featuretools has a custom language for defining features as described :ref:`here <primitives>`. All features are represented by subclasses of :class:`FeatureBase`.

    feature engineering
        The process of transforming data into representations that are better for machine learning.

    cutoff time
        The last point in time data is allowed to be used when calculating a feature

    EntitySet
        A collection of dataframes and the relationships between them. Represented by the :class:`.EntitySet` class.

    instance
        Equivalent to a row in a relational database. Each dataframe has many instances, and each instance has a value for each column and feature defined on the dataframe.

    target dataframe
        The dataframe for which we will be making features

    parent dataframe
        A dataframe that is referenced by another dataframe via relationship. The "one" in a one-to-many relationship.

    child dataframe
        A dataframe that references another dataframe via relationship. The "many" in a one-to-many relationship.

    relationship
        A mapping between a parent dataframe and a child dataframe. The child dataframe must contain a column referencing the ID column on the parent dataframe. Represented by the :class:`.Relationship` class.

.. todo
.. label maker,
.. lag
..     The amount of time before the prediction time that data is used to make a prediction. This is the time between the beginning and end of a learning window.
