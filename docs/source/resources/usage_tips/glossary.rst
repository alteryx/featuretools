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

    variable
        Equivalent to a column in a relational database. Represented by the :class:`.Variable` class.


    cutoff time
        The last point in time data is allowed to be used when calculating a feature

    entity
        Equivalent to a table in relational database. Represented by the :class:`.Entity` class.

    EntitySet
        A collection of entities and the relationships between them. Represented by the :class:`.EntitySet` class.

    instance
        Equivalent to a row in a relational database. Each entity has many instances, and each instance has a value for each variable and feature defined on the entity.

    target entity
        The entity on which we will be making a features for.

    parent entity
        An entity that is referenced by another entity via relationship. The "one" in a one-to-many relationship.

    child entity
        An entity that references another entity via relationship. The "many" in a one-to-many relationship.

    relationship
        A mapping between a parent entity and a child entity. The child entity must contain a variable referencing the ID variable on the parent entity. Represented by the :class:`.Relationship` class.

.. todo
.. label maker,
.. lag
..     The amount of time before the prediction time that data is used to make a prediction. This is the time between the beginning and end of a learning window.
