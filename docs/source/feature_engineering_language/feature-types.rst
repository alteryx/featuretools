.. _feature-types:
.. currentmodule:: featuretools


.. ipython:: python
    :suppress:

    import featuretools as ft
    import pandas as pd

    es = ft.demo.load_mock_customer(return_entityset=True)

Feature types
==============

Featuretools groups features into four general types:

* :ref:`Identity features <feature-types.identity>`
* :ref:`Transform <feature-types.transform>` and :ref:`Cumulative Transform features <feature-types.cumulative_transform>`
* :ref:`Aggregation features <feature-types.aggregation>`
* :ref:`Direct features <feature-types.direct>`

.. _feature-types.identity:

Identity Features
-----------------
In Featuretools, each feature is defined as a combination of other features.  At the lowest level are :class:`IdentityFeature <.primitives.IdentityFeature>` features which are equal to the plain old value of a single variable.
Most of the time, identity features will be defined transparently for you, such as in the transform feature example below. They may also be defined explicitly:

.. ipython:: python

    from featuretools.primitives import Feature
    time_feature = Feature(es["transactions"]["transaction_time"])
    time_feature

Note: ``Feature`` is an alias for :class:`IdentityFeature <.primitives.IdentityFeature>` if only a single argument is provided.

.. _feature-types.direct:

Direct Features
---------------
Direct features are used to "inherit" feature values from a parent to a child entity. Suppose each event is associated with a single instance of the entity `products`.
This entity has metadata about different products, such as brand, price, etc. We can pull the brand of the product into a feature of the event entity by including the event entity as an argument to ``Feature``.
In this case, ``Feature`` is an alias for :class:`primitives.DirectFeature`:

.. ipython:: python

    from featuretools.primitives import Feature
    brand = Feature(es["products"]["brand"], es["transactions"])
    brand


.. _feature-types.transform:

Transform Features
------------------
Transform features take one or more features on an :class:`.Entity` and create a single new feature for that same entity. For example, we may want to take a fine-grained "timestamp" feature and convert it into the hour of the day in which it occurred.

.. ipython:: python

    from featuretools.primitives import Hour, Weekend
    Hour(time_feature)
    Weekend(time_feature)

The :class:`Hour <.primitives.Hour>` feature takes one parameter: the variable or feature we want to transform. If a variable is passed in, as in this case, an :class:`IdentityFeature <.primitives.IdentityFeature>` will be created automatically.

Using algebraic and boolean operations, transform features can combine other features into arbitrary expressions. For example, to determine if a given event event happened in the afternoon, we can write:

.. ipython:: python

    hour_feature = Hour(time_feature)
    after_twelve = hour_feature > 12
    after_twelve
    at_twelve = hour_feature == 12
    before_five = hour_feature <= 17
    is_afternoon = after_twelve & before_five
    is_afternoon

Aggregation Features
--------------------
Aggregation features are used to create features for a :term:`parent entity` by summarizing data from a :term:`child entity`. For example, we can create a :class:`Count <.primitives.Count>` feature which counts the total number of events for each customer:

.. ipython:: python

    from featuretools.primitives import Count
    total_events = Count(es["transactions"]["transaction_id"], es["customers"])
    fm = ft.calculate_feature_matrix([total_events], es)
    fm.head()

.. note::

    For users who have written aggregations in SQL, this concept will be familar. One key difference in featuretools is that ``GROUP BY`` and ``JOIN`` are implicit. Since the parent and child entities are specified, featuretools can infer how to group the child entity and then join the resulting aggregation back to the parent entity.

Often times, we only want to aggregate using a certain amount of previous data. For example, we might only want to count events from the past 30 days. In this case, we can provide the ``use_previous`` parameter:

.. ipython:: python

    total_events_last_30_days = Count(es["transactions"]["transaction_id"],
                                      parent_entity=es["customers"],
                                      use_previous="30 days")
    fm = ft.calculate_feature_matrix([total_events_last_30_days], es)
    fm.head()

Unlike with cumulative transform features, the ``use_previous`` parameter here is evaluated relative to instances of the parent entity, not the child entity. The above feature translates roughly to the following: "For each customer, count the events which occurred in the 30 days preceding the customer's timestamp."

Find the list of the supported aggregation features :ref:`here <api_ref.aggregation_features>`.

Where clauses
-------------
When defining aggregation or cumulative transform features, we can provide a ``where`` parameter to filter the instances we are aggregating over. Using the ``is_afternoon`` feature from :ref:`earlier <feature-types.transform>`, we can count the total number of events which occurred in the afternoon:

.. ipython:: python

    afternoon_events = Count(es["transactions"]["transaction_id"],
                         parent_entity=es["customers"],
                         where=is_afternoon).rename("afternoon_events")
    fm = ft.calculate_feature_matrix([afternoon_events], es)
    fm.head()

The where argument can be any previously-defined boolean feature. Only instances for which the where feature is True are included in the final calculation.

Cumulative Transform Features
-----------------------------
Like regular transform features, cumulative transform are features for an entity based on other features already defined on that entity. However, they differ in that they use data from many :term:`instances <instance>` to compute a single value.

Each cumulative transform feature is created with a new parameter, ``use_previous``, that takes a :class:`.Timedelta` object. This parameter specifies how long before the timestamp of each instance to look for data. Think of a cumulative transform feature like a rolling function: the feature iterates over the entity and, for each instance ``i``, aggregates data from the window defined by ``(i.timestamp - use_previous, i.timestamp]``.

Say we want to calculate the number of events per customer in the past 30 days. We can create a cumulative count feature that tallies, `for each event`, the number of events which share a customer in the 30 days preceding that event's timestamp.

.. ipython:: python

    from featuretools.primitives import CumCount
    total_events = CumCount(base_feature=es["transactions"]["transaction_id"],
                            group_feature=es["transactions"]["session_id"],
                            use_previous="1 hour")
    fm = ft.calculate_feature_matrix([total_events], es)
    fm.head()


Because they use previous data, cumulative transform features can only be defined on entities that have a time index. Find the list of available cumulative transform features :ref:`here <api_ref.cumulative_features>`.

.. _feature-types.aggregation:


Aggregations of Direct Feature
------------------------------

Composing multiple feature types is an extremely powerful abstraction that Featuretools makes simple.
For instance, we can aggregate direct features on a child entity from a different parent entity. For example, to calculate the most common brand a customer interacted with:

.. ipython:: python

    from featuretools.primitives import Mode
    brand = Feature(es["products"]["brand"], es["transactions"])
    favorite_brand = Mode(brand, parent_entity=es["customers"])
    fm = ft.calculate_feature_matrix([favorite_brand], es)
    fm.head()


.. Sliding Window Aggregation Features
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. Sometimes we would like to use more than a single aggregate value from a child entity. ``SlidingWindow`` features let us define windows of time on child entities with which to aggregate data. Unlike standard ``AggregationFeatures`` that return a single value for each parent instance, SlidingWindow features return a fixed size array of values, referred to as windows.

.. The number of windows is determined by the following parameters:

.. ====================== ===========================================
.. parameter
.. ====================== ===========================================
.. ``use_previous``        amount of previous data to use
.. ``window_size``         size of each window to aggregate
.. ``gap``                 how far apart window are from each other.
.. ====================== ===========================================

.. The number of windows is thus ``use_previous`` / (``window_size`` + ``gap``).


.. Below we define a SlidingMean feature using 6 hours of previous data and 1 hour in each window.

.. .. ipython:: python

..     from featuretools.primitives import SlidingMean

..     f = SlidingMean(es["transactions"]["amount"], es["customers"],
..                     use_previous="6 hours",
..                     window_size="1 hour")
..     f.head()

Side note: Feature equality overrides default equality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Because we can check if two features are equal (or a feature is equal to a value), we override Python's equals (`==`) operator. This means to check if two feature objects are equal (instead of their computed values in the feature matrix), we need to compare their hashes:

.. ipython:: python

    hour_feature.hash() == hour_feature.hash()
    hour_feature.hash() != hour_feature.hash()

dictionaries and sets use equality underneath, so those keys need to be hashes as well

.. ipython:: python

    myset = set()
    myset.add(hour_feature.hash())
    hour_feature.hash() in myset
    mydict = dict()
    mydict[hour_feature.hash()] = hour_feature
    hour_feature.hash() in mydict

.. _feature-types.cumulative_transform:

.. The LabelTime's time occurs before the last value in the child entity's data, so the data at 5/1/2011 is left out. The ``use_previous`` parameter does reach all the way back to the first data point at 1/1/2011, so it's left out as well.

.. We are left with a 60 day window starting at 4/1/2011, and going back to  1/31/2011, and another window that goes until the end of the ``use_previous`` Timedelta, at 1/9/2011. The first takes the mean of 20 and 40, and the second is just a single value: 10.
