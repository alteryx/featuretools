.. _api_ref:

API Reference
=============

.. currentmodule:: featuretools

Demo Datasets
~~~~~~~~~~~~~
.. currentmodule:: featuretools.demo


.. autosummary::
    :toctree: generated/

    load_retail
    load_mock_customer
    load_flight

Deep Feature Synthesis
~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: featuretools

.. autosummary::
    :toctree: generated/

    dfs

.. DeepFeatureSynthesis

Timedelta
~~~~~~~~~
.. currentmodule:: featuretools

.. autosummary::
    :toctree: generated/

    Timedelta

Feature Primitives
~~~~~~~~~~~~~~~~~~

Primitive Types
---------------
.. currentmodule:: featuretools.primitives

.. autosummary::
    :toctree: generated/

    Feature
    TransformPrimitive
    AggregationPrimitive


.. _api_ref.aggregation_features:

Primitive Creation Functions
----------------------------
.. autosummary::
    :toctree: generated/

    make_agg_primitive
    make_trans_primitive

Aggregation Primitives
----------------------
.. autosummary::
    :toctree: generated/

    Count
    Mean
    Sum
    Min
    Max
    Std
    Median
    Mode
    AvgTimeBetween
    TimeSinceLast
    NUnique
    PercentTrue
    All
    Any
    Last
    Skew
    Trend

.. _api_ref.sliding_window_features:

.. Sliding Window Features
.. -----------------------
.. .. autosummary::
..     :toctree: generated/

..     SlidingMean
..     SlidingSum
..     SlidingStd

Transform Primitives
--------------------
Combine features
****************
.. autosummary::
    :toctree: generated/

    PrimitiveBase.isin
    PrimitiveBase.AND
    PrimitiveBase.OR
    PrimitiveBase.NOT
    .. PrimitiveBase.add
    .. PrimitiveBase.subtract
    .. PrimitiveBase.multiply
    .. PrimitiveBase.divide
    .. PrimitiveBase.equal_to
    .. PrimitiveBase.not_equal_to
    .. PrimitiveBase.less_than
    .. PrimitiveBase.greater_than
    .. PrimitiveBase.less_than_equal_to
    .. PrimitiveBase.greater_than_equal_to



General Transform Primitives
****************************
.. autosummary::
    :toctree: generated/

    Absolute
    TimeSince

Datetime Transform Primitives
*****************************
.. autosummary::
    :toctree: generated/

    Second
    Minute
    Weekday
    Weekend
    Hour
    Day
    Week
    Month
    Year

.. _api_ref.cumulative_features:

Cumulative Transform Primitives
*******************************
.. autosummary::
    :toctree: generated/

    CumCount
    CumSum
    CumMean
    CumMax
    CumMin
    Diff
    TimeSincePrevious

Feature methods
---------------
.. autosummary::
    :toctree: generated/

    PrimitiveBase.head
    PrimitiveBase.rename
    PrimitiveBase.get_depth


Feature calculation
~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: featuretools.computational_backends
.. autosummary::
    :toctree: generated/

    calculate_feature_matrix
    .. approximate_features

Feature encoding
~~~~~~~~~~~~~~~~~
.. currentmodule:: featuretools.synthesis
.. autosummary::
    :toctree: generated/

    encode_features

Saving and Loading Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: featuretools
.. autosummary::
    :toctree: generated/

    save_features
    load_features

.. _api_ref.dataset:

EntitySet, Entity, Relationship, Variable Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constructors
------------
.. currentmodule:: featuretools
.. autosummary::
    :toctree: generated/

    EntitySet
    Entity
    Relationship

EntitySet attributes
--------------------
.. currentmodule:: featuretools.entityset
.. autosummary::
    :toctree: generated/

    EntitySet.id
    EntitySet.name
    EntitySet.entity_names
    EntitySet.entities
    EntitySet.relationships

EntitySet load and prepare data
-------------------------------
.. autosummary::
    :toctree: generated/

    EntitySet.entity_from_csv
    EntitySet.entity_from_dataframe
    EntitySet.add_relationship
    EntitySet.normalize_entity
    EntitySet.combine_variables
    EntitySet.add_interesting_values

EntitySet serialization
-------------------------------
.. autosummary::
    :toctree: generated/

    EntitySet.to_pickle
    EntitySet.read_pickle

EntitySet query methods
-----------------------
.. autosummary::
    :toctree: generated/

    EntitySet.__getitem__
    EntitySet.find_backward_path
    EntitySet.find_forward_path
    EntitySet.get_forward_entities
    EntitySet.get_backward_entities


Entity attributes
----------------------
.. autosummary::
    :toctree: generated/

    Entity.name
    Entity.variables
    Entity.index
    Entity.time_index


Entity methods
-------------------
.. autosummary::
    :toctree: generated/

    Entity.head
    Entity.show_instance
    Entity.is_child_of
    Entity.is_parent_of
    Entity.convert_variable_type
    Entity.has_time_index
    Entity.add_interesting_values

Relationship attributes
-----------------------
.. autosummary::
    :toctree: generated/

    Relationship.parent_variable
    Relationship.child_variable
    Relationship.parent_entity
    Relationship.child_entity

Variable types
----------------
.. currentmodule:: featuretools.variable_types
.. autosummary::
    :toctree: generated/

    Index
    Id
    TimeIndex
    DatetimeTimeIndex
    Datetime
    Numeric
    Categorical
    Ordinal
    Boolean
    Text


Feature Selection
------------------
.. currentmodule:: featuretools.selection
.. autosummary::
    :toctree: generated/

    remove_low_information_features
