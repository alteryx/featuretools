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


Wrappers
~~~~~~~~
.. currentmodule:: featuretools.wrappers

Scikit-learn (BETA)
-------------------
.. autosummary::
    :toctree: generated/

    DFSTransformer



.. DeepFeatureSynthesis

Timedelta
~~~~~~~~~
.. currentmodule:: featuretools

.. autosummary::
    :toctree: generated/

    Timedelta

Time utils
~~~~~~~~~~
.. currentmodule:: featuretools

.. autosummary::
    :toctree: generated/

    make_temporal_cutoffs


Feature Primitives
~~~~~~~~~~~~~~~~~~
Primitive Types
---------------
.. currentmodule:: featuretools.primitives

.. autosummary::
    :toctree: generated/

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
    TimeSinceFirst
    NumUnique
    PercentTrue
    All
    Any
    Last
    Skew
    Trend


Transform Primitives
--------------------
Combine features
****************
.. autosummary::
    :toctree: generated/

    IsIn
    And
    Or
    Not



General Transform Primitives
****************************
.. autosummary::
    :toctree: generated/

    Absolute
    Percentile
    TimeSince

Datetime Transform Primitives
*****************************
.. autosummary::
    :toctree: generated/

    Second
    Minute
    Weekday
    IsWeekend
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

    Diff
    TimeSincePrevious
    CumCount
    CumSum
    CumMean
    CumMin
    CumMax

Text Transform Primitives
*************************
.. autosummary::
   :toctree: generated/

   NumCharacters
   NumWords

Location Transform Primitives
*****************************
.. autosummary::
   :toctree: generated/

   Latitude
   Longitude
   Haversine

Feature methods
---------------
.. currentmodule:: featuretools.feature_base
.. autosummary::
    :toctree: generated/

    FeatureBase.rename
    FeatureBase.get_depth


Feature calculation
~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: featuretools
.. autosummary::
    :toctree: generated/

    calculate_feature_matrix
    .. approximate_features

Feature encoding
~~~~~~~~~~~~~~~~~
.. currentmodule:: featuretools
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

EntitySet load and prepare data
-------------------------------
.. autosummary::
    :toctree: generated/

    EntitySet.entity_from_dataframe
    EntitySet.add_relationship
    EntitySet.normalize_entity
    EntitySet.add_interesting_values

EntitySet serialization
-------------------------------
.. currentmodule:: featuretools
.. autosummary::
    :toctree: generated/

    read_entityset

.. currentmodule:: featuretools.entityset
.. autosummary::
    :toctree: generated/

    EntitySet.to_csv
    EntitySet.to_pickle
    EntitySet.to_parquet

EntitySet query methods
-----------------------
.. autosummary::
    :toctree: generated/

    EntitySet.__getitem__
    EntitySet.find_backward_paths
    EntitySet.find_forward_paths
    EntitySet.get_forward_entities
    EntitySet.get_backward_entities

EntitySet visualization
-----------------------
.. autosummary::
    :toctree: generated/

    EntitySet.plot


Entity methods
-------------------
.. autosummary::
    :toctree: generated/

    Entity.convert_variable_type
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
    NumericTimeIndex
    Datetime
    Numeric
    Categorical
    Ordinal
    Boolean
    Text
    LatLong
    ZIPCode
    IPAddress
    FullName
    EmailAddress
    URL
    PhoneNumber
    DateOfBirth
    CountryCode
    SubRegionCode
    FilePath


Feature Selection
------------------
.. currentmodule:: featuretools.selection
.. autosummary::
    :toctree: generated/

    remove_low_information_features
