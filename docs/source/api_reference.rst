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
    get_valid_primitives


Wrappers
~~~~~~~~
.. currentmodule:: featuretools

Scikit-learn (BETA)
-------------------
.. autosummary::
    :toctree: generated/

    wrappers.DFSTransformer



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
A list of all Featuretools primitives can be obtained by visiting `primitives.featurelabs.com <https://primitives.featurelabs.com/>`__.

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
    First
    Last
    Skew
    Trend
    Entropy


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

NaturalLanguage Transform Primitives
************************************
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

Feature descriptions
~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: featuretools
.. autosummary::
    :toctree: generated/

    describe_feature

Feature visualization
~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: featuretools
.. autosummary::
    :toctree: generated/

    graph_feature

Feature encoding
~~~~~~~~~~~~~~~~~
.. currentmodule:: featuretools
.. autosummary::
    :toctree: generated/

    encode_features

Feature Selection
~~~~~~~~~~~~~~~~~
.. currentmodule:: featuretools.selection
.. autosummary::
    :toctree: generated/

    remove_low_information_features
    remove_highly_correlated_features
    remove_highly_null_features
    remove_single_value_features

Feature Matrix utils
~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: featuretools.computational_backends
.. autosummary::
    :toctree: generated/

    replace_inf_values
    

Saving and Loading Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: featuretools
.. autosummary::
    :toctree: generated/

    save_features
    load_features

.. _api_ref.dataset:

EntitySet, Relationship
~~~~~~~~~~~~~~~~~~~~~~~

Constructors
------------
.. currentmodule:: featuretools
.. autosummary::
    :toctree: generated/

    EntitySet
    Relationship

EntitySet load and prepare data
-------------------------------
.. autosummary::
    :toctree: generated/

    EntitySet.add_dataframe
    EntitySet.add_interesting_values
    EntitySet.add_last_time_indexes
    EntitySet.add_relationship
    EntitySet.add_relationships
    EntitySet.concat
    EntitySet.normalize_dataframe
    EntitySet.set_secondary_time_index
    EntitySet.update_dataframe

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
    EntitySet.get_forward_dataframes
    EntitySet.get_backward_dataframes
    EntitySet.query_by_values

EntitySet visualization
-----------------------
.. autosummary::
    :toctree: generated/

    EntitySet.plot

Relationship attributes
-----------------------
.. autosummary::
    :toctree: generated/

    Relationship.parent_column
    Relationship.child_column
    Relationship.parent_dataframe
    Relationship.child_dataframe

Data Type Util Methods
----------------------
.. currentmodule:: featuretools
.. autosummary::
    :toctree: generated/

    list_logical_types
    list_semantic_tags
