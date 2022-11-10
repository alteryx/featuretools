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
    load_weather

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

scikit-learn (BETA)
-------------------
.. autosummary::
    :toctree: generated/

    wrappers.DFSTransformer

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

Aggregation Primitives
----------------------
.. autosummary::
    :toctree: generated/

    All
    Any
    AvgTimeBetween
    Count
    CountAboveMean
    CountBelowMean
    CountGreaterThan
    CountInsideNthSTD
    CountInsideRange
    CountLessThan
    CountOutsideNthSTD
    CountOutsideRange
    Entropy
    First
    Last
    Max
    Mean
    Median
    Min
    Mode
    NMostCommon
    NumTrue
    NumUnique
    PercentTrue
    Skew
    Std
    Sum
    TimeSinceFirst
    TimeSinceLast
    Trend

Transform Primitives
--------------------
Binary Transform Primitives
***************************
.. autosummary::
    :toctree: generated/

    AddNumeric
    AddNumericScalar
    DivideByFeature
    DivideNumericScalar
    Equal
    EqualScalar
    GreaterThan
    GreaterThanEqualTo
    GreaterThanEqualToScalar
    GreaterThanScalar
    LessThan
    LessThanEqualTo
    LessThanEqualToScalar
    LessThanScalar
    ModuloByFeature
    ModuloNumeric
    ModuloNumericScalar
    MultiplyBoolean
    MultiplyNumericBoolean
    MultiplyNumericScalar
    NotEqual
    NotEqualScalar
    ScalarSubtractNumericFeature
    SubtractNumeric
    SubtractNumericScalar


Combine features
****************
.. autosummary::
    :toctree: generated/

    IsIn
    And
    Or
    Not


.. _api_ref.cumulative_features:

Cumulative Transform Primitives
*******************************
.. autosummary::
    :toctree: generated/

    Diff
    DiffDatetime
    TimeSincePrevious
    CumCount
    CumSum
    CumMean
    CumMin
    CumMax


Datetime Transform Primitives
*****************************
.. autosummary::
    :toctree: generated/

    Age
    DateToHoliday
    DateToTimeZone
    Day
    DayOfYear
    DaysInMonth
    DistanceToHoliday
    Hour
    IsFederalHoliday
    IsLeapYear
    IsLunchTime
    IsMonthEnd
    IsMonthStart
    IsQuarterEnd
    IsQuarterStart
    IsWeekend
    IsWorkingHours
    IsYearEnd
    IsYearStart
    Minute
    Month
    PartOfDay
    Quarter
    Second
    Week
    Weekday
    Year


Email and URL Transform Primitives
**********************************
.. autosummary::
    :toctree: generated/

    EmailAddressToDomain
    IsFreeEmailDomain
    URLToDomain
    URLToProtocol
    URLToTLD


Exponential Transform Primitives
********************************
.. autosummary::
    :toctree: generated/

    ExponentialWeightedAverage
    ExponentialWeightedSTD
    ExponentialWeightedVariance


General Transform Primitives
****************************
.. autosummary::
    :toctree: generated/

    Absolute
    Cosine
    IsNull
    NaturalLogarithm
    Negate
    Percentile
    RateOfChange
    Sine
    SquareRoot
    Tangent


Location Transform Primitives
*****************************
.. autosummary::
   :toctree: generated/

   CityblockDistance
   GeoMidpoint
   Haversine
   IsInGeoBox
   Latitude
   Longitude


NaturalLanguage Transform Primitives
************************************
.. autosummary::
   :toctree: generated/

   CountString
   MeanCharactersPerWord
   MedianWordLength
   NumCharacters
   NumUniqueSeparators
   NumWords
   NumberOfCommonWords
   NumberOfHashtags
   NumberOfMentions
   NumberOfUniqueWords
   NumberOfWordsInQuotes
   PunctuationCount
   TitleWordCount
   TotalWordLength
   UpperCaseCount
   WhitespaceCount


Time Series Transform Primitives
********************************
.. autosummary::
    :toctree: generated/

    Lag
    RollingCount
    RollingMax
    RollingMean
    RollingMin
    RollingOutlierCount
    RollingSTD
    RollingTrend


.. currentmodule:: nlp_primitives

.. autosummary::
   :nosignatures:

Natural Language Processing Primitives
--------------------------------------
Natural Language Processing primitives create features for textual data. For more information on how to use and install these primitives, see `here <https://github.com/alteryx/nlp_primitives>`__.

Primitives in standard install
******************************
.. autosummary::
    :toctree: generated/

    DiversityScore
    LSA
    PartOfSpeechCount
    PolarityScore
    StopwordCount

Primitives that require installing tensorflow
*********************************************
.. currentmodule:: nlp_primitives.tensorflow
.. autosummary::
    :toctree: generated/

    Elmo
    UniversalSentenceEncoder


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
    EntitySet.replace_dataframe

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

Primitive Util Methods
----------------------
.. currentmodule:: featuretools
.. autosummary::
    :toctree: generated/

    list_primitives
    summarize_primitives
