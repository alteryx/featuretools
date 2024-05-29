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
    AverageCountPerUnique
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
    DateFirstEvent
    Entropy
    First
    FirstLastTimeDelta
    HasNoDuplicates
    IsMonotonicallyDecreasing
    IsMonotonicallyIncreasing
    IsUnique
    Kurtosis
    Last
    Max
    MaxConsecutiveFalse
    MaxConsecutiveNegatives
    MaxConsecutivePositives
    MaxConsecutiveTrue
    MaxConsecutiveZeros
    MaxCount
    MaxMinDelta
    Mean
    Median
    MedianCount
    Min
    MinCount
    Mode
    NMostCommon
    NMostCommonFrequency
    NUniqueDays
    NUniqueDaysOfCalendarYear
    NUniqueMonths
    NUniqueWeeks
    NumConsecutiveGreaterMean
    NumConsecutiveLessMean
    NumFalseSinceLastTrue
    NumPeaks
    NumTrue
    NumTrueSinceLastFalse
    NumUnique
    NumZeroCrossings
    PercentTrue
    PercentUnique
    Skew
    Std
    Sum
    TimeSinceFirst
    TimeSinceLast
    TimeSinceLastFalse
    TimeSinceLastMax
    TimeSinceLastMin
    TimeSinceLastTrue
    Trend
    Variance

Transform Primitives
--------------------
Binary Transform Primitives
***************************
.. autosummary::
    :toctree: generated/

    AddNumeric
    AddNumericScalar
    DivideByFeature
    DivideNumeric
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
    MultiplyNumeric
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
    CumulativeTimeSinceLastFalse
    CumulativeTimeSinceLastTrue


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
    IsFirstWeekOfMonth
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
    NthWeekOfMonth
    PartOfDay
    Quarter
    Season
    Second
    TimeSince
    Week
    Weekday
    Year


Email, URL and File Transform Primitives
****************************************
.. autosummary::
    :toctree: generated/

    EmailAddressToDomain
    FileExtension
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

    AbsoluteDiff
    Absolute
    Cosine
    IsNull
    NaturalLogarithm
    Negate
    Percentile
    PercentChange
    RateOfChange
    SameAsPrevious
    SavgolFilter
    Sine
    SquareRoot
    Tangent
    Variance

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

Name Transform Primitives
*************************
.. autosummary::
   :toctree: generated/

    FullNameToFirstName
    FullNameToLastName
    FullNameToTitle

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
   UpperCaseWordCount
   WhitespaceCount

Postal Code Primitives
**********************
.. autosummary::
    :toctree: generated/

    OneDigitPostalCode
    TwoDigitPostalCode

Time Series Transform Primitives
********************************
.. autosummary::
    :toctree: generated/

    ExpandingCount
    ExpandingMax
    ExpandingMean
    ExpandingMin
    ExpandingSTD
    ExpandingTrend
    Lag
    RollingCount
    RollingMax
    RollingMean
    RollingMin
    RollingOutlierCount
    RollingSTD
    RollingTrend


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

    get_recommended_primitives
    list_primitives
    summarize_primitives
